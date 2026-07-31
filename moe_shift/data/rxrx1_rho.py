"""RxRx1 with an ENGINEERED experiment↔siRNA correlation (ρ) — on the REAL images/batches.

RxRx1 is natively ρ≈0 (every experiment runs the full siRNA library), which makes it the clean
"pure liability" anchor. To study the asset↔liability *transition*, we engineer a controllable
correlation between the batch (experiment) and the label (siRNA) IN THE TRAINING SET ONLY, while
keeping the images and batch effect completely real. The OOD test (unseen experiments) is left
untouched — so a model that exploits the engineered train correlation is punished OOD, exactly the
spurious-correlation regime (cf. Waterbirds / Colored-MNIST, but the shift here is real).

THE CONTROL THAT MAKES IT RIGOROUS: every ρ level uses the SAME number of images per class
(`per_class`) and the SAME total N. We only change *which experiments* a class's images are drawn
from. So ρ is the only thing that varies across the sweep — not data size, not class balance.

Mechanism: split the K train experiments into G contiguous groups and the C siRNAs into G groups;
class-group g is "preferred" by experiment-group g. For each class we draw `per_class` images, a
fraction β from its preferred experiment-group and the rest elsewhere, with
    β = 1/G + ρ·(1 − 1/G)      ρ=0 → β=1/G (≈ uniform, no correlation) ; ρ=1 → β=1 (max correlation).

`select_rho_indices` is pure NumPy (no torch/wilds) so it is unit-tested without the cluster.
`make_rxrx1_rho_loaders` wraps it into the repo's 4-loader contract, reusing rxrx1.py for the
(unchanged) eval/audit loaders. Engaged only when cfg['rho_engineer']['enabled'] is true.
"""
import numpy as np


def _contiguous_groups(ids_sorted, n_groups):
    """Map each unique id to a balanced contiguous group 0..n_groups-1."""
    n = len(ids_sorted)
    return {v: min(n_groups - 1, i * n_groups // n) for i, v in enumerate(ids_sorted)}


def _nmi(a, b):
    """Normalized mutual information in [0,1] between two integer labelings (0 = independent)."""
    a, b = np.asarray(a), np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    if len(ua) < 2 or len(ub) < 2:
        return 0.0
    ia = {v: i for i, v in enumerate(ua)}
    ib = {v: i for i, v in enumerate(ub)}
    C = np.zeros((len(ua), len(ub)))
    for x, y in zip(a, b):
        C[ia[x], ib[y]] += 1
    P = C / C.sum()
    Pi, Pj = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    outer = Pi @ Pj
    m = P > 0
    mi = float(np.sum(P[m] * np.log(P[m] / outer[m])))
    Ha = float(-np.sum(Pi[Pi > 0] * np.log(Pi[Pi > 0])))
    Hb = float(-np.sum(Pj[Pj > 0] * np.log(Pj[Pj > 0])))
    denom = np.sqrt(Ha * Hb)
    return mi / denom if denom > 0 else 0.0


def select_rho_indices(exp, label, rho, groups=4, per_class=8, seed=0, correlated=True):
    """Pick training positions so experiment-group correlates with class-group at strength `rho`,
    holding `per_class` images per class (hence total N and class balance) fixed across ρ.

    Args:
        exp:   int array [N] — experiment id per train image (ideally contiguous 0..K-1).
        label: int array [N] — siRNA class per train image (0..C-1).
        rho:   target correlation knob in [0,1] (0 = none/uniform, 1 = maximal).
        groups, per_class, seed: see module docstring.
        correlated: if True, each class concentrates into ITS class-group's experiments (creates the
            experiment↔siRNA correlation). If False (the SHUFFLED CONTROL), each class concentrates
            into a RANDOM experiment-group at the SAME β — identical concentration / N / balance, but
            no systematic mapping, so NMI≈0. Comparing correlated vs shuffled at the same ρ isolates
            the *correlation* effect from the *concentration / data-structure* effect.
    Returns:
        (selected_positions int array, stats dict). stats['nmi_group'] is the REALIZED
        experiment-group↔class-group NMI on the selection — rises with rho when correlated, ≈0 when not.
    """
    exp, label = np.asarray(exp), np.asarray(label)
    rng = np.random.default_rng(seed)
    uexp, uclass = np.unique(exp), np.unique(label)
    G = int(groups)
    eg_map = _contiguous_groups(list(uexp), G)
    cg_map = _contiguous_groups(list(uclass), G)
    exp_g = np.array([eg_map[e] for e in exp])
    beta = 1.0 / G + float(rho) * (1.0 - 1.0 / G)

    # which experiment-group each class concentrates into:
    if correlated:
        target_g = {c: cg_map[c] for c in uclass}                  # aligned to class-group -> correlation
    else:
        rg = np.random.default_rng(int(seed) + 9973)               # independent random group per class
        target_g = {c: int(rg.integers(0, G)) for c in uclass}     # -> matched concentration, NMI≈0

    sel, realized = [], []
    for c in uclass:
        gc = target_g[c]
        pos = np.where(label == c)[0]
        pref = pos[exp_g[pos] == gc]
        nonpref = pos[exp_g[pos] != gc]
        k = min(per_class, len(pos))
        n_pref = min(int(round(beta * k)), len(pref))
        n_non = k - n_pref
        if n_non > len(nonpref):                       # not enough non-preferred -> backfill preferred
            n_non = len(nonpref)
            n_pref = min(k - n_non, len(pref))
        chosen = np.concatenate([
            rng.choice(pref, size=n_pref, replace=False) if n_pref > 0 else np.empty(0, int),
            rng.choice(nonpref, size=n_non, replace=False) if n_non > 0 else np.empty(0, int),
        ]).astype(int)
        sel.append(chosen)
        realized.append(len(chosen))

    selected = np.sort(np.concatenate(sel)).astype(int)
    sel_cg = np.array([cg_map[l] for l in label[selected]])
    stats = {
        "rho_target": float(rho), "beta": float(beta), "groups": G, "per_class": int(per_class),
        "correlated": bool(correlated), "n_selected": int(selected.size),
        "per_class_min": int(np.min(realized)), "per_class_max": int(np.max(realized)),
        "per_class_mean": float(np.mean(realized)),
        "nmi_group": _nmi(exp_g[selected], sel_cg),     # realized correlation (rises with rho)
    }
    return selected, stats


def make_rxrx1_rho_loaders(cfg):
    """4-loader contract (train, within, heldout, audit) with a ρ-engineered TRAIN split.

    Eval/audit loaders are identical to the ρ≈0 setup (reused from rxrx1.make_rxrx1_loaders) — only
    the train loader is replaced by the correlation-engineered subsample. Reads cfg['shift']['rho']
    (target) and cfg['rho_engineer'] = {groups, per_class, seed}.
    """
    from torch.utils.data import DataLoader, Subset
    from wilds import get_dataset
    from .rxrx1 import make_rxrx1_loaders, _SiteView, _rxrx1_transform

    # eval/audit (+ sites.K) exactly as the uniform case; we discard only its train loader.
    _train_full, within, heldout, audit = make_rxrx1_loaders(cfg)

    re = cfg.get("rho_engineer", {})
    rho = float(cfg.get("shift", {}).get("rho", 0.0))
    img_size = cfg.get("img_size") or cfg["model"].get("vit", {}).get("img", 256)
    ds = get_dataset(dataset="rxrx1", root_dir=cfg["data_root"], download=False)
    exp_col = ds.metadata_fields.index("experiment")
    train_sub = ds.get_subset("train", transform=_rxrx1_transform(img_size, True))

    raw_exp = train_sub.metadata_array[:, exp_col].numpy()
    train_exps = sorted(set(raw_exp.tolist()))
    remap = {e: i for i, e in enumerate(train_exps)}
    exp_contig = np.array([remap[e] for e in raw_exp])
    labels = train_sub.y_array.numpy()

    correlated = re.get("correlated", True)
    correlated = (correlated.lower() in ("1", "true", "yes")) if isinstance(correlated, str) else bool(correlated)
    selected, stats = select_rho_indices(
        exp_contig, labels, rho,
        groups=int(re.get("groups", 4)), per_class=int(re.get("per_class", 8)),
        seed=int(re.get("seed", 0)), correlated=correlated)
    print(f"[rxrx1-rho] target ρ={rho} correlated={correlated} -> realized group-NMI={stats['nmi_group']:.3f}  "
          f"|train|={stats['n_selected']} (per_class={stats['per_class']}, G={stats['groups']})")

    bs, nw = cfg["train"]["batch_size"], cfg["train"]["num_workers"]
    train = DataLoader(Subset(_SiteView(train_sub, exp_col, remap), selected.tolist()),
                       batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True,
                       drop_last=True, persistent_workers=(nw > 0))
    return train, within, heldout, audit
