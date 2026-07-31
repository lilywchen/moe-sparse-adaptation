"""Camelyon17 (WILDS) loaders — REAL histopathology batch effects, the cross-modality replication.

Same study as RxRx1, different acquisition: nuisance = HOSPITAL (5 medical centres; H&E stain colour
+ scanner differ), label = tumour vs normal (binary). Identical (x, y, site) contract — `site` is a
contiguous train-hospital index 0..K-1 (OOD hospitals -> -1), so the DANN adversary, routing-MI, and
leakage probes all work unchanged; the 4th tuple element `env` is the raw hospital id, which is
what per-environment accuracy is bucketed by (an OOD split has no seen sites, only real envs). WILDS 'official' split:
    train   -> seen hospitals                         -> train
    id_val  -> held-out patches, SEEN hospitals       -> test_within + audit
    test    -> OOD hospital (unseen)                  -> test_heldout (the headline)
    val     -> OOD hospital (unseen, for selection)   -> acc_val

Patches are 96px, resized to cfg img_size (224) so the ImageNet-pretrained backbones transfer.
Geometry-only aug (NO colour jitter — it would imitate the stain batch effect, exactly as in rxrx1).
"""
from torch.utils.data import DataLoader

from .rxrx1 import _SiteView, _rxrx1_transform


def _within_split(ds):
    """In-distribution held-out split (SEEN hospitals): prefer id_val, then id_test, else val."""
    sd = getattr(ds, "split_dict", {})
    for name in ("id_val", "id_test"):
        if name in sd:
            return name
    return "val"


def make_camelyon17_loaders(cfg):
    """Returns (train, test_within, test_heldout, audit). Mutates cfg['sites']['K'] = #train hospitals."""
    from wilds import get_dataset

    img_size = cfg.get("img_size") or cfg["model"].get("vit", {}).get("img", 224)
    ds = get_dataset(dataset="camelyon17", root_dir=cfg["data_root"],
                     download=bool(cfg.get("download", False)))
    dom_col = ds.metadata_fields.index("hospital")        # the domain column
    tf_tr, tf_ev = _rxrx1_transform(img_size, True), _rxrx1_transform(img_size, False)
    within_name = _within_split(ds)
    train_sub  = ds.get_subset("train",     transform=tf_tr)
    within_sub = ds.get_subset(within_name, transform=tf_ev)   # seen hospitals, held-out patches
    ood_sub    = ds.get_subset("test",      transform=tf_ev)   # OOD hospital

    train_doms = sorted(set(train_sub.metadata_array[:, dom_col].tolist()))
    remap = {d: i for i, d in enumerate(train_doms)}
    K = len(train_doms)
    cfg.setdefault("sites", {})["K"] = K                  # adversary n_sites + leakage chance (1/K)
    print(f"[camelyon17] {K} train hospitals (sites); |train|={len(train_sub)} "
          f"|{within_name}|={len(within_sub)} |ood_test|={len(ood_sub)}")

    bs, nw = cfg["train"]["batch_size"], cfg["train"]["num_workers"]
    mk = lambda d, sh: DataLoader(d, batch_size=bs, shuffle=sh, num_workers=nw, pin_memory=True,
                                  drop_last=sh, persistent_workers=(nw > 0))
    within = _SiteView(within_sub, dom_col, remap)
    return (
        mk(_SiteView(train_sub, dom_col, remap), True),   # train (seen hospitals)
        mk(within, False),                                # test_within (seen, held-out patches)
        mk(_SiteView(ood_sub, dom_col, remap), False),    # test_heldout (OOD hospital)
        mk(within, False),                                # audit (seen hospitals)
    )


def make_camelyon17_val_loader(cfg):
    """OOD validation loader (WILDS 'val' = held-out hospital, distinct from test). None if absent."""
    from wilds import get_dataset

    img_size = cfg.get("img_size") or cfg["model"].get("vit", {}).get("img", 224)
    ds = get_dataset(dataset="camelyon17", root_dir=cfg["data_root"], download=False)
    if "val" not in getattr(ds, "split_dict", {}):
        return None
    dom_col = ds.metadata_fields.index("hospital")
    val_sub = ds.get_subset("val", transform=_rxrx1_transform(img_size, False))
    train_doms = sorted(set(ds.get_subset("train").metadata_array[:, dom_col].tolist()))
    remap = {d: i for i, d in enumerate(train_doms)}
    bs, nw = cfg["train"]["batch_size"], cfg["train"]["num_workers"]
    print(f"[camelyon17] |ood_val|={len(val_sub)} (selection; test untouched)")
    return DataLoader(_SiteView(val_sub, dom_col, remap), batch_size=bs, shuffle=False,
                      num_workers=nw, pin_memory=True, persistent_workers=(nw > 0))
