#!/usr/bin/env python
"""CCAS aggregation.

Everything here is PAIRED. Each Stage-1 cell is run at the same seeds as its depth-matched
dense-wide control and its frozen-router counterpart, from the identical checkpoint, data order
and init, so the informative quantity is the per-seed difference, not two independent means:

    conditional gain = OOD(learned MoE) - OOD(depth-matched dense-wide)    same placement, seed
    routing gain     = OOD(learned MoE) - OOD(frozen-router MoE)           same cell, seed
    capacity gain    = OOD(learned MoE) - OOD(original dense)              same seed
    route reliance   = OOD(learned routes) - OOD(randomized routes)        within one run

STAGE GATE. Stage 1 and 2 are decided on the OOD VALIDATION split; the OOD test split is not
looked at until the Stage-3 confirmatory runs. This script therefore reports `acc_selection`
(= OOD val) by default and REFUSES to print test numbers unless invoked with --stage3, which is
also the only mode in which the test column is written to the report.

The plan reports FACTOR EFFECTS, not just the winning cell: main effects and two-way interactions
are estimated by least squares over the 36-cell design with sum-to-zero coding, and uncertainty
comes from a bootstrap that resamples SEEDS (the unit that is paired) rather than runs.

    python scripts/aggregate_ccas.py                 # Stage 1/2 view (OOD val only)
    python scripts/aggregate_ccas.py --stage3        # confirmatory view (reveals OOD test)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(os.environ.get("MOE_RESULTS", "./RESULTS/ccas"))
FACTORS = ["placement", "routing_unit", "geometry", "pressure"]
LEVELS = {"placement": ["early", "middle", "late"], "routing_unit": ["image", "token"],
          "geometry": ["linear", "cosine"],
          "pressure": ["canonical", "route", "output"]}
N_BOOT = 5000
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- loading
def load(results=RESULTS):
    """Every result JSON as one row. Legacy runs (no `acc_selection`) fall back to acc_heldout."""
    rows = []
    for p in sorted(Path(results).rglob("*.json")):
        if p.name.endswith(".trainlog.jsonl") or p.name == "ccas_summary.json":
            continue
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if "variant" not in r:
            continue
        if r.get("acc_selection") is None:
            # pre-stage-gate runs recorded only acc_heldout / acc_val
            r["acc_selection"] = r.get("acc_val", r.get("acc_heldout"))
            r.setdefault("selection_split", "legacy")
        if r.get("acc_selection") is None:
            continue
        if "pressure" not in r:
            r["pressure"] = ("route" if r.get("balance") in
                             ("within_batch", "within_environment") else "canonical")
        r.pop("config", None)                       # keep the frame flat and printable
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- statistics
def paired_bootstrap(diffs_by_seed, n_boot=N_BOOT, rng=RNG):
    """Mean difference + percentile CI, resampling SEEDS (the paired unit).

    diffs_by_seed: 1-D array of per-seed paired differences. With the 3 Stage-1 seeds this CI is
    wide by construction -- that is the honest width, and it is exactly why Stage 3 uses 5 fresh
    seeds before anything is claimed.
    """
    d = np.asarray([x for x in diffs_by_seed if x is not None and np.isfinite(x)], dtype=float)
    if d.size == 0:
        return dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan, p_sign=np.nan)
    if d.size == 1:
        return dict(n=1, mean=float(d[0]), lo=np.nan, hi=np.nan, p_sign=np.nan)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    n_pos = int((d > 0).sum())
    return dict(n=int(d.size), mean=float(d.mean()),
                lo=float(np.percentile(means, 2.5)), hi=float(np.percentile(means, 97.5)),
                p_sign=_sign_test_p(n_pos, int((d != 0).sum())))


def _sign_test_p(n_pos, n):
    """Two-sided exact sign test. Nonparametric, makes no normality claim at n=3."""
    if n == 0:
        return float("nan")
    from math import comb
    tail = sum(comb(n, k) for k in range(0, min(n_pos, n - n_pos) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def cluster_bootstrap_env(per_env_acc, per_env_n, n_boot=1000, rng=RNG):
    """CI for one run's accuracy, resampling ENVIRONMENTS (experiments / hospitals) with weights.

    The plan is explicit that uncertainty is clustered by experiment (RxRx1) or patient/slide
    (Camelyon17), not by image: images inside one experiment are not independent draws.
    """
    envs = sorted(set(per_env_acc) & set(per_env_n))
    if len(envs) < 2:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n_env=len(envs))
    a = np.array([float(per_env_acc[e]) for e in envs])
    w = np.array([float(per_env_n[e]) for e in envs])
    idx = rng.integers(0, len(envs), size=(n_boot, len(envs)))
    boots = (a[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    return dict(mean=float((a * w).sum() / w.sum()), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), n_env=len(envs))


def _codes(series, levels):
    """Sum-to-zero (effect) coding: k levels -> k-1 columns, coefficients are deviations from the
    grand mean, so main effects stay interpretable in the presence of interactions."""
    k = len(levels)
    M = np.zeros((len(series), k - 1))
    for i, v in enumerate(series):
        j = levels.index(v)
        if j < k - 1:
            M[i, j] = 1.0
        else:
            M[i, :] = -1.0
    return M


def factorial_fit(df, y="conditional_gain", factors=FACTORS, interactions=True):
    """Least-squares factorial model of the response over the four design factors.

    Returns a tidy frame of effects. Main effects are reported as the marginal deviation of each
    level from the grand mean; two-way interactions as the extra product-term coefficient. This is
    the analysis the plan asks for -- a design principle, not a winning cell.
    """
    df = df.dropna(subset=[y])
    if df.empty:
        return pd.DataFrame()
    cols, names = [np.ones((len(df), 1))], ["intercept"]
    blocks = {}
    for f in factors:
        lv = [l for l in LEVELS[f] if l in set(df[f])]
        if len(lv) < 2:
            continue
        M = _codes(list(df[f]), lv)
        blocks[f] = (M, lv)
        cols.append(M)
        names += [f"{f}[{l}]" for l in lv[:-1]]
    if interactions:
        fs = list(blocks)
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                A, la = blocks[fs[i]]
                B, lb = blocks[fs[j]]
                P = np.concatenate([A[:, [a]] * B[:, [b]] for a in range(A.shape[1])
                                    for b in range(B.shape[1])], axis=1)
                cols.append(P)
                names += [f"{fs[i]}[{la[a]}]:{fs[j]}[{lb[b]}]"
                          for a in range(A.shape[1]) for b in range(B.shape[1])]
    X = np.concatenate(cols, axis=1)
    yv = df[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    dof = max(len(yv) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)
    out = pd.DataFrame({"term": names, "estimate": beta, "se": se})
    out["t"] = out.estimate / out.se.replace(0, np.nan)
    # the omitted last level of each factor is minus the sum of the others -- report it too
    extra = []
    for f, (_, lv) in blocks.items():
        est = out[out.term.str.startswith(f"{f}[") & ~out.term.str.contains(":")]
        if len(est):
            extra.append({"term": f"{f}[{lv[-1]}]", "estimate": -est.estimate.sum(),
                          "se": np.nan, "t": np.nan})
    if extra:
        out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)
    out["r2"] = 1 - float(resid @ resid) / max(float(((yv - yv.mean()) ** 2).sum()), 1e-12)
    return out


# --------------------------------------------------------------------------- contrasts
def paired_contrasts(df, metric="acc_selection"):
    """Per-(cell, seed) paired differences against each control. Returns the MoE frame augmented
    with conditional_gain / routing_gain / capacity_gain, plus a note of unmatched cells."""
    moe = df[df.variant == "moe"].copy()
    notes = []
    if moe.empty:
        return moe, notes

    wide = df[df.variant == "dense_wide"].copy()
    if not wide.empty:
        # Route-level pressure has no operation on a dense model, so it shares the canonical
        # dense comparator. Output pressure is matched DANN-for-DANN.
        moe["dense_pressure"] = np.where(moe.pressure == "output", "output", "canonical")
        wide["dense_pressure"] = wide.pressure
        w = wide.groupby(["dataset", "placement", "dense_pressure", "seed"])[metric].mean()
        w = w.rename("dense_wide")
        moe = moe.join(w, on=["dataset", "placement", "dense_pressure", "seed"])
        moe["conditional_gain"] = moe[metric] - moe["dense_wide"]
        miss = int(moe["dense_wide"].isna().sum())
        if miss:
            notes.append(f"{miss} MoE runs have no depth-matched dense_wide at the same seed")
    else:
        notes.append("no dense_wide runs: conditional gain is undefined (the PRIMARY contrast)")

    frozen = df[df.variant == "moe_frozen"]
    if not frozen.empty:
        keys = ["dataset"] + FACTORS + ["seed"]
        f = frozen.groupby(keys)[metric].mean().rename("moe_frozen")
        moe = moe.join(f, on=keys)
        moe["routing_gain"] = moe[metric] - moe["moe_frozen"]
    else:
        notes.append("no moe_frozen runs yet (Stage 2 control)")

    orig = df[df.variant == "original"]
    if not orig.empty:
        o = orig.groupby(["dataset", "seed"])[metric].mean().rename("original")
        moe = moe.join(o, on=["dataset", "seed"])
        moe["capacity_gain"] = moe[metric] - moe["original"]
    return moe, notes


def gain_table(moe, col, metric_name):
    """Per-cell paired gain with a seed bootstrap CI and an exact sign test."""
    if col not in moe.columns:
        return pd.DataFrame()
    rows = []
    for keys, g in moe.groupby(["dataset"] + FACTORS):
        st = paired_bootstrap(g[col].to_numpy())
        rows.append(dict(zip(["dataset"] + FACTORS, keys), **{
            "n_seeds": st["n"], metric_name: round(st["mean"], 4),
            "ci_lo": round(st["lo"], 4), "ci_hi": round(st["hi"], 4),
            "p_sign": round(st["p_sign"], 3) if np.isfinite(st["p_sign"]) else np.nan}))
    t = pd.DataFrame(rows).sort_values(metric_name, ascending=False)
    return t


# --------------------------------------------------------------------------- audits
def budget_audit(df, tol_pct=0.1):
    """Per-run and per-dataset check that MoE and dense-wide hold the SAME total parameters, with
    the only difference being the router (E*d + 1). A violation invalidates the comparison."""
    lines, ok = [], True
    if "total_params" not in df.columns:
        return ["no total_params recorded -- cannot audit the budget"], False
    for dset, sub in df.groupby("dataset"):
        pm = sub[sub.variant.isin(["moe", "moe_frozen"])]["total_params"]
        pw = sub[sub.variant == "dense_wide"]["total_params"]
        if pm.nunique() > 1:
            lines.append(f"  {dset}: MoE totals are not constant across cells "
                         f"({pm.min()}..{pm.max()}) [VIOLATION]"); ok = False
        if len(pm) and len(pw):
            d = 100.0 * (pw.min() - pm.min()) / pm.min()
            flag = "OK" if abs(d) < tol_pct else "VIOLATION"
            ok &= abs(d) < tol_pct
            lines.append(f"  {dset}: dense_wide vs moe = {d:+.4f}%  (tolerance {tol_pct}%) [{flag}]")
            rp = sub[sub.variant == "moe"]["router_params"]
            if len(rp):
                lines.append(f"      difference should be exactly the router: "
                             f"{int(pm.min() - pw.min()):+d} params vs router {int(rp.min())}")
        else:
            lines.append(f"  {dset}: missing one side (moe={len(pm)}, dense_wide={len(pw)})")
    return lines, ok


def coverage(df):
    """How much of the predeclared Stage-1 grid actually exists, per dataset."""
    rows = []
    for dset, sub in df.groupby("dataset"):
        m = sub[sub.variant == "moe"]
        rows.append({"dataset": dset, "moe_runs": len(m),
                     "cells_seen": m.groupby(FACTORS).ngroups if len(m) else 0,
                     "cells_expected": 36,
                     "seeds": sorted(sub.seed.unique().tolist()),
                     "dense_wide": int((sub.variant == "dense_wide").sum()),
                     "original": int((sub.variant == "original").sum()),
                     "moe_frozen": int((sub.variant == "moe_frozen").sum())})
    return pd.DataFrame(rows)


def leakage_audit(df, stage3):
    """Refuse to be the instrument of test-set contamination."""
    msgs = []
    if "test_evaluated" in df.columns:
        n_test = int(df["test_evaluated"].fillna(False).astype(bool).sum())
        if n_test and not stage3:
            msgs.append(f"NOTE: {n_test} runs evaluated the OOD test split. Those numbers are "
                        f"withheld here; rerun with --stage3 only for the confirmatory analysis.")
    if "selection_split" in df.columns:
        bad = df[df.selection_split.astype(str).str.startswith("ood_test")]
        if len(bad):
            msgs.append(f"WARNING: {len(bad)} runs selected on the test split (no val split "
                        f"available for {sorted(bad.dataset.unique())}).")
        legacy = int((df.selection_split == "legacy").sum())
        if legacy:
            msgs.append(f"NOTE: {legacy} pre-stage-gate runs; their selection metric was "
                        f"reconstructed as acc_val, falling back to acc_heldout.")
    return msgs


def env_breakdown_audit(df):
    """Catch the per-environment breakdown collapsing to a single bucket.

    PLAN.md's uncertainty on a single run is a CLUSTER BOOTSTRAP OVER ENVIRONMENTS. That estimator
    needs >= 2 environments; with one bucket it silently returns NaN and `worst_env_*` becomes
    identically equal to overall accuracy — a vacuous metric that still prints like a real one.

    Historically this happened for a specific reason worth naming: per-environment accuracy was
    keyed on the TRAIN-remapped site index, which is the sentinel -1 for every row of an OOD split.
    Every OOD run therefore reported `{-1: n}`. The loaders now emit the raw environment id as a
    4th tuple element; this audit exists so a regression cannot pass unnoticed again.
    """
    msgs = []
    for col in ("per_env_val", "per_env_heldout"):
        if col not in df.columns:
            continue
        sentinel, single = [], []
        for rid, d in zip(df.get("run_id", df.index), df[col]):
            if not isinstance(d, dict) or not d:
                continue
            keys = {str(k) for k in d}
            if keys == {"-1"}:
                sentinel.append(rid)
            elif len(keys) < 2:
                single.append(rid)
        if sentinel:
            msgs.append(
                f"ERROR: {len(sentinel)} runs report `{col}` as the single sentinel bucket "
                f"'-1' (e.g. {sentinel[0]}). The environment id never reached evaluate(), so "
                f"worst-environment accuracy is vacuous and the cluster bootstrap over "
                f"environments cannot be computed. These runs must be re-evaluated.")
        if single:
            msgs.append(
                f"WARNING: {len(single)} runs report a single environment in `{col}` "
                f"(e.g. {single[0]}); the cluster bootstrap over environments is a point "
                f"estimate for them.")
    return msgs


# --------------------------------------------------------------------------- report
def build_report(df, stage3=False):
    out = []
    P = out.append
    metric = "acc_selection"
    P(f"# CCAS aggregation — {len(df)} runs")
    P("")
    P(f"Selection metric: `{metric}` (OOD validation). "
      f"{'OOD TEST REVEALED (Stage 3).' if stage3 else 'OOD test withheld (Stage 1/2 gate).'}")
    P("")
    for m in leakage_audit(df, stage3) + env_breakdown_audit(df):
        P(f"- {m}")
    P("")

    P("## Coverage")
    P("")
    P(coverage(df).to_string(index=False))
    P("")

    P("## Variant summary (OOD validation accuracy)")
    P("")
    g = df.groupby(["dataset", "variant"])[metric]
    P(pd.DataFrame({"n": g.size(), "mean": g.mean().round(4), "sd": g.std().round(4)}).to_string())
    P("")

    moe, notes = paired_contrasts(df, metric)
    for n in notes:
        P(f"- {n}")
    P("")

    for col, name, blurb in [
        ("conditional_gain", "conditional_gain",
         "MoE - depth-matched dense-wide at the SAME total parameters (>0 favours conditionality)"),
        ("routing_gain", "routing_gain",
         "learned - frozen router (>0 means the learned assignment policy, not just sparsity, matters)"),
        ("capacity_gain", "capacity_gain",
         "MoE - original dense (confounds added capacity with conditionality; secondary)")]:
        t = gain_table(moe, col, name)
        if t.empty:
            continue
        P(f"## {name}")
        P("")
        P(blurb)
        P("")
        n_seeds = int(t["n_seeds"].max()) if "n_seeds" in t else 0
        if n_seeds <= 3:
            P(f"At {n_seeds} seeds the exact sign test cannot go below p={2/2**n_seeds:.2f}, so "
              f"`p_sign` here separates consistent from inconsistent cells and is NOT evidence of "
              f"significance. Stage 3 supplies the 5 fresh seeds that carry the claim.")
            P("")
        P(t.head(24).to_string(index=False))
        P("")

    if "conditional_gain" in moe.columns and moe["conditional_gain"].notna().any():
        P("## Factor effects on conditional gain")
        P("")
        P("Sum-to-zero least squares over the 36-cell design; `estimate` is the deviation of that "
          "level from the grand mean, in accuracy points.")
        P("")
        for dset, sub in moe.groupby("dataset"):
            fit = factorial_fit(sub, "conditional_gain")
            if fit.empty:
                continue
            main = fit[~fit.term.str.contains(":") & (fit.term != "intercept")]
            inter = fit[fit.term.str.contains(":")]
            P(f"### {dset}  (grand mean {float(fit.loc[fit.term=='intercept','estimate'].iloc[0]):+.4f}, "
              f"R² {float(fit.r2.iloc[0]):.3f})")
            P("")
            P(main.drop(columns=["r2"]).round(4).to_string(index=False))
            P("")
            if len(inter):
                top = inter.reindex(inter.estimate.abs().sort_values(ascending=False).index).head(6)
                P("Largest two-way interactions:")
                P("")
                P(top.drop(columns=["r2"]).round(4).to_string(index=False))
                P("")

    mech_cols = [c for c in ["route_reliance", "randomized_routes_acc", "routing_mi_site",
                             "routing_mi_class", "routing_entropy", "experts_used",
                             "site_leakage", "class_decodability", "worst_env_val",
                             "degradation_gap"] if c in df.columns]
    if mech_cols and not moe.empty:
        P("## Mechanism (MoE cells, measured on the selection split)")
        P("")
        P(moe.groupby(["dataset"] + FACTORS)[mech_cols].mean().round(3).to_string())
        P("")

    P("## Parameter budget audit")
    P("")
    lines, ok = budget_audit(df)
    for l in lines:
        P(l)
    P("")
    P(f"budget audit: {'PASS' if ok else 'FAIL — the fixed-budget claim is not currently supported'}")
    P("")

    if stage3:
        test = df[df.get("test_evaluated", pd.Series(False, index=df.index)).fillna(False).astype(bool)]
        if len(test):
            P("## Confirmatory: OOD TEST")
            P("")
            gt = test.groupby(["dataset", "variant"])["acc_heldout"]
            P(pd.DataFrame({"n": gt.size(), "mean": gt.mean().round(4),
                            "sd": gt.std().round(4)}).to_string())
            P("")
            mt, _ = paired_contrasts(test, "acc_heldout")
            t = gain_table(mt, "conditional_gain", "conditional_gain_test")
            if not t.empty:
                P(t.to_string(index=False))
                P("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--stage3", action="store_true",
                    help="reveal the OOD test split (confirmatory analysis only)")
    ap.add_argument("--out", default=None, help="markdown report path (default <results>/ccas_report.md)")
    args = ap.parse_args()

    res = Path(args.results)
    df = load(res)
    if df.empty:
        print(f"[aggregate_ccas] no results in {res} yet.")
        return
    pd.set_option("display.width", 220, "display.max_columns", 60)

    report = build_report(df, stage3=args.stage3)
    print(report)

    res.mkdir(parents=True, exist_ok=True)
    df.to_csv(res / "ccas_summary.csv", index=False)
    moe, _ = paired_contrasts(df, "acc_selection")
    if not moe.empty:
        moe.to_csv(res / "ccas_paired.csv", index=False)
        eff = []
        for dset, sub in moe.groupby("dataset"):
            f = factorial_fit(sub, "conditional_gain")
            if not f.empty:
                f.insert(0, "dataset", dset)
                eff.append(f)
        if eff:
            pd.concat(eff, ignore_index=True).to_csv(res / "ccas_effects.csv", index=False)
    out = Path(args.out) if args.out else res / "ccas_report.md"
    out.write_text(report)
    print(f"\n[written] {res/'ccas_summary.csv'}, {res/'ccas_paired.csv'}, {out}")


if __name__ == "__main__":
    main()
