#!/usr/bin/env python
"""Aggregate the frozen RxRx1 HUVEC batch-effect wave into statistics and a visual report."""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.data.rxrx1_huvec_batch import matched_pseudo_target_wells, unit


def atomic_text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(value); os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True))


def _metric(result, role, unit_name="site"):
    return float(result["metrics"][role][unit_name]["top1"])


def result_table(result_root):
    root = Path(result_root).expanduser().resolve()
    manifest_path = root / "wave_manifest.json"
    if not manifest_path.is_file():
        return pd.DataFrame(), None, None
    manifest = json.loads(manifest_path.read_text())
    registry = json.loads((root / "study_registry.json").read_text())
    splits = {row["split_id"]: row for row in registry["splits"]}
    rows = []
    for spec in manifest["runs"]:
        run_dir = root / "runs" / spec["run_id"]
        result_path, status_path = run_dir / "RESULT.json", run_dir / "STATUS.json"
        payload = None
        state = "pending"
        if result_path.is_file():
            payload = json.loads(result_path.read_text()); state = "complete"
        elif status_path.is_file():
            payload = json.loads(status_path.read_text()); state = payload.get("state", "unknown")
        split = splits[spec["split_id"]]
        row = {
            "run_id": spec["run_id"], "stage": spec["stage"], "model": spec["model"],
            "split_id": spec["split_id"], "kind": split["kind"],
            "target_experiment": int(split["target_experiments"][0]),
            "composition": split.get("composition", ""),
            "difficulty": float(split["cell_dino_difficulty"]), "state": state,
            "epoch": None, "best_epoch": None, "iid_site": None, "target_site": None,
            "iid_well": None, "target_well": None, "elapsed_hours": None,
        }
        if payload:
            row["epoch"] = payload.get("terminal_epoch", payload.get("epoch"))
            row["best_epoch"] = payload.get("best_epoch")
            row["elapsed_hours"] = (float(payload.get("elapsed_seconds", 0)) / 3600
                                    if payload.get("elapsed_seconds") is not None else None)
        if state == "complete":
            row.update({
                "iid_site": _metric(payload, "iid_validation", "site"),
                "target_site": _metric(payload, "target", "site"),
                "iid_well": _metric(payload, "iid_validation", "well"),
                "target_well": _metric(payload, "target", "well"),
                "train_site": _metric(payload, "train", "site"),
                "site_gap": float(payload["site_iid_to_target_gap"]),
                "well_gap": float(payload["well_iid_to_target_gap"]),
            })
        rows.append(row)
    return pd.DataFrame(rows), manifest, registry


def print_status(table, manifest):
    if manifest is None:
        print("[pending] the frozen wave has not been prepared")
        return
    counts = table.state.value_counts().to_dict()
    print(f"[status] {counts.get('complete', 0)}/{len(table)} complete "
          f"| training={counts.get('training', 0)} failed={counts.get('failed', 0)} "
          f"interrupted={counts.get('interrupted', 0)} pending={counts.get('pending', 0)}")
    active = table[table.state.isin(["training", "failed", "interrupted"])]
    for row in active.itertuples(index=False):
        suffix = f" epoch={row.epoch}" if row.epoch is not None else ""
        print(f"  {row.state:11s} {row.run_id}{suffix}")


def _permutation_correlation(x, y, permutations=10000, seed=20260815):
    x, y = np.asarray(x, float), np.asarray(y, float)
    observed = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.asarray([np.corrcoef(x, rng.permutation(y))[0, 1]
                       for _ in range(int(permutations))])
    p = float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null)))
    return observed, p


def _bootstrap_mean(values, iterations=10000, seed=20260815):
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    draws = rng.choice(values, (int(iterations), len(values)), replace=True).mean(1)
    return [float(values.mean()), float(np.quantile(draws, .025)),
            float(np.quantile(draws, .975))]


def _savefig(path):
    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()


def _cell_dino_geometry(registry, analysis, figures):
    meta = pd.read_parquet(registry["well_metadata"])
    embedding = np.load(registry["well_cell_dino"]).astype(np.float32)
    pool = set(map(int, registry["diagnostic_source_pool"]))
    keep = meta.experiment.astype(int).isin(pool).to_numpy()
    meta = meta.loc[keep].reset_index(drop=True); embedding = unit(embedding[keep])
    label_mean = pd.DataFrame(embedding).groupby(meta.label.to_numpy()).transform("mean").to_numpy()
    residual = unit(embedding - label_mean)

    # Batch decoding is cross-validated by perturbation: held-out labels never form centroids.
    label_fold = meta.label.to_numpy(np.int64) % 5
    predicted, truth = [], []
    experiment_order = sorted(pool)
    for fold in range(5):
        train = label_fold != fold; test = ~train
        centroids = np.stack([unit(residual[train & (meta.experiment == exp)].mean(0)[None])[0]
                              for exp in experiment_order])
        predicted.extend(np.asarray(experiment_order)[np.argmax(residual[test] @ centroids.T, 1)])
        truth.extend(meta.loc[test, "experiment"].astype(int))
    decoder = float(np.mean(np.asarray(predicted) == np.asarray(truth)))

    sample = np.linspace(0, len(residual) - 1, min(12000, len(residual))).astype(int)
    centered = residual[sample] - residual[sample].mean(0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[:2].T
    plt.figure(figsize=(8.4, 6.2))
    scatter = plt.scatter(projected[:, 0], projected[:, 1],
                          c=meta.experiment.to_numpy()[sample], s=4, alpha=.38,
                          cmap="turbo", rasterized=True)
    plt.colorbar(scatter, label="experiment / batch")
    plt.xlabel("residual PC1"); plt.ylabel("residual PC2")
    plt.title("Cell-DINO after subtracting perturbation identity")
    _savefig(figures / "cell_dino_class_residual_pca.png")
    return {"cross_label_batch_decoder_accuracy": decoder,
            "chance_accuracy": 1 / len(experiment_order), "experiments": len(experiment_order)}


def aggregate(result_root, require_complete=True):
    root = Path(result_root).expanduser().resolve()
    table, manifest, registry = result_table(root)
    if manifest is None:
        raise FileNotFoundError("wave_manifest.json does not exist")
    incomplete = table[table.state != "complete"]
    if len(incomplete) and require_complete:
        print_status(table, manifest)
        raise RuntimeError(f"cannot finalize: {len(incomplete)} of {len(table)} runs incomplete")
    complete = table[table.state == "complete"].copy()
    analysis = root / "analysis"; figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    complete.to_csv(analysis / "run_metrics.csv", index=False)
    if len(incomplete):
        print_status(table, manifest); return

    diagnostic = complete[(complete.model == "vit_tiny") &
                          (complete.kind == "diagnostic_loo")].sort_values("target_experiment")
    pseudo_rows, class_rows = [], []
    for row in diagnostic.itertuples(index=False):
        run_dir = root / "runs" / row.run_id
        well = pd.read_parquet(run_dir / "well_predictions.parquet")
        site = pd.read_parquet(run_dir / "site_predictions.parquet")
        pseudo = matched_pseudo_target_wells(
            well[well.role == "iid_validation"], well[well.role == "target"],
            row.target_experiment, manifest["pseudo_target_resamples"])
        for resample, group in pseudo.groupby("resample"):
            selected = set(group.pseudo_well_id)
            pseudo_site = site[(site.role == "iid_validation") & site.well_id.isin(selected)]
            target_site = site[site.role == "target"]
            pseudo_rows.append({
                "run_id": row.run_id, "target_experiment": row.target_experiment,
                "resample": int(resample), "target_site": float(target_site.correct_top1.mean()),
                "pseudo_site": float(pseudo_site.correct_top1.mean()),
                "target_well": float(group.target_correct_top1.mean()),
                "pseudo_well": float(group.pseudo_correct_top1.mean()),
            })
        target_by_class = site[site.role == "target"].groupby("label").correct_top1.mean()
        class_rows.extend({"target_experiment": row.target_experiment, "label": int(label),
                           "site_top1": float(score)}
                          for label, score in target_by_class.items())
    pseudo = pd.DataFrame(pseudo_rows); pseudo.to_csv(analysis / "matched_iid_null.csv", index=False)
    pseudo_summary = pseudo.groupby("target_experiment").agg(
        pseudo_site_mean=("pseudo_site", "mean"), pseudo_site_sd=("pseudo_site", "std"),
        pseudo_well_mean=("pseudo_well", "mean"), pseudo_well_sd=("pseudo_well", "std"),
    ).reset_index()
    diagnostic = diagnostic.merge(pseudo_summary, on="target_experiment", validate="one_to_one")
    diagnostic["matched_site_degradation"] = diagnostic.pseudo_site_mean - diagnostic.target_site
    diagnostic["matched_well_degradation"] = diagnostic.pseudo_well_mean - diagnostic.target_well
    diagnostic.to_csv(analysis / "diagnostic_folds.csv", index=False)

    difficulty_r, difficulty_p = _permutation_correlation(
        diagnostic.difficulty, diagnostic.matched_site_degradation)
    statistics = {
        "folds": len(diagnostic),
        "site_matched_degradation_mean_ci95": _bootstrap_mean(
            diagnostic.matched_site_degradation),
        "well_matched_degradation_mean_ci95": _bootstrap_mean(
            diagnostic.matched_well_degradation),
        "difficulty_vs_site_degradation_pearson_r": difficulty_r,
        "difficulty_permutation_p_two_sided": difficulty_p,
        "cell_dino_geometry": _cell_dino_geometry(registry, analysis, figures),
    }

    x = np.arange(len(diagnostic)); width = .38
    plt.figure(figsize=(10, 5.5)); plt.bar(x - width/2, diagnostic.pseudo_site_mean, width,
        label="matched source-IID pseudo-target"); plt.bar(x + width/2, diagnostic.target_site,
        width, label="held-out batch")
    plt.xticks(x, diagnostic.target_experiment, rotation=0); plt.ylabel("site top-1 accuracy")
    plt.xlabel("held-out experiment"); plt.title("Same perturbations, matched wells, different batch")
    plt.legend(); _savefig(figures / "matched_iid_vs_target.png")

    ordered = diagnostic.sort_values("matched_site_degradation")
    plt.figure(figsize=(10, 5.5)); colors = np.where(ordered.matched_site_degradation >= 0,
                                                    "#c94c4c", "#3d7ea6")
    plt.bar(np.arange(len(ordered)), ordered.matched_site_degradation, color=colors)
    plt.axhline(0, color="black", lw=.8); plt.xticks(np.arange(len(ordered)),
        ordered.target_experiment); plt.ylabel("matched IID minus target site accuracy")
    plt.xlabel("held-out experiment"); plt.title("Batch degradation varies substantially")
    _savefig(figures / "batch_degradation_waterfall.png")

    plt.figure(figsize=(7, 5.4)); plt.scatter(diagnostic.difficulty,
        diagnostic.matched_site_degradation, s=58)
    fit = np.polyfit(diagnostic.difficulty, diagnostic.matched_site_degradation, 1)
    grid = np.linspace(diagnostic.difficulty.min(), diagnostic.difficulty.max(), 100)
    plt.plot(grid, np.polyval(fit, grid), color="#c94c4c")
    for row in diagnostic.itertuples():
        plt.annotate(str(row.target_experiment), (row.difficulty, row.matched_site_degradation),
                     xytext=(3, 3), textcoords="offset points", fontsize=8)
    plt.xlabel("Cell-DINO same-perturbation displacement")
    plt.ylabel("matched IID minus target site accuracy")
    plt.title(f"Does embedding shift predict degradation? r={difficulty_r:.2f}, p={difficulty_p:.3g}")
    _savefig(figures / "degradation_vs_difficulty.png")

    class_matrix = pd.DataFrame(class_rows).pivot(
        index="label", columns="target_experiment", values="site_top1")
    plt.figure(figsize=(10, 8)); plt.imshow(class_matrix.to_numpy(), aspect="auto",
        interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="site accuracy"); plt.xticks(np.arange(len(class_matrix.columns)),
        class_matrix.columns); plt.xlabel("held-out experiment"); plt.ylabel("perturbation class")
    plt.title("Which perturbations fail in which batches?")
    _savefig(figures / "class_by_batch_accuracy.png")

    controlled = complete[(complete.model == "vit_tiny") &
                          (complete.kind == "source_composition")].copy()
    order = ["near", "diverse", "far"]
    plt.figure(figsize=(8.5, 5.5))
    for target, group in controlled.groupby("target_experiment"):
        group = group.set_index("composition").reindex(order)
        plt.plot(order, group.target_site, marker="o", label=f"target {target}")
    plt.ylabel("target site accuracy"); plt.xlabel("equal-size source composition")
    plt.title("Source-batch composition changes generalization")
    plt.legend(); _savefig(figures / "source_composition_intervention.png")

    capacity = complete[complete.stage == "capacity_mechanism"].copy()
    capacity_pivot = capacity.pivot(index="target_experiment", columns="model",
                                    values=["target_site", "iid_site"])
    comparison_rows = []
    for target in capacity_pivot.index:
        dense = float(diagnostic.set_index("target_experiment").loc[target, "target_site"])
        moe = float(capacity_pivot.loc[target, ("target_site", "vit_tiny_moe")])
        matched = float(capacity_pivot.loc[target, ("target_site", "vit_tiny_dense_matched")])
        comparison_rows.append({"target_experiment": int(target), "dense": dense,
            "moe": moe, "dense_total_matched": matched, "moe_minus_dense": moe-dense,
            "moe_minus_total_matched": moe-matched})
    comparison = pd.DataFrame(comparison_rows); comparison.to_csv(
        analysis / "capacity_comparison.csv", index=False)
    x = np.arange(len(comparison)); width = .25
    plt.figure(figsize=(8.5, 5.4)); plt.bar(x-width, comparison.dense, width, label="dense ViT-Tiny")
    plt.bar(x, comparison.moe, width, label="4-expert sparse MoE")
    plt.bar(x+width, comparison.dense_total_matched, width, label="dense total-param matched")
    plt.xticks(x, comparison.target_experiment); plt.xlabel("held-out experiment")
    plt.ylabel("target site accuracy"); plt.title("MoE versus compute- and capacity-oriented controls")
    plt.legend(); _savefig(figures / "moe_capacity_comparison.png")

    plt.figure(figsize=(7, 5.4)); plt.scatter(diagnostic.target_site, diagnostic.target_well, s=58)
    limit = [min(diagnostic.target_site.min(), diagnostic.target_well.min()),
             max(diagnostic.target_site.max(), diagnostic.target_well.max())]
    plt.plot(limit, limit, "--", color="gray"); plt.xlabel("site accuracy (primary)")
    plt.ylabel("two-site mean-logit well accuracy (secondary)")
    plt.title("Pooling sites is reported, but does not define the result")
    _savefig(figures / "site_vs_well_accuracy.png")

    statistics["moe_mean_gain_over_dense"] = float(comparison.moe_minus_dense.mean())
    statistics["moe_mean_gain_over_total_parameter_matched_dense"] = float(
        comparison.moe_minus_total_matched.mean())
    atomic_json(analysis / "statistics.json", statistics)

    figure_cards = "".join(
        f'<figure><img src="figures/{html.escape(path.name)}"><figcaption>{html.escape(path.stem.replace("_", " "))}</figcaption></figure>'
        for path in sorted(figures.glob("*.png")))
    diagnostic_html = diagnostic[["target_experiment", "difficulty", "pseudo_site_mean",
        "target_site", "matched_site_degradation", "target_well"]].round(4).to_html(index=False)
    report = f"""<!doctype html><html><head><meta charset="utf-8"><title>RxRx1 HUVEC batch-effect study</title>
<style>body{{font:16px system-ui;max-width:1180px;margin:40px auto;padding:0 24px;color:#18212b}}
h1,h2{{letter-spacing:-.02em}} .lead{{font-size:20px;line-height:1.55}} .flow{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:28px 0}}
.node{{padding:14px 18px;border:1px solid #9fb2c4;border-radius:12px;background:#f5f8fb}} .arrow{{font-size:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:22px}} figure{{margin:0;border:1px solid #dbe3ea;border-radius:12px;padding:12px}}
img{{width:100%;height:auto}} figcaption{{text-transform:capitalize;color:#526170;padding:8px}} table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid #ddd;padding:6px;text-align:right}} th:first-child,td:first-child{{text-align:left}} .callout{{background:#fff6dd;padding:18px;border-left:5px solid #e0a92e}}</style></head><body>
<h1>RxRx1 HUVEC: batch degradation and MoE pilot</h1>
<p class="lead">This study asks whether a model that learns perturbations in one set of HUVEC experiments loses accuracy on another experiment containing the same perturbations—and whether that loss tracks measured embedding shift or changes with source-batch composition.</p>
<div class="flow"><div class="node">16 diagnostic experiments</div><div class="arrow">→</div><div class="node">hold out one whole experiment</div><div class="arrow">→</div><div class="node">train on sites from the other experiments</div><div class="arrow">→</div><div class="node">select only on source-IID sites</div><div class="arrow">→</div><div class="node">open target once</div></div>
<div class="callout"><b>Leakage rule.</b> The eight original primary_fold0 targets remain sealed. Within every diagnostic fold, target pixels are not loaded until the checkpoint has been selected. Site prediction is primary; averaging the two site logits within a well is a secondary view.</div>
<h2>What each arm establishes</h2><p><b>16 leave-one-experiment-out folds</b> estimate batch-specific degradation. <b>50 matched IID pseudo-targets per fold</b> control perturbation and site-count composition. <b>12 equal-size source interventions</b> test whether near/diverse/far batch composition causally changes transfer. <b>Four difficulty anchors</b> compare dense ViT-Tiny, sparse MoE, and a dense total-parameter-matched control under the same training and checkpoint rule.</p>
<h2>Key frozen statistics</h2><pre>{html.escape(json.dumps(statistics, indent=2))}</pre>
<h2>Diagnostic folds</h2>{diagnostic_html}<h2>Figures</h2><div class="grid">{figure_cards}</div>
</body></html>"""
    atomic_text(analysis / "REPORT.html", report)
    print(json.dumps({"state": "aggregated", "runs": len(complete),
        "report": str(analysis / "REPORT.html"), "statistics": statistics}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.status:
        print_status(*result_table(args.result_root)[:2]); return
    aggregate(args.result_root, require_complete=not args.allow_partial)


if __name__ == "__main__":
    main()
