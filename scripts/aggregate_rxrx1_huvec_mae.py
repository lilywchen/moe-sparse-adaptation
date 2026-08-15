#!/usr/bin/env python
"""Aggregate the 14-run ORCD MAE grid into a compact table, figures, and HTML."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


RUNS = (
    "vit_tiny", "vit_micro", "tiny_mask50", "tiny_mask90", "micro_mask50",
    "micro_mask90", "tiny_source4", "tiny_source8", "micro_source4",
    "micro_source8", "tiny_per_image", "micro_per_image", "tiny_noaug",
    "micro_noaug",
)


def _pretrain_dir(root, run):
    return root / ("runs" if run in ("vit_tiny", "vit_micro") else "grid") / run


def aggregate(result_root):
    root = Path(result_root).expanduser().resolve()
    rows = []
    missing = []
    for run in RUNS:
        pretrain_path = _pretrain_dir(root, run) / "RESULT.json"
        evaluation_path = root / "evaluation" / run / "RESULT.json"
        if not pretrain_path.is_file() or not evaluation_path.is_file():
            missing.append(run); continue
        pretrain = json.loads(pretrain_path.read_text())
        evaluation = json.loads(evaluation_path.read_text())
        config = pretrain["config"]
        rows.append({
            "run": run, "model": config["model"], "mask_ratio": config["mask_ratio"],
            "source_experiments": config.get("source_experiment_count", 16),
            "normalization": config.get("normalization_mode", "frozen_global"),
            "augmentation": config.get("train_augmentation", True),
            "epochs": pretrain["epochs_completed"], "stop_reason": pretrain["stop_reason"],
            "best_reconstruction": pretrain["best_validation_reconstruction_loss"],
            "iid_prototype_top1": evaluation["prototype_retrieval"]["iid_validation"]["top1"],
            "target_prototype_top1": evaluation["prototype_retrieval"]["target"]["top1"],
            "iid_ridge_top1": evaluation["ridge_linear_probe"]["iid_validation"]["top1"],
            "target_ridge_top1": evaluation["ridge_linear_probe"]["target"]["top1"],
            "target_ridge_mrr": evaluation["ridge_linear_probe"]["target"][
                "mean_reciprocal_rank"],
            "batch_probe_top1": evaluation["source_batch_probe"]["iid_accuracy"],
            "pretrain_hours": pretrain["elapsed_seconds"] / 3600.0,
            "evaluation_hours": evaluation["elapsed_seconds"] / 3600.0,
        })
    if missing:
        raise RuntimeError(f"cannot aggregate incomplete MAE runs: {missing}")
    frame = pd.DataFrame(rows).sort_values("target_ridge_top1", ascending=False)
    analysis = root / "analysis"; figures = analysis / "figures"
    analysis.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    frame.to_csv(analysis / "mae_grid_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for model, group in frame.groupby("model"):
        ax.scatter(group.batch_probe_top1, group.target_ridge_top1, label=model, s=60)
        for row in group.itertuples():
            ax.annotate(row.run, (row.batch_probe_top1, row.target_ridge_top1), fontsize=7)
    ax.set_xlabel("Source-batch predictability from frozen embedding")
    ax.set_ylabel("Held-out-batch perturbation ridge-probe top-1")
    ax.legend(); fig.tight_layout()
    fig.savefig(figures / "target_accuracy_vs_batch_encoding.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ordered = frame.sort_values("target_ridge_top1")
    ax.barh(ordered.run, ordered.target_ridge_top1)
    ax.set_xlabel("Held-out-batch perturbation ridge-probe top-1")
    fig.tight_layout(); fig.savefig(figures / "target_accuracy_by_run.png", dpi=180); plt.close(fig)

    best = frame.iloc[0]
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>RxRx1 HUVEC MAE grid</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1400px}} table{{border-collapse:collapse}}
th,td{{padding:.4rem;border:1px solid #ddd}} th{{background:#f2f2f2;position:sticky;top:0}}
img{{max-width:100%;height:auto}} code{{background:#f4f4f4;padding:.1rem .25rem}}
</style></head><body><h1>RxRx1 HUVEC MAE grid</h1>
<p>Fourteen one-seed, target-sealed pretraining runs. Checkpoints were selected only by fixed-mask
reconstruction validation. Perturbation labels were introduced afterward for frozen prototype
retrieval and ridge linear probing.</p>
<p><strong>Best target ridge probe:</strong> <code>{html.escape(str(best.run))}</code>
({best.target_ridge_top1:.4f}). Batch probe is diagnostic, not a removal objective.</p>
<img src='figures/target_accuracy_by_run.png' alt='Target accuracy by run'>
<img src='figures/target_accuracy_vs_batch_encoding.png' alt='Accuracy versus batch encoding'>
{frame.to_html(index=False, float_format=lambda value: f'{value:.5f}')}
</body></html>"""
    (analysis / "mae_grid_report.html").write_text(document)
    print(frame.to_string(index=False))
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    aggregate(parser.parse_args().result_root)


if __name__ == "__main__":
    main()
