#!/usr/bin/env python
"""Aggregate the focused ViT-Tiny MAE study, including a random encoder control."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


RUNS = (
    "random_init", "vit_tiny", "tiny_mask50", "tiny_mask90", "tiny_source4",
    "tiny_source8", "tiny_per_image", "tiny_noaug",
)


def _pretrain_dir(root, run):
    if run == "random_init":
        return None
    return root / ("runs" if run == "vit_tiny" else "grid") / run


def aggregate(result_root):
    root = Path(result_root).expanduser().resolve()
    rows, missing = [], []
    for run in RUNS:
        evaluation_path = root / "evaluation_tiny" / run / "RESULT.json"
        pretrain_dir = _pretrain_dir(root, run)
        pretrain_path = pretrain_dir / "RESULT.json" if pretrain_dir else None
        if not evaluation_path.is_file() or (pretrain_path and not pretrain_path.is_file()):
            missing.append(run)
            continue
        evaluation = json.loads(evaluation_path.read_text())
        pretrain = json.loads(pretrain_path.read_text()) if pretrain_path else None
        config = pretrain["config"] if pretrain else evaluation["pretraining_run"]
        rows.append({
            "run": run,
            "initialization": evaluation.get("initialization", "mae"),
            "mask_ratio": config.get("mask_ratio"),
            "source_experiments": config.get("source_experiment_count", 16),
            "normalization": config.get("normalization_mode", "frozen_global"),
            "augmentation": config.get("train_augmentation"),
            "epochs": pretrain.get("epochs_completed") if pretrain else 0,
            "best_reconstruction": (
                pretrain.get("best_validation_reconstruction_loss") if pretrain else None),
            "iid_prototype_top1": evaluation["prototype_retrieval"]["iid_validation"]["top1"],
            "target_prototype_top1": evaluation["prototype_retrieval"]["target"]["top1"],
            "iid_ridge_top1": evaluation["ridge_linear_probe"]["iid_validation"]["top1"],
            "target_ridge_top1": evaluation["ridge_linear_probe"]["target"]["top1"],
            "iid_minus_target_ridge": (
                evaluation["ridge_linear_probe"]["iid_validation"]["top1"]
                - evaluation["ridge_linear_probe"]["target"]["top1"]),
            "target_ridge_mrr": evaluation["ridge_linear_probe"]["target"][
                "mean_reciprocal_rank"],
            "batch_probe_top1": evaluation["source_batch_probe"]["iid_accuracy"],
            "pretrain_hours": pretrain["elapsed_seconds"] / 3600.0 if pretrain else 0.0,
            "evaluation_hours": evaluation["elapsed_seconds"] / 3600.0,
        })
    if missing:
        raise RuntimeError(f"cannot aggregate incomplete Tiny evaluations: {missing}")

    frame = pd.DataFrame(rows).sort_values("target_ridge_top1", ascending=False)
    analysis = root / "analysis_tiny"; figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frame.to_csv(analysis / "tiny_mae_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = frame.sort_values("target_ridge_top1")
    colors = ["#8c8c8c" if value == "random" else "#3976b8"
              for value in ordered.initialization]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(ordered.run, ordered.target_ridge_top1, color=colors)
    ax.set_xlabel("Held-out-batch perturbation ridge-probe top-1")
    ax.set_title("Frozen ViT-Tiny representations")
    fig.tight_layout(); fig.savefig(figures / "target_accuracy_by_run.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(frame.batch_probe_top1, frame.target_ridge_top1, c=[
        "#8c8c8c" if value == "random" else "#3976b8" for value in frame.initialization])
    for row in frame.itertuples():
        ax.annotate(row.run, (row.batch_probe_top1, row.target_ridge_top1), fontsize=8)
    ax.set_xlabel("Source-batch predictability from frozen embedding")
    ax.set_ylabel("Held-out-batch perturbation ridge-probe top-1")
    fig.tight_layout(); fig.savefig(figures / "accuracy_vs_batch_encoding.png", dpi=180)
    plt.close(fig)

    best = frame.iloc[0]
    random_row = frame[frame.run == "random_init"].iloc[0]
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Focused RxRx1 HUVEC ViT-Tiny MAE study</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1400px}} table{{border-collapse:collapse}}
th,td{{padding:.4rem;border:1px solid #ddd}} th{{background:#f2f2f2;position:sticky;top:0}}
img{{max-width:100%;height:auto}} code{{background:#f4f4f4;padding:.1rem .25rem}}
</style></head><body><h1>Focused RxRx1 HUVEC ViT-Tiny MAE study</h1>
<p>One seeded, target-sealed comparison using the same ViT-Tiny architecture throughout.
MAE checkpoints were selected only by source reconstruction validation. Probe regularization
was selected only on source-IID wells. The held-out target batches were scored afterward.</p>
<p><strong>Best target ridge probe:</strong> <code>{html.escape(str(best.run))}</code>
({best.target_ridge_top1:.4f}); random encoder control {random_row.target_ridge_top1:.4f}.</p>
<img src='figures/target_accuracy_by_run.png' alt='Target accuracy by run'>
<img src='figures/accuracy_vs_batch_encoding.png' alt='Accuracy versus batch encoding'>
{frame.to_html(index=False, float_format=lambda value: f'{value:.5f}')}
</body></html>"""
    (analysis / "tiny_mae_report.html").write_text(document)
    print(frame.to_string(index=False))
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    aggregate(parser.parse_args().result_root)


if __name__ == "__main__":
    main()
