#!/usr/bin/env python
"""Aggregate the matched random-versus-MAE ViT-Tiny fine-tuning study."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


RUNS = (
    "random_standard", "mae_standard", "mae_per_image_standard",
    "mae_per_image_lr250e6",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    args = parser.parse_args()
    root = Path(args.result_root).expanduser().resolve()
    rows, missing = [], []
    for run in args.runs:
        train_path = root / "recipe_certification" / run / "vit_tiny" / "PLATEAU_RESULT.json"
        eval_path = root / "evaluation" / run / "RESULT.json"
        if not train_path.is_file() or not eval_path.is_file():
            missing.append(run); continue
        train = json.loads(train_path.read_text()); evaluation = json.loads(eval_path.read_text())
        selected = train["selected_attempt"]
        metrics = evaluation["metrics"]
        initialization = train["initialization"]
        pretraining_normalization = initialization.get("pretraining_normalization_mode")
        if initialization["kind"] == "random":
            initialization_family = "random"
        elif pretraining_normalization == "per_image":
            initialization_family = "per_image_mae"
        else:
            initialization_family = "canonical_mae"
        rows.append({
            "run": run, "initialization": initialization["kind"],
            "initialization_family": initialization_family,
            "pretraining_normalization_mode": pretraining_normalization,
            "normalization_mode": train.get("normalization_mode", "frozen_global"),
            "learning_rate": selected["recipe"]["lr"],
            "selected_epoch": selected["best_source_iid"]["epoch"],
            "terminal_epoch": selected["terminal_epoch"],
            "train_site_top1": metrics["train"]["site_top1"],
            "train_well_top1": metrics["train"]["top1"],
            "iid_site_top1": metrics["iid_validation"]["site_top1"],
            "iid_well_top1": metrics["iid_validation"]["top1"],
            "target_site_top1": metrics["target"]["site_top1"],
            "target_well_top1": metrics["target"]["top1"],
            "iid_minus_target_site": (
                metrics["iid_validation"]["site_top1"] - metrics["target"]["site_top1"]),
            "training_hours": selected["elapsed_seconds"] / 3600.0,
            "evaluation_hours": evaluation["elapsed_seconds"] / 3600.0,
        })
    if missing:
        raise RuntimeError(f"cannot aggregate incomplete fine-tuning runs: {missing}")
    frame = pd.DataFrame(rows).sort_values("target_site_top1", ascending=False)
    analysis = root / "analysis"; figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    frame.to_csv(analysis / "finetune_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = frame.sort_values("target_site_top1")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    palette = {"random": "#8c8c8c", "canonical_mae": "#3976b8",
               "per_image_mae": "#2a9d6f"}
    colors = [palette[value] for value in ordered.initialization_family]
    ax.barh(ordered.run, ordered.target_site_top1, color=colors)
    ax.set_xlabel("Held-out-batch perturbation site top-1")
    ax.set_title("End-to-end ViT-Tiny: random versus MAE initialization")
    fig.tight_layout(); fig.savefig(figures / "target_accuracy.png", dpi=180); plt.close(fig)

    best = frame.iloc[0]
    standard = frame.set_index("run")
    canonical_delta = (
        standard.loc["mae_standard", "target_site_top1"]
        - standard.loc["random_standard", "target_site_top1"]
        if {"mae_standard", "random_standard"} <= set(standard.index) else float("nan"))
    per_image_delta = (
        standard.loc["mae_per_image_standard", "target_site_top1"]
        - standard.loc["random_standard", "target_site_top1"]
        if {"mae_per_image_standard", "random_standard"} <= set(standard.index)
        else float("nan"))
    contrasts = []
    if canonical_delta == canonical_delta:
        contrasts.append(
            "The direct canonical-MAE-minus-random target-site difference is "
            f"<strong>{canonical_delta:+.4f}</strong>.")
    if per_image_delta == per_image_delta:
        contrasts.append(
            "The per-image-MAE-minus-random target-site difference is "
            f"<strong>{per_image_delta:+.4f}</strong>.")
    named_contrasts = (
        ("random_per_image_standard", "random_global_anchor",
         "Supervised per-image normalization effect under random initialization"),
        ("mae_canonical_global_anchor", "random_global_anchor",
         "Canonical-MAE effect with frozen-global supervised normalization"),
        ("mae_canonical_per_image_standard", "random_per_image_standard",
         "Canonical-MAE effect with per-image supervised normalization"),
        ("mae_per_image_global_anchor", "random_global_anchor",
         "Per-image-MAE effect with frozen-global supervised normalization"),
        ("mae_per_image_matched_standard", "random_per_image_standard",
         "Matched per-image-MAE effect with per-image supervised normalization"),
        ("mae_per_image_matched_standard", "mae_canonical_per_image_standard",
         "Per-image versus canonical MAE under per-image supervised normalization"),
    )
    for numerator, denominator, label in named_contrasts:
        if {numerator, denominator} <= set(standard.index):
            delta = (standard.loc[numerator, "target_site_top1"]
                     - standard.loc[denominator, "target_site_top1"])
            contrasts.append(f"{label}: <strong>{delta:+.4f}</strong>.")
    contrast_html = " ".join(contrasts)
    report = f"""<!doctype html><html><head><meta charset='utf-8'><title>ViT-Tiny fine-tuning</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1300px}}table{{border-collapse:collapse}}
th,td{{padding:.4rem;border:1px solid #ddd}}th{{background:#f2f2f2}}img{{max-width:100%}}</style>
</head><body><h1>Matched ViT-Tiny fine-tuning</h1>
<p>All checkpoints were selected using source-IID accuracy. Target batches were loaded only
after selection. {contrast_html}</p>
<p>Best arm: <code>{html.escape(str(best.run))}</code> at {best.target_site_top1:.4f}.</p>
<img src='figures/target_accuracy.png' alt='Target accuracy'>
{frame.to_html(index=False, float_format=lambda value: f'{value:.5f}')}</body></html>"""
    (analysis / "finetune_report.html").write_text(report)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
