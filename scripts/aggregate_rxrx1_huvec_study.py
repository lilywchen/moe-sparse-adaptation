#!/usr/bin/env python
"""Aggregate the one-seed RxRx1 HUVEC systematic study into readable tables and figures."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def load_rows(result_root):
    root = Path(result_root)
    manifest = _json(root / "wave_manifest.json") or {"runs": []}
    rows = []
    for spec in manifest.get("runs", []):
        result = _json(root / "runs" / f"{spec['run_id']}.json")
        rows.append({"spec": spec, "result": result})
    return root, manifest, rows


def status_table(result_root):
    root, _manifest, rows = load_rows(result_root)
    by_stage = []
    for stage in ("canary", "F_G", "H", "I", "J"):
        selected = [row for row in rows if row["spec"]["stage"] == stage]
        complete = sum(row["result"] is not None for row in selected)
        certified = sum(bool((row["result"] or {}).get(
            "canary_passed", (row["result"] or {}).get("training_certified", False)))
                        for row in selected)
        by_stage.append((stage, complete, len(selected), certified))
    lines = [f"RxRx1 HUVEC fast study — {sum(r['result'] is not None for r in rows)}/{len(rows)} runs complete",
             "stage     complete  certified", "--------- --------- ----------"]
    lines.extend(f"{stage:<9} {complete:>3}/{total:<3}   {certified:>3}"
                 for stage, complete, total, certified in by_stage)
    failures = sorted((root / "failures").glob("*.json")) if (root / "failures").is_dir() else []
    if failures:
        lines += ["", f"FAILURES: {len(failures)}"] + [f"- {path.stem}" for path in failures]
    report = root / "analysis" / "REPORT.md"
    if report.is_file():
        lines += ["", f"Report: {report}"]
    return "\n".join(lines)


def _target_rows(rows):
    output = []
    for row in rows:
        result, spec = row["result"], row["spec"]
        if not result or result.get("canary"):
            continue
        for experiment, metrics in result["target"]["per_experiment"].items():
            output.append({
                "run_id": result["run_id"], "stage": spec["stage"], "model": result["model"],
                "split_id": result["split_id"], "split_kind": result["split_kind"],
                "difficulty_tier": result["difficulty_tier"],
                "target_experiment": int(experiment), "target_top1": metrics["top1"],
                "target_top5": metrics["top5"], "mean_rank": metrics["mean_rank"],
                "iid_top1": result["iid_validation"]["top1"],
                "train_top1": result["train"]["top1"],
                "iid_to_target_gap": result["iid_validation"]["top1"] - metrics["top1"],
                "cell_dino_difficulty": result["target_difficulty"][experiment],
                "raw_qc_difficulty": result["raw_qc_target_difficulty"][experiment],
                "observed_target_labels": result["target_label_coverage"][experiment][
                    "observed_labels"],
                "source_matched_labels": result["target_label_coverage"][experiment][
                    "source_matched_labels"],
                "target_label_fraction": result["target_label_coverage"][experiment]["fraction"],
                "training_certified": result["training_certified"],
                "total_params": result["model_audit"]["total_params"],
                "best_epoch": result["best_epoch"],
            })
    return pd.DataFrame(output)


def _slope(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def _corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def should_run_parameter_match(result_root):
    _, _, rows = load_rows(result_root)
    target = _target_rows(rows)
    primary = target[(target.split_kind == "primary") &
                     target.model.isin(["vit_tiny", "vit_tiny_moe"])]
    pivot = primary.pivot_table(index="target_experiment", columns="model",
                                values="target_top1", aggfunc="first")
    if not {"vit_tiny", "vit_tiny_moe"} <= set(pivot.columns) or len(pivot) != 24:
        return False, {"reason": "primary dense/MoE results incomplete"}
    gain = pivot.vit_tiny_moe - pivot.vit_tiny
    return bool(gain.mean() > 0), {
        "mean_moe_minus_dense": float(gain.mean()),
        "median_moe_minus_dense": float(gain.median()),
        "positive_targets": int((gain > 0).sum()), "n_targets": len(gain),
    }


def aggregate(result_root, require_complete=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, _manifest, rows = load_rows(result_root)
    expected = [row for row in rows if row["spec"]["stage"] != "I" or
                (root / "runs" / f"{row['spec']['run_id']}.json").exists()]
    incomplete = [row["spec"]["run_id"] for row in expected if row["result"] is None]
    if require_complete and incomplete:
        raise RuntimeError(f"cannot finalize; incomplete runs: {incomplete}")
    analysis = root / "analysis"; figures = analysis / "figures"
    analysis.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    target = _target_rows(rows)
    target.to_csv(analysis / "raw_image_target_metrics.csv", index=False)

    contrasts = []
    for model, group in target.groupby("model"):
        natural = group[group.split_kind == "primary"]
        contrasts.append({
            "model": model, "n_target_rows": len(group),
            "mean_target_top1": float(group.target_top1.mean()),
            "mean_iid_top1": float(group.iid_top1.mean()),
            "cell_dino_difficulty_accuracy_correlation": _corr(
                natural.cell_dino_difficulty, natural.target_top1),
            "cell_dino_difficulty_accuracy_slope": _slope(
                natural.cell_dino_difficulty, natural.target_top1),
            "raw_qc_difficulty_accuracy_correlation": _corr(
                natural.raw_qc_difficulty, natural.target_top1),
        })
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(analysis / "model_contrasts.csv", index=False)

    primary = target[(target.split_kind == "primary") &
                     target.model.isin(["vit_tiny", "vit_tiny_moe"])]
    pivot = primary.pivot_table(index=["target_experiment", "cell_dino_difficulty"],
                                columns="model", values="target_top1", aggfunc="first").reset_index()
    if {"vit_tiny", "vit_tiny_moe"} <= set(pivot.columns):
        pivot["moe_gain"] = pivot.vit_tiny_moe - pivot.vit_tiny
        pivot.to_csv(analysis / "moe_paired_target_gains.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.axhline(0, color="black", linewidth=1)
        ax.scatter(pivot.cell_dino_difficulty, pivot.moe_gain)
        ax.set_xlabel("Cell-DINO matched target difficulty")
        ax.set_ylabel("MoE − dense well accuracy")
        ax.set_title("Does MoE advantage increase with batch difficulty?")
        fig.tight_layout(); fig.savefig(figures / "moe_gain_vs_difficulty.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model, group in target[target.split_kind == "primary"].groupby("model"):
        ax.scatter(group.cell_dino_difficulty, group.target_top1, label=model, alpha=0.75)
    ax.set_xlabel("Cell-DINO matched target difficulty"); ax.set_ylabel("Well-level top-1")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figures / "raw_model_accuracy_vs_difficulty.png", dpi=180); plt.close(fig)

    controlled = target[target.split_kind == "controlled"]
    if not controlled.empty:
        order = ["low", "medium", "high"]
        fig, ax = plt.subplots(figsize=(9, 6))
        for model, group in controlled.groupby("model"):
            means = group.groupby("difficulty_tier").target_top1.mean().reindex(order)
            ax.plot(order, means, marker="o", label=model)
        ax.set_xlabel("Cell-DINO-selected source difficulty tier")
        ax.set_ylabel("Well-level top-1"); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(figures / "controlled_resampling_accuracy.png", dpi=180); plt.close(fig)

    # Compact learning-curve panel.
    fig, ax = plt.subplots(figsize=(9, 6))
    for row in rows:
        if row["result"] is None or row["result"].get("canary"):
            continue
        curve_path = root / "runs" / f"{row['spec']['run_id']}.curves.jsonl"
        if not curve_path.is_file():
            continue
        curve = [json.loads(line) for line in curve_path.read_text().splitlines() if line.strip()]
        supervised = [item for item in curve if item.get("phase") == "supervised"]
        ax.plot([item["epoch"] for item in supervised],
                [item["train_augmented_top1"] for item in supervised], alpha=0.18,
                color={"resnet18": "C0", "vit_tiny": "C1", "vit_tiny_moe": "C2"}.get(
                    row["result"]["model"], "gray"))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Augmented site-level train accuracy")
    ax.set_title("Training audit curves (one line per run)")
    fig.tight_layout(); fig.savefig(figures / "training_curves.png", dpi=180); plt.close(fig)

    parameter_gate, parameter_detail = should_run_parameter_match(root)
    fmt = lambda value: "—" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.4f}"
    lines = [
        "# RxRx1 HUVEC systematic fast study", "",
        "This is a one-seed screening study. It supports direction finding, not final uncertainty claims.", "",
        f"Completed raw-image runs: **{sum(row['result'] is not None for row in rows)}/{len(rows)}**.", "",
        "## Model summary", "",
        "| Model | Target rows | Mean target top-1 | Mean IID top-1 | Cell-DINO difficulty→accuracy r | Slope |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in contrasts.itertuples(index=False):
        lines.append(f"| {row.model} | {row.n_target_rows} | {fmt(row.mean_target_top1)} | "
                     f"{fmt(row.mean_iid_top1)} | "
                     f"{fmt(row.cell_dino_difficulty_accuracy_correlation)} | "
                     f"{fmt(row.cell_dino_difficulty_accuracy_slope)} |")
    lines += ["", "## Conditional parameter-matched control", "",
              f"Launch gate: **{'run' if parameter_gate else 'skip'}**.", "",
              "```json", json.dumps(parameter_detail, indent=2, sort_keys=True), "```", "",
              "## Interpretation guardrails", "",
              "- High batch-probe accuracy alone is not evidence of harmful batch effects.",
              "- Harm is supported when source-IID accuracy exceeds target accuracy and target accuracy declines with independently measured distance.",
              "- MoE evidence requires a paired improvement over the certified dense ViT, especially on more distant targets.",
              "- The parameter-matched dense control separates conditional computation from ordinary added capacity.",
              "- MAE target images are excluded from pretraining; supervised fine-tuning is end to end.", ""]
    lines.insert(-1, "- Target accuracy is computed over each experiment's observed treatment "
                       "wells; label coverage is reported in the raw metrics table.")
    (analysis / "REPORT.md").write_text("\n".join(lines))
    marker = {"completed_at": time.time(), "complete_results": sum(r["result"] is not None for r in rows),
              "declared_runs": len(rows), "report": str(analysis / "REPORT.md"),
              "parameter_match_gate": parameter_gate, "parameter_match_detail": parameter_detail}
    temporary = root / f"AGGREGATED.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True))
    os.replace(temporary, root / "AGGREGATED.json")
    return marker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(status_table(args.result_root)); return
    print(json.dumps(aggregate(args.result_root, args.require_complete), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
