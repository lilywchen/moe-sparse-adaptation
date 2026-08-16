#!/usr/bin/env python
"""Aggregate residual mechanisms against reused dense anchors."""
from __future__ import annotations
import argparse, html, json, os
from pathlib import Path
import numpy as np
import pandas as pd


def atomic_text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(value); os.replace(tmp, path)


def collect(root):
    root = Path(root).resolve(); manifest = json.loads((root / "wave_manifest.json").read_text())
    rows = []
    for spec in manifest["runs"]:
        path = root / "runs" / spec["run_id"] / "RESULT.json"
        if not path.is_file():
            rows.append({"run_id": spec["run_id"], "model": spec["model"],
                         "target": int(spec["split_id"].split("t")[-1]), "state": "pending"})
            continue
        value = json.loads(path.read_text()); shared = value.get("shared_only_metrics") or {}
        rows.append({"run_id": spec["run_id"], "model": spec["model"],
                     "target": int(spec["split_id"].split("t")[-1]), "state": "complete",
                     "train_site": value["metrics"]["train"]["site"]["top1"],
                     "iid_site": value["metrics"]["iid_validation"]["site"]["top1"],
                     "target_site": value["metrics"]["target"]["site"]["top1"],
                     "gap": value["site_iid_to_target_gap"],
                     "shared_iid": shared.get("iid_validation", {}).get("site_top1"),
                     "shared_target": shared.get("target", {}).get("site_top1"),
                     "best_epoch": value["best_epoch"],
                     "hours": value["elapsed_seconds"] / 3600})
    return pd.DataFrame(rows), manifest


def aggregate(root, require_complete=True):
    root = Path(root).resolve(); table, manifest = collect(root)
    complete = table[table.state == "complete"].copy()
    if require_complete and len(complete) != len(table):
        raise RuntimeError(f"{len(complete)}/{len(table)} runs complete")
    analysis = root / "analysis"; analysis.mkdir(parents=True, exist_ok=True)
    table.to_csv(analysis / "run_metrics.csv", index=False)
    anchor = Path(manifest["anchor_result_root"])
    dense = []
    for target in sorted(table.target.unique()):
        path = anchor / "runs" / f"huvec_batch12_dense_loo_t{target}" / "RESULT.json"
        if path.is_file():
            value = json.loads(path.read_text())
            dense.append({"target": target, "dense_iid": value["metrics"]["iid_validation"]["site"]["top1"],
                          "dense_target": value["metrics"]["target"]["site"]["top1"]})
    comparison = complete.merge(pd.DataFrame(dense), on="target", how="left")
    comparison["target_gain_vs_dense"] = comparison.target_site - comparison.dense_target
    comparison["residual_contribution_target"] = comparison.target_site - comparison.shared_target
    comparison.to_csv(analysis / "mechanism_comparison.csv", index=False)
    summary = comparison.groupby("model").agg(
        runs=("run_id", "count"), target_site=("target_site", "mean"),
        gain_vs_dense=("target_gain_vs_dense", "mean"),
        residual_contribution=("residual_contribution_target", "mean"),
        iid_site=("iid_site", "mean"), hours=("hours", "sum")).reset_index()
    summary.to_csv(analysis / "mechanism_summary.csv", index=False)
    cards = "".join(f"<tr><td>{html.escape(r.model)}</td><td>{r.runs}</td><td>{r.target_site:.3f}</td><td>{r.gain_vs_dense:+.3f}</td><td>{r.residual_contribution:+.3f}</td><td>{r.hours:.1f}</td></tr>" for r in summary.itertuples())
    report = f"""<!doctype html><meta charset='utf-8'><title>HUVEC residual MoE study</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:40px auto;line-height:1.5;color:#18212b}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{background:#eef6ff;padding:16px;border-radius:10px}}</style>
<h1>RxRx1 HUVEC: residual conditional processing</h1>
<p class='note'><b>Question.</b> Once a dense ViT is a working supervised instrument, does an always-active shared FFN plus sparse routed corrections improve held-out-batch perturbation accuracy? Target images never select checkpoints. Dense anchor results are reused, not retrained.</p>
<h2>Design</h2><p>Four held-out experiments × token routing, image routing, within-experiment balancing, and frozen-router control. Each trained model is re-evaluated with routed residuals disabled; that ablation measures whether the conditional path actually contributes.</p>
<h2>Results ({len(complete)}/{len(table)} complete)</h2><table><tr><th>mechanism</th><th>runs</th><th>target site top-1</th><th>gain vs dense</th><th>routed contribution</th><th>GPU h</th></tr>{cards}</table>
<h2>Interpretation guide</h2><ul><li>A gain over dense supports conditional capacity, not merely train fit.</li><li>A drop when residuals are disabled shows the routed branch is functionally used.</li><li>Image routing beating token routing supports batch/style-level regimes; token routing supports local morphological specialization.</li><li>Frozen routing matching learned routing argues against meaningful discovered regimes.</li></ul>
<p>Machine-readable files: run_metrics.csv, mechanism_comparison.csv, mechanism_summary.csv.</p>"""
    atomic_text(analysis / "REPORT.html", report)
    return summary


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--result-root", required=True); p.add_argument("--allow-incomplete", action="store_true")
    a=p.parse_args(); print(aggregate(a.result_root, not a.allow_incomplete).to_string(index=False))
