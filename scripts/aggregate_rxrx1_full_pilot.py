#!/usr/bin/env python
"""Aggregate completed full-RxRx1 random-versus-MAE evaluations."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import pandas as pd


def _atomic_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text); os.replace(temporary, path)


def aggregate(result_root):
    root = Path(result_root).expanduser().resolve()
    rows, cell_rows = [], []
    for path in sorted((root / "evaluation").glob("*/RESULT.json")):
        result = json.loads(path.read_text())
        run_name = path.parent.name
        arm, fold = run_name.split("_full_", 1)
        fold = "full_" + fold
        metrics = result["metrics"]
        row = {
            "run": run_name, "arm": arm, "fold": fold,
            "normalization": result.get("normalization_mode"),
            "train_site_top1": metrics["train"]["site_top1"],
            "iid_site_top1": metrics["iid_validation"]["site_top1"],
            "target_site_top1": metrics["target"]["site_top1"],
            "train_well_top1": metrics["train"]["top1"],
            "iid_well_top1": metrics["iid_validation"]["top1"],
            "target_well_top1": metrics["target"]["top1"],
        }
        row["site_iid_minus_target"] = row["iid_site_top1"] - row["target_site_top1"]
        row["well_iid_minus_target"] = row["iid_well_top1"] - row["target_well_top1"]
        rows.append(row)
        for role, role_metrics in metrics.items():
            for cell, values in role_metrics.get("per_cell_type", {}).items():
                cell_rows.append({
                    "run": run_name, "arm": arm, "fold": fold, "role": role,
                    "cell_type": cell, **values,
                })
    if not rows:
        raise FileNotFoundError(f"no completed full-RxRx1 evaluations under {root}")
    summary = pd.DataFrame(rows).sort_values(["fold", "arm"])
    per_cell = pd.DataFrame(cell_rows).sort_values(
        ["fold", "arm", "role", "cell_type"])
    analysis = root / "analysis"; analysis.mkdir(parents=True, exist_ok=True)
    summary.to_csv(analysis / "full_rxrx1_summary.csv", index=False)
    per_cell.to_csv(analysis / "full_rxrx1_per_cell.csv", index=False)
    title = "Full RxRx1 pilot: random initialization versus source-only MAE"
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#1f2937}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:7px;border:1px solid #d1d5db;text-align:right}}
th:first-child,td:first-child{{text-align:left}}h1,h2{{color:#17324d}}code{{background:#f3f4f6;padding:2px 4px}}</style></head>
<body><h1>{html.escape(title)}</h1>
<p>All checkpoints were selected only on source-IID accuracy. Held-out target experiments were
loaded afterward. Positive values in <code>IID minus target</code> measure batch-held-out degradation.</p>
<h2>Overall perturbation accuracy</h2>{summary.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
<h2>Accuracy by cell type</h2>{per_cell.to_html(index=False, float_format=lambda x: f'{x:.4f}')}
</body></html>"""
    _atomic_text(analysis / "full_rxrx1_report.html", document)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args(); aggregate(args.result_root)


if __name__ == "__main__":
    main()
