#!/usr/bin/env python
"""Compact source-only progress table for the packed Perlmutter sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


RUNS = (
    "random_global_anchor", "random_per_image_standard",
    "mae_canonical_global_anchor", "mae_canonical_per_image_standard",
    "mae_per_image_global_anchor", "mae_per_image_matched_standard",
    "mae_per_image_matched_lr250e6", "mae_canonical_per_image_lr250e6",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    root = (Path(args.result_root).expanduser().resolve()
            / "finetune_tiny_perlmutter" / "recipe_certification")
    for run in RUNS:
        path = root / run / "vit_tiny" / "status.json"
        if not path.is_file():
            print(f"{run:42s} waiting")
            continue
        row = json.loads(path.read_text())
        latest = row.get("latest_evaluation") or {}
        best = row.get("best_source_iid") or {}
        print(
            f"{run:42s} {row.get('state', '?'):10s} "
            f"epoch={row.get('epoch', 0):>2}/80 "
            f"train={latest.get('train_site_top1', float('nan')):.4f} "
            f"iid={latest.get('iid_site_top1', float('nan')):.4f} "
            f"best_iid={best.get('selection_iid_top1', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
