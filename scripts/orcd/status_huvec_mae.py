#!/usr/bin/env python
"""Compact status view for the two ORCD HUVEC MAE runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    root = Path(args.result_root).expanduser()
    for mode in ("smoke", "runs"):
        for model in ("vit_tiny", "vit_micro"):
            status_path = root / mode / model / "STATUS.json"
            if not status_path.is_file():
                print(f"{mode:5s} {model:9s} waiting")
                continue
            row = json.loads(status_path.read_text())
            print(
                f"{mode:5s} {model:9s} {row.get('state', '?'):17s} "
                f"epoch={row.get('epoch', row.get('epochs_completed', '-'))} "
                f"train={row.get('train_reconstruction_loss', '-')} "
                f"val={row.get('validation_reconstruction_loss', '-')} "
                f"best={row.get('best_validation_reconstruction_loss', '-')}"
            )


if __name__ == "__main__":
    main()
