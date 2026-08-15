#!/usr/bin/env python
"""Evaluate a source-selected supervised checkpoint on the still-sealed target batches."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.data.rxrx1_huvec import EXPECTED_TREATMENTS
from moe_shift.models.huvec import build_study_model
from scripts.run_rxrx1_huvec_study import _make_loaders, _split_hash, evaluate


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _atomic_parquet(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False); os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune-dir", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--site-manifest", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    started = time.time()
    finetune_dir = Path(args.finetune_dir).expanduser().resolve()
    selection_path = finetune_dir / "PLATEAU_RESULT.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"source-only checkpoint selection is incomplete: {selection_path}")
    selection = json.loads(selection_path.read_text())
    selected_attempt = selection["selected_attempt"]
    checkpoint_path = Path(selected_attempt["best_source_iid_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    registry = json.loads(Path(args.registry).read_text())
    registry["site_manifest"] = str(Path(args.site_manifest).expanduser().resolve())
    registry["raw_root"] = str(Path(args.raw_root).expanduser().resolve())
    split_id = selection["split_id"]
    split_matches = [row for row in registry["main_training_splits"]
                     if row["split_id"] == split_id]
    if len(split_matches) != 1:
        raise ValueError(f"split {split_id!r} occurs {len(split_matches)} times")
    split = split_matches[0]

    # The target Dataset is created only after the source-selected checkpoint exists above.
    assignment, loaders = _make_loaders(
        Path(args.output_dir), registry, split, args.batch_size, args.workers,
        int(selection["image_size"]),
        canary=False, include_target=True, train_augmentation=False)
    if _split_hash(assignment) != selection["split_hash"]:
        raise RuntimeError("evaluation split differs from the source-only training split")
    device = torch.device("cuda")
    model, model_audit = build_study_model(
        selection["model"], EXPECTED_TREATMENTS, int(selection["image_size"]))
    model.load_state_dict(checkpoint["model"], strict=True); model.to(device)

    metrics, predictions = {}, []
    for role, loader_name in (("train", "train_eval"),
                              ("iid_validation", "iid_validation"),
                              ("target", "target")):
        role_metrics, role_predictions, _ = evaluate(model, loaders[loader_name], device)
        metrics[role] = role_metrics
        role_predictions.insert(0, "role", role)
        predictions.append(role_predictions)
    output = Path(args.output_dir).expanduser().resolve()
    _atomic_parquet(output / "well_predictions.parquet", pd.concat(predictions, ignore_index=True))
    result = {
        "schema_version": 1, "state": "complete", "model": selection["model"],
        "split_id": split_id, "split_hash": selection["split_hash"],
        "initialization": selection.get("initialization"),
        "selected_attempt": selected_attempt, "checkpoint": str(checkpoint_path),
        "metrics": metrics,
        "target_used_for_checkpoint_selection": False,
        "target_first_loaded_after_selection": True,
        "model_audit": model_audit,
        "elapsed_seconds": time.time() - started,
        "device": torch.cuda.get_device_name(device),
    }
    _atomic_json(output / "RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
