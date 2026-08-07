#!/usr/bin/env python
"""Evaluate a completed CCAS selection checkpoint on the RxRx1 OOD test split.

This is an evaluation-only path: it restores the exact terminal checkpoint, performs no
optimization, and writes a separate ``*.ood_test.json`` artifact so the selection-stage result
is never overwritten.  It is intended for an explicitly declared all-candidate exploratory
test readout or for a final confirmatory evaluation.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_shift.capacity.model import build_ccas
from moe_shift.capacity.naming import run_id_from
from moe_shift.data import make_loaders
from scripts.run_ccas import _sha256_file, evaluate, git_info


def validate_checkpoint(payload):
    required = ("run_id", "epoch", "config", "model", "milestone")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"checkpoint missing required fields: {missing}")
    cfg = payload["config"]
    run_id = str(payload["run_id"])
    if run_id != run_id_from(cfg):
        raise ValueError("checkpoint run_id does not match embedded config")
    epoch = int(payload["epoch"])
    if epoch != int(cfg["train"]["epochs"]):
        raise ValueError("test evaluation requires the declared terminal checkpoint")
    milestone = payload["milestone"]
    if milestone.get("run_id") != run_id or int(milestone.get("epoch", -1)) != epoch:
        raise ValueError("checkpoint milestone identity mismatch")
    if milestone.get("selection_split") != "ood_val":
        raise ValueError("checkpoint was not selected on OOD validation")
    return cfg, run_id, epoch, milestone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--role",
        choices=("exploratory_all_predefined_arms", "confirmatory"),
        default="exploratory_all_predefined_arms",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint.with_name(checkpoint.name.removesuffix(".pt") + ".ood_test.json")
    )
    if output.exists() and not args.force:
        print(f"[skip] OOD-test artifact already exists: {output}")
        return

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg, run_id, epoch, milestone = validate_checkpoint(payload)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    _train_loader, _id_loader, test_loader, _audit_loader = make_loaders(cfg)
    model = build_ccas(cfg).to(args.device)
    model.load_state_dict(payload["model"], strict=True)
    acc, worst, per_env, per_env_n = evaluate(model, test_loader, args.device)

    git_sha, git_dirty = git_info()
    record = {
        "run_id": run_id,
        "dataset": cfg["dataset"],
        "seed": int(cfg["seed"]),
        "variant": model.capacity.variant,
        "block_indices": list(model.capacity.block_indices),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_epoch": epoch,
        "checkpoint_selection_split": milestone["selection_split"],
        "checkpoint_ood_val_acc": milestone["acc_selection"],
        "test_evaluated": True,
        "test_role": args.role,
        "acc_heldout": acc,
        "worst_env_heldout": worst,
        "per_env_heldout": per_env,
        "per_env_n_heldout": per_env_n,
        "evaluation_git_sha": git_sha,
        "evaluation_git_dirty": git_dirty,
        "training_git_sha": "f21ed043b9564125562edd3aa629a08fd61db17e",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    os.replace(temporary, output)
    print(
        f"[ood-test] {run_id} accuracy={acc:.6f} worst={worst:.6f} -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
