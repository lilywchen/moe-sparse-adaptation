#!/usr/bin/env python
"""Rank Cell-DINO FFN layers by training-experiment gradient conflict."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe_shift.audit.gradient_conflict import profile_gradient_conflict
from moe_shift.capacity.model import build_ccas
from moe_shift.data import make_loaders
from moe_shift.utils.config import apply_overrides, load_config


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-environment", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sketch-size", type=int, default=4096)
    parser.add_argument("--max-environments", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["model"]["variant"] = "original"
    cfg["model"]["freeze_backbone"] = False
    cfg["model"]["unfreeze_last_n_blocks"] = 0
    plan = {
        "analysis": "experiment_stratified_gradient_conflict",
        "data_scope": "train_only",
        "selection_split": None,
        "test_evaluated": False,
        "acc_heldout": None,
        "worst_env_heldout": None,
        "samples_per_environment": args.samples_per_environment,
        "rounds": args.rounds,
        "sketch_size": args.sketch_size,
        "max_environments": args.max_environments,
        "checkpoint": Path(args.checkpoint).name if args.checkpoint else None,
        "config": cfg,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _id_loader, _sealed_test_loader, _audit_loader = make_loaders(cfg)
    model = build_ccas(cfg).to(device)
    checkpoint_hash = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        payload = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(payload.get("model", payload), strict=True)
        checkpoint_hash = sha256(checkpoint)

    report = profile_gradient_conflict(
        model, train_loader.dataset, device,
        samples_per_environment=args.samples_per_environment,
        rounds=args.rounds, sketch_size=args.sketch_size, seed=int(cfg["seed"]),
        max_environments=args.max_environments)
    result = {
        **plan,
        "checkpoint_sha256": checkpoint_hash,
        "git_sha": git_sha(),
        "profile": report,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"[done] ranked {len(report['layers'])} FFNs -> {output}")


if __name__ == "__main__":
    main()
