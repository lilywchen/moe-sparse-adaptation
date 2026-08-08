#!/usr/bin/env python
"""Run an inference-only routing audit for an existing CCAS terminal checkpoint.

This deliberately writes a sidecar ``*.mechanism.json`` rather than modifying the
training result.  It is useful when a completed sweep disabled mechanism audits to
save training time.  It never optimizes or reads the held-out test split.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moe_shift.audit import routing as audit_routing
from moe_shift.capacity.model import build_ccas
from moe_shift.capacity.naming import run_id_from
from moe_shift.data import make_loaders, make_val_loader
from scripts.run_ccas import _sha256_file, counterfactual_reroute, evaluate, git_info


def _validate(payload):
    for key in ("run_id", "epoch", "config", "model", "milestone"):
        if key not in payload:
            raise ValueError(f"checkpoint missing required field {key!r}")
    cfg = payload["config"]
    run_id = str(payload["run_id"])
    if run_id != run_id_from(cfg):
        raise ValueError("checkpoint run_id does not match embedded config")
    if int(payload["epoch"]) != int(cfg["train"]["epochs"]):
        raise ValueError("audit requires the declared terminal checkpoint")
    return cfg, run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = (Path(args.output).expanduser().resolve() if args.output else
              checkpoint.with_name(checkpoint.stem + ".mechanism.json"))
    if output.exists() and not args.force:
        print(f"[skip] mechanism artifact already exists: {output}")
        return
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg, run_id = _validate(payload)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    _train, _id, _heldout, audit_loader = make_loaders(cfg)
    val_loader = make_val_loader(cfg)
    if val_loader is None:
        raise ValueError("mechanism audit requires an OOD-validation loader")
    model = build_ccas(cfg).to(args.device)
    model.load_state_dict(payload["model"], strict=True)
    if not model.moe_blocks:
        raise ValueError("checkpoint has no routed blocks to audit")

    acc_val = evaluate(model, val_loader, args.device)[0]
    per_block = {}
    for block_index, block in zip(model.capacity.block_indices, model.moe_blocks):
        eidx, site, label = audit_routing.capture(model, audit_loader, args.device, block=block)
        used, entropy = audit_routing.expert_usage(eidx, cfg["model"]["n_experts"])
        per_block[str(block_index)] = {
            "routing_mi_site": float(audit_routing.routing_mi(eidx, site)),
            "routing_mi_class": float(audit_routing.routing_mi(eidx, label)),
            "experts_used": int(used),
            "routing_entropy": float(entropy),
        }
    randomized = counterfactual_reroute(model, val_loader, args.device, seed=cfg["seed"])
    sha, dirty = git_info()
    result = {
        "run_id": run_id,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "selection_split": payload["milestone"].get("selection_split"),
        "acc_val": acc_val,
        "randomized_routes_acc": randomized,
        "route_reliance": None if randomized is None else acc_val - randomized,
        "routing_by_block": per_block,
        "audit_git_sha": sha,
        "audit_git_dirty": dirty,
        "inference_only": True,
        "heldout_test_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
