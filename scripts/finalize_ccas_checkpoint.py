#!/usr/bin/env python
"""Recover a sealed-test CCAS result from a completed selection-stage checkpoint.

This is intentionally a post-training finalizer, not a resume path.  It exists for the narrow
case where optimization and the declared milestone checkpoint completed but the ordinary result
writer failed afterwards.  The checkpoint's embedded config and terminal milestone are the
authority; the script refuses partial checkpoints, run-id mismatches, stage-3/test evaluation,
or a missing OOD-validation split.
"""
import argparse
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe_shift.audit import leakage as audit_leak
from moe_shift.audit import routing as audit_routing
from moe_shift.capacity.model import build_ccas
from moe_shift.capacity.naming import run_id_from
from moe_shift.data import make_loaders, make_val_loader
from scripts.run_ccas import (
    _sha256_file,
    counterfactual_reroute,
    evaluate,
    git_info,
    normalize_withheld_ood_fields,
    validate_stage1_artifacts,
)


def validate_recovery_checkpoint(payload, expected_run_id=None):
    """Fail closed unless this is a complete selection-stage terminal checkpoint."""
    required = ("run_id", "epoch", "config", "model", "milestone")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"recovery checkpoint missing {missing}")
    cfg = payload["config"]
    rid = str(payload["run_id"])
    if rid != run_id_from(cfg):
        raise ValueError("checkpoint run_id does not match embedded config")
    if expected_run_id is not None and rid != expected_run_id:
        raise ValueError("checkpoint run_id does not match --expected-run-id")
    epoch = int(payload["epoch"])
    if epoch != int(cfg["train"]["epochs"]):
        raise ValueError("recovery requires the declared terminal checkpoint")
    if int(cfg.get("stage", 1)) >= 3:
        raise ValueError("recovery finalizer is selection-stage only; OOD test stays sealed")
    milestone = payload["milestone"]
    if milestone.get("run_id") != rid or int(milestone.get("epoch", -1)) != epoch:
        raise ValueError("checkpoint milestone identity mismatch")
    if milestone.get("selection_split") != "ood_val" or milestone.get("test_evaluated") is not False:
        raise ValueError("checkpoint milestone violates sealed-test selection protocol")
    for key in ("acc_train", "acc_within", "acc_selection", "worst_env_val"):
        if not math.isfinite(float(milestone[key])):
            raise ValueError(f"checkpoint milestone has non-finite {key}")
    return cfg, rid, epoch, milestone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--expected-run-id", default=None)
    args = ap.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.results_dir).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg, rid, epoch, checkpoint_milestone = validate_recovery_checkpoint(
        payload, args.expected_run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{rid}.json"
    if out_json.exists():
        print(f"[skip] {rid} already finalized")
        return

    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()
    train_loader, test_within, _test_heldout_unread, audit_loader = make_loaders(cfg)
    val_loader = make_val_loader(cfg)
    if val_loader is None:
        raise RuntimeError("checkpoint recovery requires OOD validation; test fallback is forbidden")

    model = build_ccas(cfg).to(device)
    model.load_state_dict(payload["model"], strict=True)
    cap = model.capacity
    if tuple(cap.block_indices) != tuple(payload["config"]["model"]["ffn_block_indices"]):
        raise ValueError("recovered model block identity differs from checkpoint config")

    acc_within, worst_within, _, _ = evaluate(model, test_within, device)
    acc_val, worst_val, per_env_val, per_env_n_val = evaluate(model, val_loader, device)
    acc_train = checkpoint_milestone["acc_train"]
    worst_train = checkpoint_milestone["worst_env_train"]

    mech = {}
    if bool(cfg.get("analysis", {}).get("run_mechanism", True)) and model.moe_blocks:
        per_block = {}
        for block_index, block in zip(cap.block_indices, model.moe_blocks):
            try:
                eidx, site, label = audit_routing.capture(model, audit_loader, device, block=block)
                used, entropy = audit_routing.expert_usage(eidx, cfg["model"]["n_experts"])
                per_block[str(block_index)] = {
                    "routing_mi_site": float(audit_routing.routing_mi(eidx, site)),
                    "routing_mi_class": float(audit_routing.routing_mi(eidx, label)),
                    "experts_used": float(used),
                    "routing_entropy": float(entropy),
                }
            except Exception as exc:
                per_block[str(block_index)] = {"routing_error": str(exc)}
        mech["routing_by_block"] = per_block
        first = per_block.get(str(cap.block_indices[0]), {})
        for key in ("routing_mi_site", "routing_mi_class", "experts_used", "routing_entropy"):
            if key in first:
                mech[key] = first[key]
        randomized = counterfactual_reroute(model, val_loader, device, seed=cfg["seed"])
        mech["randomized_routes_acc"] = randomized
        if randomized is not None:
            mech["route_reliance"] = acc_val - randomized
    if bool(cfg.get("analysis", {}).get("run_mechanism", True)):
        try:
            feats, site, label = audit_leak.features_site_label(model, audit_loader, device)
            mech["site_leakage"] = float(audit_leak.site_leakage(feats, site))
            mech["class_decodability"] = float(audit_leak.class_decodability(feats, label))
        except Exception as exc:
            mech["leakage_error"] = str(exc)

    pressure = cfg["model"].get("pressure", "canonical")
    sha, dirty = git_info()
    protocol = {
        "variant": cap.variant,
        "block_index": cap.block_index,
        "block_indices": list(cap.block_indices),
        "n_blocks": len(model.blocks),
        "n_blocks_converted": cap.n_converted_blocks,
        "exactly_one_block_converted": cap.n_converted_blocks == 1,
        "training_pressure": pressure,
        "route_balance": cfg["model"]["balance"],
        "output_adversary": False,
        "classification_objective": str(cfg["train"].get("objective", "erm")),
        "milestone_epochs": list(cfg["train"].get("milestone_epochs", [])),
        "checkpoint_epochs": list(cfg["train"].get("save_checkpoint_epochs", [])),
        "recovered_from_terminal_checkpoint": True,
    }
    if cap.variant in ("moe", "moe_frozen"):
        protocol["experts_are_upcycled_copies"] = True
        protocol["router_trainable"] = cap.variant == "moe"

    result = {
        "run_id": rid,
        "dataset": cfg["dataset"],
        "seed": cfg["seed"],
        "variant": cap.variant,
        "placement": cap.placement,
        "routing_unit": cfg["model"]["routing_unit"],
        "geometry": cfg["model"]["geometry"],
        "pressure": pressure,
        "balance": cfg["model"]["balance"],
        "classification_objective": str(cfg["train"].get("objective", "erm")),
        "n_experts": cfg["model"]["n_experts"],
        "top_k": cfg["model"]["top_k"],
        "stage": int(cfg.get("stage", 1)),
        "selection_split": "ood_val",
        "test_evaluated": False,
        "acc_selection": acc_val,
        "acc_val": acc_val,
        "worst_env_val": worst_val,
        "per_env_val": per_env_val,
        "per_env_n_val": per_env_n_val,
        "acc_heldout": None,
        "worst_env_heldout": None,
        "per_env_heldout": None,
        "per_env_n_heldout": None,
        "acc_within": acc_within,
        "acc_train": acc_train,
        "worst_env_train": worst_train,
        "degradation_gap": acc_within - acc_val,
        "degradation_gap_test": None,
        "total_params": cap.total_params,
        "ffn_block_params": cap.ffn_block_params,
        "router_params": cap.router_params,
        "active_ffn_params": cap.active_ffn_params,
        "training_total_params": cap.total_params,
        "adversary_params": 0,
        "block_index": cap.block_index,
        "block_indices": list(cap.block_indices),
        "n_blocks_converted": cap.n_converted_blocks,
        **mech,
        "protocol": protocol,
        "git_sha": sha,
        "git_dirty": dirty,
        "backbone_provenance": model.backbone_provenance,
        "tracking": None,
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "wall_seconds": round(time.time() - started, 1),
        "config": cfg,
        "recovery": {
            "reason": "post_training_result_writer_failure",
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_epoch": epoch,
            "training_not_repeated": True,
        },
    }
    result = normalize_withheld_ood_fields(result)
    tmp = out_json.with_name(out_json.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out_json)
    milestone_path = out_dir / f"{rid}.milestones.jsonl"
    validate_stage1_artifacts(result, milestone_path)
    print(f"[recovered] {rid} ood_val={acc_val:.6f} id={acc_within:.6f} -> {out_json}")


if __name__ == "__main__":
    main()
