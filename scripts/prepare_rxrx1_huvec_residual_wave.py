#!/usr/bin/env python
"""Freeze the 16-run HUVEC residual-MoE mechanism wave."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

from scripts.prepare_rxrx1_huvec_batch_effect import atomic_json

DEFAULT_ANCHOR = Path("/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/substrate_rxrx1/huvec_batch_effect_12h_20260815")
DEFAULT_RESULT = Path("/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/substrate_rxrx1/huvec_residual_16h_20260816")
MODELS = (
    "vit_tiny_residual_token", "vit_tiny_residual_image",
    "vit_tiny_residual_within", "vit_tiny_residual_frozen",
)


def prepare(anchor_root, result_root):
    anchor, root = Path(anchor_root).resolve(), Path(result_root).resolve()
    if (root / "wave_manifest.json").is_file():
        return json.loads((root / "wave_manifest.json").read_text())
    old_registry = json.loads((anchor / "study_registry.json").read_text())
    anchors = list(map(int, old_registry["difficulty_anchor_targets"]))
    split_by_id = {row["split_id"]: row for row in old_registry["splits"]}
    splits = [copy.deepcopy(split_by_id[f"loo_t{target}"]) for target in anchors]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    runs = []
    for target in anchors:
        for model in MODELS:
            runs.append({
                "run_id": f"huvec_res16_{model.removeprefix('vit_tiny_')}_t{target}",
                "stage": "residual_mechanism", "model": model,
                "split_id": f"loo_t{target}", "seed": 0, "image_size": 224,
                "batch_size": 96, "num_workers": 6,
                "recipe": {"name": "adamw_standard_extended", "optimizer": "adamw",
                           "lr": 7.5e-4, "weight_decay": 0.05,
                           "schedule_epochs": 160, "warmup_epochs": 5,
                           "min_lr_ratio": 0.02, "augmentation": True},
                "selection": {"metric": "source_iid_site_top1",
                              "eval_every_epochs": 5, "patience_evaluations": 4,
                              "minimum_delta": 0.001, "minimum_epochs": 30,
                              "maximum_epochs": 80},
            })
    registry = copy.deepcopy(old_registry)
    registry.update({"schema_version": 2, "study": "rxrx1_huvec_residual_16h_20260816",
                     "anchor_result_root": str(anchor), "splits": splits})
    manifest = {
        "schema_version": 2, "campaign": "rxrx1_huvec_residual_16h_20260816",
        "created_at": time.time(), "source_git_commit": commit,
        "source_git_dirty": False, "result_root": str(root),
        "expected_runs": 16, "training_unit": "site",
        "primary_evaluation_unit": "site", "anchor_result_root": str(anchor),
        "design": {"targets": anchors, "models": list(MODELS),
                   "dense_baselines_reused": True, "shared_only_ablation": True},
        "runs": runs,
    }
    atomic_json(root / "study_registry.json", registry)
    atomic_json(root / "wave_manifest.json", manifest)
    atomic_json(root / "PREPARED.json", {"state": "prepared", "expected_runs": 16,
                                          "source_git_commit": commit})
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-root", default=str(DEFAULT_ANCHOR))
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT))
    args = parser.parse_args(); print(json.dumps(prepare(args.anchor_root, args.result_root), indent=2))
