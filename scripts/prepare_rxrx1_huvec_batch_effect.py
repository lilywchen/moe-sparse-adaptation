#!/usr/bin/env python
"""Freeze the 12-hour RxRx1 HUVEC batch-degradation and MoE study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.data.rxrx1_huvec import deterministic_split, normalization_from_qc
from moe_shift.data.rxrx1_huvec_batch import (
    choose_difficulty_anchors,
    role_label_coverage,
    source_compositions,
    target_difficulty,
)
from scripts.run_rxrx1_huvec_study import _split_hash


DEFAULT_BASE_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/huvec_systematic_fast_20260814"
)
DEFAULT_RESULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/huvec_batch_effect_12h_20260815"
)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_identity():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
        text=True).strip())
    return commit, dirty


def _primary_fold0(base_registry):
    matches = [row for row in base_registry["main_training_splits"]
               if row["split_id"] == "primary_fold0"]
    if len(matches) != 1:
        raise ValueError(f"expected one primary_fold0, found {len(matches)}")
    return matches[0]


def _run_id(model, split_id):
    short = {"vit_tiny": "dense", "vit_tiny_moe": "moe",
             "vit_tiny_dense_matched": "dense_total_matched"}[model]
    return f"huvec_batch12_{short}_{split_id}"


def planned_runs(splits, anchor_targets):
    by_id = {row["split_id"]: row for row in splits}
    diagnostic = [row for row in splits if row["kind"] == "diagnostic_loo"]
    controlled = [row for row in splits if row["kind"] == "source_composition"]
    rows = []

    def add(model, split, stage):
        rows.append({
            "run_id": _run_id(model, split["split_id"]),
            "stage": stage, "model": model, "split_id": split["split_id"],
            "seed": 0, "image_size": 224, "batch_size": 128, "num_workers": 6,
            "recipe": {
                "name": "adamw_standard_extended", "optimizer": "adamw",
                "lr": 7.5e-4, "weight_decay": 0.05,
                "schedule_epochs": 160, "warmup_epochs": 5,
                "min_lr_ratio": 0.02, "augmentation": True,
            },
            "selection": {
                "metric": "source_iid_site_top1", "eval_every_epochs": 5,
                "patience_evaluations": 4, "minimum_delta": 0.001,
                "minimum_epochs": 30, "maximum_epochs": 80,
            },
        })

    for split in diagnostic:
        add("vit_tiny", split, "diagnostic_loo")
    for split in controlled:
        add("vit_tiny", split, "source_composition")
    for index, target in enumerate(anchor_targets):
        split = by_id[f"loo_t{int(target)}"]
        models = (["vit_tiny_moe", "vit_tiny_dense_matched"] if index % 2 == 0
                  else ["vit_tiny_dense_matched", "vit_tiny_moe"])
        for model in models:
            add(model, split, "capacity_mechanism")
    if len(rows) != 36 or len({row["run_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"expected 36 unique runs, found {len(rows)}")
    return rows


def prepare(base_result_root, result_root):
    base = Path(base_result_root).expanduser().resolve()
    root = Path(result_root).expanduser().resolve()
    manifest_path = root / "wave_manifest.json"
    registry_path = root / "study_registry.json"
    if manifest_path.is_file() or registry_path.is_file():
        if not (manifest_path.is_file() and registry_path.is_file()):
            raise RuntimeError("partial frozen study exists; do not overwrite it")
        print(json.dumps({
            "state": "already_prepared", "manifest": str(manifest_path),
            "registry": str(registry_path),
        }, indent=2))
        return json.loads(manifest_path.read_text()), json.loads(registry_path.read_text())

    base_registry_path = base / "study_registry.json"
    site_manifest_path = base / "data" / "huvec_sites.parquet"
    site_qc_path = base / "cache" / "site_qc.parquet"
    well_meta_path = base / "cache" / "well_metadata.parquet"
    well_embedding_path = base / "cache" / "well_cell_dino.npy"
    distance_path = base / "analysis" / "cell_dino_experiment_distance.npy"
    required = [base_registry_path, site_manifest_path, site_qc_path, well_meta_path,
                well_embedding_path, distance_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"base HUVEC study artifacts are missing: {missing}")

    base_registry = json.loads(base_registry_path.read_text())
    primary = _primary_fold0(base_registry)
    source_pool = sorted(map(int, primary["source_experiments"]))
    sealed_targets = sorted(map(int, primary["target_experiments"]))
    if len(source_pool) != 16 or len(sealed_targets) != 8:
        raise ValueError("primary_fold0 must provide 16 diagnostic and 8 sealed experiments")
    sites = pd.read_parquet(site_manifest_path)
    site_qc = pd.read_parquet(site_qc_path)
    well_meta = pd.read_parquet(well_meta_path)
    embeddings = np.load(well_embedding_path)
    distance = np.load(distance_path)
    experiment_order = list(map(int, base_registry["experiments"]))
    experiments = well_meta.experiment.to_numpy(np.int64)
    labels = well_meta.label.to_numpy(np.int64)
    anchors, diagnostic_difficulty = choose_difficulty_anchors(
        source_pool, experiments, labels, embeddings, count=4)

    split_rows = []
    for target in source_pool:
        split_rows.append({
            "split_id": f"loo_t{target}", "kind": "diagnostic_loo",
            "target_experiments": [target],
            "source_experiments": [value for value in source_pool if value != target],
            "composition": "all_other_diagnostic_sources",
        })
    for target in anchors:
        for composition in source_compositions(
                target, source_pool, experiment_order, distance, size=8):
            split_rows.append({
                "split_id": f"composition_t{target}_{composition['composition']}",
                "kind": "source_composition", "target_experiments": [target],
                **composition,
            })
    if len(split_rows) != 28 or len({row["split_id"] for row in split_rows}) != 28:
        raise RuntimeError("the frozen design must contain 16 LOO and 12 composition splits")

    for split in split_rows:
        assignment = deterministic_split(
            sites, split["source_experiments"], split["target_experiments"],
            split["split_id"])
        means, stds = normalization_from_qc(
            assignment[assignment.role == "train"], site_qc)
        target = int(split["target_experiments"][0])
        difficulty, matched = target_difficulty(
            target, split["source_experiments"], experiments, labels, embeddings)
        split.update({
            "normalization": {"mean": means, "std": stds},
            "cell_dino_difficulty": float(difficulty),
            "cell_dino_matched_labels": int(matched),
            "role_label_coverage": role_label_coverage(assignment),
            "split_hash": _split_hash(assignment),
        })

    commit, dirty = git_identity()
    if dirty:
        raise RuntimeError("refuse to freeze the batch study from a dirty tracked checkout")
    runs = planned_runs(split_rows, anchors)
    root.mkdir(parents=True, exist_ok=True)
    registry = {
        "schema_version": 1,
        "study": "rxrx1_huvec_batch_effect_12h_20260815",
        "base_result_root": str(base),
        "base_registry": str(base_registry_path),
        "base_registry_sha256": sha256(base_registry_path),
        "site_manifest": str(site_manifest_path),
        "site_manifest_sha256": sha256(site_manifest_path),
        "site_qc": str(site_qc_path), "raw_root": base_registry["raw_root"],
        "well_metadata": str(well_meta_path),
        "well_cell_dino": str(well_embedding_path),
        "experiment_distance": str(distance_path),
        "experiment_order": experiment_order,
        "diagnostic_source_pool": source_pool,
        "sealed_primary_targets": sealed_targets,
        "difficulty_anchor_targets": anchors,
        "diagnostic_target_difficulty": diagnostic_difficulty,
        "splits": split_rows,
        "target_policy": (
            "The eight primary_fold0 target experiments are never used by this diagnostic wave. "
            "Each diagnostic target is excluded from normalization, training, source-IID "
            "evaluation, checkpoint selection, and plateau stopping for its fold."
        ),
    }
    manifest = {
        "schema_version": 1,
        "campaign": "rxrx1_huvec_batch_effect_12h_20260815",
        "created_at": time.time(), "source_git_commit": commit,
        "source_git_dirty": False, "result_root": str(root),
        "expected_runs": len(runs), "seed_policy": "one frozen pilot seed",
        "training_unit": "site", "primary_evaluation_unit": "site",
        "secondary_evaluation_unit": "well_mean_logits",
        "checkpoint_rule": "highest source-IID site accuracy; earlier epoch breaks ties",
        "pseudo_target_resamples": 50,
        "design": {
            "diagnostic_loo_dense": 16,
            "source_composition_dense": 12,
            "anchor_moe": 4,
            "anchor_total_parameter_matched_dense": 4,
        },
        "runs": runs,
    }
    atomic_json(registry_path, registry)
    atomic_json(manifest_path, manifest)
    atomic_json(root / "PREPARED.json", {
        "state": "prepared", "created_at": time.time(), "expected_runs": len(runs),
        "anchor_targets": anchors, "sealed_primary_targets": sealed_targets,
        "source_git_commit": commit,
    })
    print(json.dumps({
        "state": "prepared", "expected_runs": len(runs),
        "splits": len(split_rows), "anchor_targets": anchors,
        "sealed_primary_targets": sealed_targets, "result_root": str(root),
    }, indent=2, sort_keys=True))
    return manifest, registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-result-root", default=str(DEFAULT_BASE_ROOT))
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    args = parser.parse_args()
    prepare(args.base_result_root, args.result_root)


if __name__ == "__main__":
    main()
