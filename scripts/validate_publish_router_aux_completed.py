#!/usr/bin/env python
"""Strictly validate and publish completed router-auxiliary runs lacking manifests."""

import hashlib
import json
import math
import os
from pathlib import Path

from huggingface_hub import HfApi

from scripts.run_ccas import publish_hf_run, validate_stage1_artifacts
from scripts.sweep_rxrx1_router_aux import CONFIG, cells


ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/router_aux60_20260802"
)
REPORT = Path("/home/idies/workspace/hb_router_publish0800.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


registry = {run_id: tag for tag, _, run_id in cells(CONFIG)}
reports = []
for result_path in sorted(ROOT.glob("*.json")):
    if result_path.name.endswith("artifact_manifest.json"):
        continue
    result = json.loads(result_path.read_text())
    rid = result["run_id"]
    manifest_path = ROOT / f"{rid}.artifact_manifest.json"
    if manifest_path.exists():
        continue
    assert rid in registry
    tag = registry[rid]
    pressure = tag.split("_", 1)[0]
    assert result["dataset"] == "rxrx1" and result["seed"] == 0
    assert (
        result["variant"], result["placement"], result["routing_unit"],
        result["geometry"], result["pressure"], result["n_experts"], result["top_k"],
    ) == ("moe", "early", "token", "cosine", pressure, 8, 1)
    assert result["classification_objective"] == "erm"
    assert result["selection_split"] == "ood_val" and result["test_evaluated"] is False
    assert all(
        result.get(key) is None
        for key in (
            "acc_heldout", "worst_env_heldout", "per_env_heldout",
            "per_env_n_heldout", "degradation_gap_test",
        )
    )
    assert set(result["per_env_val"]) == {"7", "27", "42", "49"}
    assert sum(result["per_env_n_val"].values()) == 9854
    milestone_path = ROOT / f"{rid}.milestones.jsonl"
    rows = validate_stage1_artifacts(result, milestone_path)
    assert len(rows) == 3 and [row["epoch"] for row in rows] == [10, 30, 60]
    assert all(
        row["run_id"] == rid
        and row["selection_split"] == "ood_val"
        and row["test_evaluated"] is False
        for row in rows
    )
    assert result["total_params"] == 30_676_212
    assert result["training_total_params"] == 30_676_212
    assert result["git_sha"] == "7dcb42e" and result["git_dirty"] is False
    assert result["tracking"]["group"] == "rxrx1-cell-dino-router-aux60-20260802"
    assert result["tracking"]["run_id"]
    assert all(
        math.isfinite(float(result[key]))
        for key in ("acc_train", "acc_within", "acc_selection", "worst_env_val")
    )
    log_path = ROOT / f"{rid}.log"
    assert not any(
        marker in log_path.read_text()
        for marker in ("Traceback", "CUDA out of memory", "FATAL")
    )
    checkpoint_path = ROOT / f"{rid}.epoch060.pt"
    assert checkpoint_path.is_file() and checkpoint_path.stat().st_size > 300_000_000
    published = publish_hf_run(
        result,
        [result_path, log_path, milestone_path, checkpoint_path],
        ROOT,
    )
    remote = set(
        HfApi(token=os.environ["HF_TOKEN"]).list_repo_files(
            os.environ["CCAS_HF_REPO"], repo_type="dataset"
        )
    )
    expected_remote = {
        f"{published['prefix']}/{path.name}"
        for path in (result_path, log_path, milestone_path, checkpoint_path, manifest_path)
    }
    assert expected_remote <= remote
    reports.append({
        "status": "strict_validation_and_hf_publish_pass",
        "run_id": rid,
        "metrics": {
            "train": result["acc_train"],
            "id": result["acc_within"],
            "ood_val": result["acc_selection"],
            "worst_experiment": result["worst_env_val"],
        },
        "milestones": [10, 30, 60],
        "checkpoint": {
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
        },
        "manifest_sha256": sha256(manifest_path),
        "hf_prefix": published["prefix"],
        "hf_verified_file_count": len(expected_remote),
        "selection_split": "ood_val",
        "test_evaluated": False,
    })

REPORT.write_text(json.dumps(reports, indent=2))
print("STRICT_ROUTER_BATCH_PUBLISH_PASS", len(reports))
