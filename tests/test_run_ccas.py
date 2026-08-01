import pytest
import torch

import json

from scripts.run_ccas import (
    classification_objective, milestone_epochs, validate_stage1_artifacts,
)


def test_environment_balanced_objective_equalizes_experiments():
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    labels = torch.tensor([0, 0, 0])
    env = torch.tensor([0, 0, 1])
    per_example = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = 0.5 * (per_example[:2].mean() + per_example[2:].mean())
    assert classification_objective(logits, labels, env, "environment_balanced") == pytest.approx(expected)
    assert classification_objective(logits, labels, env, "erm") == pytest.approx(per_example.mean())


def test_milestones_are_one_indexed_bounded_and_checkpoints_are_subset():
    cfg = {"train": {"epochs": 90, "milestone_epochs": [90, 10, 30, 10],
                     "save_checkpoint_epochs": [10, 90]}}
    assert milestone_epochs(cfg) == ([10, 30, 90], [10, 90])
    cfg["train"]["save_checkpoint_epochs"] = [60]
    with pytest.raises(ValueError, match="subset"):
        milestone_epochs(cfg)


def test_publish_validation_rejects_test_access_and_checks_milestones(tmp_path):
    result = {
        "run_id": "run", "selection_split": "ood_val", "test_evaluated": False,
        "acc_heldout": None, "worst_env_heldout": None, "acc_selection": 0.2,
        "acc_val": 0.2, "worst_env_val": 0.1, "acc_within": 0.3,
    }
    milestone = {
        "run_id": "run", "epoch": 10, "selection_split": "ood_val",
        "test_evaluated": False, "acc_train": 0.4, "acc_within": 0.3,
        "acc_selection": 0.2, "worst_env_val": 0.1,
    }
    path = tmp_path / "milestones.jsonl"
    path.write_text(json.dumps(milestone) + "\n")
    assert validate_stage1_artifacts(result, path) == [milestone]
    result["test_evaluated"] = True
    with pytest.raises(ValueError, match="test_evaluated=false"):
        validate_stage1_artifacts(result, path)
