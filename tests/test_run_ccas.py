import pytest
import torch

import json

from scripts.run_ccas import (
    classification_objective, cross_experiment_contrastive_loss, milestone_epochs, router_gradient_norms,
    router_parameter_deltas, snapshot_routers, validate_publishable_artifacts,
    validate_stage1_artifacts,
)
from moe_shift.capacity import MoEFFN


class _Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _RouterAuditModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = MoEFFN(_Mlp(), n_experts=2, routing_estimator="selected_st")
        self._moe_blocks = [self.block]
        self.capacity = type("Capacity", (), {"block_indices": (3,)})()

    @property
    def moe_blocks(self):
        return tuple(self._moe_blocks)


def test_environment_balanced_objective_equalizes_experiments():
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    labels = torch.tensor([0, 0, 0])
    env = torch.tensor([0, 0, 1])
    per_example = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = 0.5 * (per_example[:2].mean() + per_example[2:].mean())
    assert classification_objective(logits, labels, env, "environment_balanced") == pytest.approx(expected)
    assert classification_objective(logits, labels, env, "erm") == pytest.approx(per_example.mean())


def test_cross_experiment_contrastive_uses_only_different_environment_positives():
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 0, 1, 1])
    environments = torch.tensor([0, 1, 0, 1])
    aligned = cross_experiment_contrastive_loss(features, labels, environments, temperature=0.1)
    shuffled = cross_experiment_contrastive_loss(
        features[[0, 2, 1, 3]], labels, environments, temperature=0.1)
    assert aligned < shuffled


def test_router_gradient_and_parameter_delta_audits():
    torch.manual_seed(0)
    model = _RouterAuditModel()
    initial = snapshot_routers(model)
    loss = model.block(torch.randn(3, 2, 4)).square().mean()
    norms = router_gradient_norms(loss, model)
    assert norms["3"] > 0
    with torch.no_grad():
        next(model.block.router.parameters()).add_(0.1)
    deltas = router_parameter_deltas(model, initial)
    assert deltas["3"]["l2"] > 0
    assert deltas["3"]["relative_l2"] > 0


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


def test_stage3_publication_requires_finite_test_and_keeps_milestones_blind(tmp_path):
    result = {
        "run_id": "run", "stage": 3, "selection_split": "ood_val",
        "test_evaluated": True, "acc_selection": 0.3, "acc_val": 0.3,
        "worst_env_val": 0.1, "acc_within": 0.5, "acc_heldout": 0.4,
        "worst_env_heldout": 0.08, "per_env_heldout": {1: 0.4, 2: 0.3},
        "per_env_n_heldout": {1: 10, 2: 10},
    }
    milestone = {
        "run_id": "run", "epoch": 30, "selection_split": "ood_val",
        "test_evaluated": False, "acc_train": 1.0, "acc_within": 0.5,
        "acc_selection": 0.3, "worst_env_val": 0.1,
    }
    path = tmp_path / "milestones.jsonl"
    path.write_text(json.dumps(milestone) + "\n")
    assert validate_publishable_artifacts(result, path) == [milestone]
    result["stage"] = 2
    with pytest.raises(ValueError, match="stage >= 3"):
        validate_publishable_artifacts(result, path)
