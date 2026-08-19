import importlib.util
from pathlib import Path

import torch

from moe_shift.capacity.batch_corrector import BatchFeatureCorrector
from moe_shift.capacity.naming import run_id_from
from moe_shift.data.rxrx1 import ExperimentBatchSampler


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_sampler_is_homogeneous_and_complete_at_eval():
    environments = [0, 0, 0, 1, 1, 1, 1]
    sampler = ExperimentBatchSampler(environments, batch_size=2, shuffle=False, drop_last=False)
    batches = list(sampler)
    assert len(batches) == 4
    assert sorted(index for batch in batches for index in batch) == list(range(7))
    assert all(len({environments[index] for index in batch}) == 1 for batch in batches)


def test_adabn_corrector_standardizes_each_environment():
    corrector = BatchFeatureCorrector(3, mode="adabn")
    features = torch.tensor([
        [1.0, 3.0, 5.0], [3.0, 5.0, 7.0],
        [10.0, 20.0, 30.0], [14.0, 24.0, 34.0],
    ])
    corrector.set_environment(torch.tensor([2, 2, 9, 9]))
    output = corrector(features)
    for environment in (2, 9):
        values = output[torch.tensor([2, 2, 9, 9]) == environment]
        assert torch.allclose(values.mean(0), torch.zeros(3), atol=1e-5)
        assert torch.allclose(values.square().mean(0), torch.ones(3), atol=1e-4)


def test_support_query_statistics_do_not_depend_on_queries():
    corrector = BatchFeatureCorrector(2, mode="adabn")
    support = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    mean = support.mean(0)
    std = ((support - mean).square().mean(0) + corrector.eps).sqrt()
    query = torch.tensor([[7.0, 14.0]])
    alone = corrector.forward_with_statistics(query, mean, std)
    with_extreme_peer = corrector.forward_with_statistics(
        torch.cat((query, torch.tensor([[1000.0, -1000.0]]))), mean, std)[:1]
    assert torch.allclose(alone, with_extreme_peer)


def test_dual_moe_is_function_preserving_at_initialization_and_trains():
    corrector = BatchFeatureCorrector(8, mode="moe_dual", n_experts=4, rank=2)
    features = torch.randn(12, 8)
    environments = torch.tensor([0] * 6 + [1] * 6)
    corrector.set_environment(environments)
    output = corrector(features)
    means = torch.cat((features[:6].mean(0).repeat(6, 1),
                       features[6:].mean(0).repeat(6, 1)))
    stds = torch.cat((((features[:6] - features[:6].mean(0)).square().mean(0)
                       + corrector.eps).sqrt().repeat(6, 1),
                      ((features[6:] - features[6:].mean(0)).square().mean(0)
                       + corrector.eps).sqrt().repeat(6, 1)))
    expected = (features - means) / stds
    assert torch.allclose(output, expected, atol=1e-5)
    (output.square().mean() + corrector.aux_loss(0.01, 0.001)).backward()
    assert corrector.up.grad is not None
    assert torch.isfinite(corrector.up.grad).all()


def test_campaign_is_unique_predeclared_and_balanced():
    sweep = _load_script("sweep_rxrx1_batch_correctors.py")
    rows = sweep.campaign_rows()
    assert len(rows) == 36
    assert len({row["run_id"] for row in rows}) == 36
    assert sum(row["phase"] == "confirmatory" for row in rows) == 12
    assert {row["label"] for row in rows if row["phase"] == "confirmatory"} == {
        "original_grouped", "HarmonyDG", "H2_adabn", "TransportMoE"}
    assert all(row["stage"] == 3 for row in rows if row["phase"] == "confirmatory")
    assert all(row["stage"] == 1 for row in rows if row["phase"] != "confirmatory")
    assert all(sum(index % 4 == shard for index in range(len(rows))) == 9
               for shard in range(4))


def test_corrector_changes_run_identity():
    base = {
        "dataset": "rxrx1", "seed": 0,
        "model": {"variant": "original", "placement": "middle", "n_experts": 8,
                  "batch_corrector": {"mode": "none"}},
        "train": {"epochs": 12},
    }
    changed = {**base, "model": {**base["model"], "batch_corrector": {
        "mode": "moe_dual", "n_experts": 4, "rank": 16}}}
    assert run_id_from(base) != run_id_from(changed)
    assert "bc-moe_dual-E4r16" in run_id_from(changed)
