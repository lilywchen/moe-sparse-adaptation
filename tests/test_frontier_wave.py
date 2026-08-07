"""Guards for the frontier-MoE wave: objective, BTX, run identity, sweeper, aggregator.

These protect against the failures that would make the wave produce numbers that LOOK valid:
colliding run ids (the sweep silently skips the second arm), a replacement variant sneaking back
in as a confound, mechanism auditing switched off, and specialists being compared with full-data
arms.
"""
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from moe_shift.capacity import btx
from moe_shift.capacity.ffn import SharedResidualMoEFFN
from moe_shift.capacity.frontier import LowRankResidualMoEFFN
from moe_shift.capacity.naming import run_id_from

DIM, HIDDEN = 32, 64


class Mlp(nn.Module):
    def __init__(self, c=DIM, h=HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(c, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, c)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ------------------------------------------------------------------ GroupDRO
def _group_dro():
    from run_ccas import GroupDRO
    return GroupDRO


def test_group_dro_upweights_the_worst_group():
    """The whole point: the group with the larger loss must gain weight."""
    dro = _group_dro()(n_groups=3, step_size=1.0)
    per_example = torch.tensor([0.1, 0.1, 5.0, 5.0])
    env = torch.tensor([0, 0, 1, 1])
    dro(per_example, env)
    weights = dro.q.tolist()
    assert weights[1] > weights[0], "the high-loss group must be upweighted"
    assert weights[1] > weights[2], "an absent group must not overtake the worst present group"


def test_group_dro_loss_lies_between_group_losses():
    dro = _group_dro()(n_groups=2, step_size=0.01)
    per_example = torch.tensor([0.0, 0.0, 4.0, 4.0])
    env = torch.tensor([0, 0, 1, 1])
    loss = float(dro(per_example, env))
    assert 0.0 <= loss <= 4.0
    assert loss > 0.0, "a reweighting must not zero out the worst group"


def test_group_dro_weights_stay_normalised():
    dro = _group_dro()(n_groups=4, step_size=0.5)
    for _ in range(20):
        dro(torch.rand(8), torch.randint(0, 4, (8,)))
    assert float(dro.q.sum()) == pytest.approx(1.0, abs=1e-5)
    assert float(dro.q.min()) >= 0.0


def test_group_dro_rejects_negative_and_out_of_range_environments():
    dro = _group_dro()(n_groups=2, step_size=0.1)
    with pytest.raises(ValueError, match="non-negative"):
        dro(torch.rand(2), torch.tensor([-1, 0]))
    with pytest.raises(ValueError, match="n_groups"):
        dro(torch.rand(2), torch.tensor([0, 7]))


def test_group_dro_requires_at_least_two_groups():
    with pytest.raises(ValueError, match="at least two"):
        _group_dro()(n_groups=1)


def test_group_dro_state_is_serialisable():
    dro = _group_dro()(n_groups=3, step_size=0.1)
    dro(torch.rand(6), torch.randint(0, 3, (6,)))
    json.dumps(dro.state())          # must not raise


def test_objective_dispatch_and_label_smoothing():
    from run_ccas import OBJECTIVES, classification_objective

    assert set(OBJECTIVES) == {"erm", "environment_balanced", "group_dro"}
    logits = torch.randn(6, 5)
    labels = torch.randint(0, 5, (6,))
    env = torch.tensor([0, 0, 1, 1, 2, 2])
    plain = float(classification_objective(logits, labels, env, "erm"))
    smoothed = float(classification_objective(logits, labels, env, "erm", label_smoothing=0.1))
    assert plain != smoothed, "label smoothing must change the loss"
    with pytest.raises(ValueError, match="group_dro requires"):
        classification_objective(logits, labels, env, "group_dro")
    with pytest.raises(ValueError, match="unknown train.objective"):
        classification_objective(logits, labels, env, "nonsense")


def test_batch_group_ids_selects_the_right_column():
    from run_ccas import batch_group_ids

    batch = (torch.zeros(2), torch.zeros(2), torch.tensor([5, 6]),
             torch.tensor([70, 71]), torch.tensor([1, 2]))
    assert batch_group_ids(batch, None) is None
    assert batch_group_ids(batch, "cell_type").tolist() == [1, 2]
    # environment routing uses the TRAIN-REMAPPED site, which is -1 on OOD rows by design
    assert batch_group_ids(batch, "environment").tolist() == [5, 6]
    with pytest.raises(ValueError, match="unknown group_source"):
        batch_group_ids(batch, "plate")
    with pytest.raises(ValueError, match="does not expose cell_type"):
        batch_group_ids(batch[:4], "cell_type")


# ------------------------------------------------------------------ BTX
def test_cluster_environments_partitions_every_environment_exactly_once():
    vectors = {10: [1.0, 0.0], 11: [0.99, 0.01], 20: [0.0, 1.0], 21: [0.01, 0.99]}
    clusters = btx.cluster_environments(vectors, n_clusters=2, seed=0)
    members = sorted(e for group in clusters.values() for e in group)
    assert members == [10, 11, 20, 21]
    assert sum(len(v) for v in clusters.values()) == 4, "no environment may be duplicated"
    grouped = {frozenset(v) for v in clusters.values()}
    assert grouped == {frozenset({10, 11}), frozenset({20, 21})}


def test_cluster_indices_are_contiguous_even_when_a_cluster_is_empty():
    """n_clusters is an upper bound; empty clusters must not leave gaps in the expert bank."""
    vectors = {i: [float(i == 0), float(i != 0)] for i in range(4)}
    clusters = btx.cluster_environments(vectors, n_clusters=4, seed=0)
    assert sorted(clusters) == list(range(len(clusters)))


def test_cluster_environments_is_deterministic_for_a_seed():
    vectors = {i: [float(i), float(i * i % 7), 1.0] for i in range(12)}
    first = btx.cluster_environments(vectors, n_clusters=3, seed=7)
    second = btx.cluster_environments(vectors, n_clusters=3, seed=7)
    assert first == second, "cluster identity is part of the run's scientific identity"


def test_cluster_environments_requires_two_environments():
    with pytest.raises(ValueError, match="at least two"):
        btx.cluster_environments({5: [1.0]}, n_clusters=2)


def test_cluster_from_conflict_matrix_groups_agreeing_environments():
    matrix = {
        "0": {"0": 1.0, "1": 0.9, "2": -0.8, "3": -0.9},
        "1": {"0": 0.9, "1": 1.0, "2": -0.85, "3": -0.8},
        "2": {"0": -0.8, "1": -0.85, "2": 1.0, "3": 0.95},
        "3": {"0": -0.9, "1": -0.8, "2": 0.95, "3": 1.0},
    }
    clusters = btx.cluster_from_conflict_matrix(matrix, n_clusters=2, seed=0)
    grouped = {frozenset(v) for v in clusters.values()}
    assert grouped == {frozenset({0, 1}), frozenset({2, 3})}


def test_load_clusters_reindexes_contiguously(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps({"clusters": {"3": [1, 2], "7": [5]}}))
    clusters = btx.load_clusters(path)
    assert clusters == {0: [1, 2], 1: [5]}


def test_state_for_block_finds_chunked_cell_dino_paths():
    """Cell-DINO stores blocks inside BlockChunk containers, so the prefix is not fixed."""
    state = {
        "backbone.blocks.0.1.mlp.fc1.weight": torch.zeros(2, 2),
        "backbone.blocks.11.mlp.fc1.weight": torch.ones(3, 3),
        "backbone.blocks.11.mlp.fc2.bias": torch.ones(3),
        "fc.weight": torch.zeros(1),
    }
    extracted = btx._state_for_block(state, 11)
    assert set(extracted) == {"fc1.weight", "fc2.bias"}
    with pytest.raises(KeyError, match="no 'blocks.4.mlp"):
        btx._state_for_block(state, 4)


def test_mix_specialists_into_block_loads_distinct_experts_and_freezes_them():
    block = SharedResidualMoEFFN(Mlp(), n_experts=2, top_k=1)
    donors = [Mlp(), Mlp()]
    states = [{k: v.clone() for k, v in donor.state_dict().items()} for donor in donors]
    report = btx.mix_specialists_into_block(block, states, freeze_experts=True)

    assert report["n_experts_filled"] == 2
    assert report["experts_frozen"] is True
    assert report["function_preserving_at_init"] is False
    for expert, donor in zip(block.experts, donors):
        assert torch.allclose(expert.fc2.weight, donor.fc2.weight)
        assert all(not p.requires_grad for p in expert.parameters())
    # The specialists were independent, so the bank must not be N copies of one function.
    assert not torch.allclose(block.experts[0].fc2.weight, block.experts[1].fc2.weight)
    # The shared path must remain trainable: it is the anchor to the pretrained model.
    assert all(p.requires_grad for p in block.shared.parameters())


def test_mix_specialists_rejects_wrong_expert_count():
    block = SharedResidualMoEFFN(Mlp(), n_experts=3, top_k=1)
    states = [Mlp().state_dict()]
    with pytest.raises(ValueError, match="cannot fill"):
        btx.mix_specialists_into_block(block, states)


def test_mix_specialists_rejects_lowrank_target():
    """Low-rank experts have no FFN-shaped tensors to receive specialist weights."""
    block = LowRankResidualMoEFFN(Mlp(), n_experts=2, top_k=1, expert_rank=8)
    with pytest.raises(TypeError, match="low-rank"):
        btx.mix_specialists_into_block(block, [Mlp().state_dict(), Mlp().state_dict()])


def test_write_manifest_round_trips(tmp_path):
    path = tmp_path / btx.MANIFEST_NAME
    btx.write_manifest(path, {0: [1, 2], 1: [3]},
                       [{"cluster": 0, "checkpoint": "a.pt"},
                        {"cluster": 1, "checkpoint": "b.pt"}],
                       "feature_mean", extra={"specialist_epochs": 5})
    payload = json.loads(path.read_text())
    assert payload["n_clusters"] == 2
    assert payload["specialist_epochs"] == 5
    assert btx.load_clusters(path) == {0: [1, 2], 1: [3]}


# ------------------------------------------------------------------ run identity
def _cfg(model, train=None):
    return {
        "dataset": "rxrx1", "seed": 0,
        "model": model,
        "train": {"epochs": 30, **(train or {})},
    }


def test_frontier_run_ids_are_unique_across_every_arm():
    """Two arms sharing a run id would make the sweep skip the second one SILENTLY."""
    arms = [
        _cfg({"variant": "oracle_moe", "n_experts": 4, "top_k": 1, "ffn_block_indices": [10, 11],
              "group_source": "cell_type", "expert_dropout": 0.5}),
        _cfg({"variant": "oracle_moe", "n_experts": 33, "top_k": 1, "ffn_block_indices": [10, 11],
              "group_source": "environment", "expert_dropout": 0.5}),
        _cfg({"variant": "condln_moe", "n_experts": 8, "top_k": 1,
              "ffn_block_indices": [10, 11], "geometry": "cosine"}),
        _cfg({"variant": "lowrank_moe", "n_experts": 24, "top_k": 8, "expert_rank": 16,
              "ffn_block_indices": [10, 11], "routing_unit": "token", "geometry": "cosine"}),
        _cfg({"variant": "soft_moe", "n_experts": 8, "top_k": 1, "slots_per_expert": 1,
              "expert_rank": 0, "ffn_block_indices": [10, 11], "geometry": "cosine"}),
        _cfg({"variant": "lowrank_moe", "n_experts": 24, "top_k": 8, "expert_rank": 16,
              "diversity_w": 0.05, "ffn_block_indices": [10, 11],
              "routing_unit": "token", "geometry": "cosine"},
             {"anneal_top_k_epochs": 10}),
        _cfg({"variant": "shared_moe", "n_experts": 3, "top_k": 1,
              "ffn_block_indices": [10, 11], "routing_unit": "token", "geometry": "cosine"},
             {"objective": "group_dro"}),
        _cfg({"variant": "shared_moe", "n_experts": 4, "top_k": 1,
              "ffn_block_indices": [10, 11], "routing_unit": "token", "geometry": "cosine",
              "btx_manifest": "/tmp/btx_manifest.json"}),
    ]
    ids = [run_id_from(cfg) for cfg in arms]
    assert len(set(ids)) == len(ids), f"colliding run ids: {ids}"


@pytest.mark.parametrize("changed", [
    {"diversity_w": 0.05},
    {"expert_rank": 32},
    {"n_experts": 16},
    {"top_k": 4},
])
def test_lowrank_identity_responds_to_every_functional_factor(changed):
    base = {"variant": "lowrank_moe", "n_experts": 24, "top_k": 8, "expert_rank": 16,
            "ffn_block_indices": [10, 11], "routing_unit": "token", "geometry": "cosine"}
    assert run_id_from(_cfg(base)) != run_id_from(_cfg({**base, **changed}))


def test_btx_and_environment_subset_appear_in_run_identity():
    base = {"variant": "shared_moe", "n_experts": 4, "top_k": 1,
            "ffn_block_indices": [10, 11], "routing_unit": "token", "geometry": "cosine"}
    plain = run_id_from(_cfg(base))
    mixed = run_id_from(_cfg({**base, "btx_manifest": "/tmp/m.json"}))
    unfrozen = run_id_from(_cfg({**base, "btx_manifest": "/tmp/m.json",
                                 "btx_freeze_experts": False}))
    specialist = run_id_from(_cfg({"variant": "original", "placement": "middle"},
                                  {"environment_subset": [1, 2, 3]}))
    assert len({plain, mixed, unfrozen, specialist}) == 4
    assert "btx" in mixed and "btxopen" in unfrozen
    assert "envsub3" in specialist


def test_group_dro_and_label_smoothing_appear_in_run_identity():
    base = {"variant": "shared_moe", "n_experts": 3, "top_k": 1,
            "ffn_block_indices": [10, 11], "routing_unit": "token", "geometry": "cosine"}
    plain = run_id_from(_cfg(base))
    dro = run_id_from(_cfg(base, {"objective": "group_dro"}))
    smoothed = run_id_from(_cfg(base, {"label_smoothing": 0.1}))
    assert plain != dro != smoothed and plain != smoothed
    assert "dro" in dro and "ls01" in smoothed


# ------------------------------------------------------------------ sweeper
def _sweeper():
    import sweep_rxrx1_frontier_moe as sweeper
    return sweeper


def test_wave_has_exactly_eight_arms_with_unique_labels():
    sweeper = _sweeper()
    labels = [label for label, _entry, _ov in sweeper.SPECS]
    assert len(labels) == 8
    assert len(set(labels)) == 8


def test_wave_rows_build_and_enforce_protocol():
    sweeper = _sweeper()
    rows = sweeper.wave_rows(str(ROOT / sweeper.CONFIG))
    assert len(rows) == 8
    run_ids = [run_id for _l, _e, _o, run_id, _c in rows]
    assert len(set(run_ids)) == 8, "arms must not collide on run id"
    for label, entry, _overrides, _run_id, cfg in rows:
        assert entry in ("run", "btx")
        assert cfg["model"]["variant"] in sweeper.SHARED_PATH_VARIANTS, label
        assert cfg["analysis"]["run_mechanism"] is True, label
        assert cfg["train"]["label_smoothing"] == 0.1, label
        assert cfg["train"]["epochs"] == 30, label
        assert cfg["seed"] == 0, label
        assert cfg["stage"] == 3, label
        assert cfg["model"]["ffn_block_indices"] == [10, 11], label


def test_wave_includes_both_ceilings_and_the_estimator_fixes():
    sweeper = _sweeper()
    variants = {label: overrides for label, _e, overrides in sweeper.SPECS}
    assert "oracle_cell_type" in variants and "oracle_environment" in variants
    assert any("model.variant=soft_moe" in ov for ov in variants.values())
    assert any("model.variant=condln_moe" in ov for ov in variants.values())
    assert any("model.diversity_w=0.05" in ov for ov in variants.values())
    assert any("train.objective=group_dro" in ov for ov in variants.values())
    assert any("train.anneal_top_k_epochs=10" in ov for ov in variants.values())


def test_shards_partition_the_wave_without_overlap():
    sweeper = _sweeper()
    rows = sweeper.wave_rows(str(ROOT / sweeper.CONFIG))
    shards = [sweeper.sharded_rows(rows, i, 4) for i in range(4)]
    flattened = [row[3] for shard in shards for row in shard]
    assert sorted(flattened) == sorted(row[3] for row in rows)
    assert len(set(flattened)) == len(flattened)
    assert all(len(shard) == 2 for shard in shards), "8 arms over 4 shards is 2 each"


def test_btx_arm_runs_the_pipeline_entry_point(tmp_path):
    sweeper = _sweeper()
    rows = {label: (label, entry, ov, rid, cfg)
            for label, entry, ov, rid, cfg in sweeper.wave_rows(str(ROOT / sweeper.CONFIG))}
    btx_row = rows["btx_specialists"]
    assert btx_row[1] == "btx"
    command = sweeper.command_for(btx_row, sweeper.CONFIG, tmp_path)
    assert "scripts/btx_rxrx1.py" in command and "run-all" in command
    run_row = rows["condln_stats"]
    assert "scripts/run_ccas.py" in sweeper.command_for(run_row, sweeper.CONFIG, tmp_path)


def test_manifest_records_the_held_constant_factors(tmp_path):
    sweeper = _sweeper()
    rows = sweeper.wave_rows(str(ROOT / sweeper.CONFIG))
    payload = sweeper.write_manifest(tmp_path, rows)
    assert payload["placement_fixed"] == [10, 11]
    assert payload["mechanism_auditing"] is True
    assert payload["label_smoothing_fixed"] == 0.1
    assert len(payload["runs"]) == 8
    assert (tmp_path / "wave_manifest.json").is_file()


def test_render_table_handles_a_partially_complete_wave(tmp_path):
    sweeper = _sweeper()
    rows = sweeper.wave_rows(str(ROOT / sweeper.CONFIG))
    sweeper.write_manifest(tmp_path, rows)
    table = sweeper.render_table(tmp_path)
    assert "0/8 complete" in table
    for label, _e, _o, _rid, _c in rows:
        assert label in table


# ------------------------------------------------------------------ aggregator
def _aggregator():
    import aggregate_frontier_moe as aggregator
    return aggregator


def _fake_result(label, **overrides):
    result = {
        "run_id": f"rid_{label}", "variant": "condln_moe",
        "acc_val": 0.23, "acc_heldout": 0.39, "worst_env_heldout": 0.085,
        "acc_within": 0.56, "acc_train": 1.0,
        "selection_split": "ood_val", "test_evaluated": True,
        "route_reliance": 0.004, "randomized_routes_acc": 0.226,
        "experts_used": 8.0, "routing_entropy": 0.9,
        "expert_output_cosine": 0.97,
        "protocol": {"pretrained_shared_expert_always_active": True},
    }
    result.update(overrides)
    return result


def _write_wave(tmp_path, results):
    sweeper = _sweeper()
    rows = sweeper.wave_rows(str(ROOT / sweeper.CONFIG))
    sweeper.write_manifest(tmp_path, rows)
    manifest = json.loads((tmp_path / "wave_manifest.json").read_text())
    by_label = {entry["label"]: entry["run_id"] for entry in manifest["runs"]}
    for label, result in results.items():
        (tmp_path / f"{by_label[label]}.json").write_text(json.dumps(result))
    return manifest


def test_aggregator_loads_only_completed_arms(tmp_path):
    aggregator = _aggregator()
    _write_wave(tmp_path, {"condln_stats": _fake_result("condln_stats")})
    _root, manifest, results = aggregator.load_wave(tmp_path)
    assert len(manifest["runs"]) == 8
    assert set(results) == {"condln_stats"}


def test_aggregator_flags_missing_mechanism_audit(tmp_path):
    aggregator = _aggregator()
    broken = _fake_result("condln_stats")
    for key in ("route_reliance", "randomized_routes_acc"):
        broken.pop(key)
    _write_wave(tmp_path, {"condln_stats": broken})
    _root, _manifest, results = aggregator.load_wave(tmp_path)
    problems = aggregator.validate(results)
    assert any("routing counterfactual" in problem for problem in problems)


def test_aggregator_flags_sealed_test_and_bad_accuracy(tmp_path):
    aggregator = _aggregator()
    _write_wave(tmp_path, {
        "condln_stats": _fake_result("condln_stats", test_evaluated=False, acc_heldout=None),
    })
    _root, _manifest, results = aggregator.load_wave(tmp_path)
    problems = aggregator.validate(results)
    assert any("test_evaluated is false" in problem for problem in problems)
    assert any("acc_heldout" in problem for problem in problems)


def test_aggregator_flags_environment_subset_arms_as_incomparable(tmp_path):
    aggregator = _aggregator()
    specialist = _fake_result("condln_stats")
    specialist["protocol"]["trained_on_environment_subset"] = True
    _write_wave(tmp_path, {"condln_stats": specialist})
    _root, _manifest, results = aggregator.load_wave(tmp_path)
    assert any("environment subset" in p for p in aggregator.validate(results))


def test_aggregator_flags_replacement_variant(tmp_path):
    aggregator = _aggregator()
    replacement = _fake_result("condln_stats", variant="moe")
    replacement["protocol"]["pretrained_shared_expert_always_active"] = False
    _write_wave(tmp_path, {"condln_stats": replacement})
    _root, _manifest, results = aggregator.load_wave(tmp_path)
    assert any("dense FFN active" in p for p in aggregator.validate(results))


def test_aggregator_reports_reliance_gate_verdicts(tmp_path):
    aggregator = _aggregator()
    _write_wave(tmp_path, {
        "condln_stats": _fake_result("condln_stats", route_reliance=0.004),
        "soft_moe_E8": _fake_result("soft_moe_E8", route_reliance=0.031,
                                    expert_output_cosine=0.42),
    })
    _root, _manifest, results = aggregator.load_wave(tmp_path)
    text = "\n".join(aggregator.render_mechanism(results))
    assert "below reliance gate" in text
    assert "RELIANCE GATE PASSED" in text
    assert "experts near-interchangeable" in text
    assert "experts differentiated" in text


def test_aggregator_detects_a_vanished_router_gradient(tmp_path):
    aggregator = _aggregator()
    result = _fake_result("condln_stats", router_grad_norm_by_epoch={
        "0": {"10": 1.0, "11": 1.0}, "29": {"10": 1e-6, "11": 1e-6},
    })
    _write_wave(tmp_path, {"condln_stats": result})
    _root, _manifest, results = aggregator.load_wave(tmp_path)
    assert "VANISHED" in "\n".join(aggregator.render_mechanism(results))


def test_aggregator_renders_ceilings_and_contrasts(tmp_path):
    aggregator = _aggregator()
    _write_wave(tmp_path, {
        "oracle_cell_type": _fake_result("oracle_cell_type", variant="oracle_moe",
                                         acc_val=0.28, shared_only_acc=0.24,
                                         oracle_expert_contribution=0.04),
        "condln_stats": _fake_result("condln_stats"),
    })
    _root, manifest, results = aggregator.load_wave(tmp_path)
    ceilings = "\n".join(aggregator.render_ceilings(results))
    assert "oracle_cell_type" in ceilings and "shared-only" in ceilings
    assert "ceiling clears dense" in ceilings
    assert "oracle_environment: not complete" in ceilings
    # Default reference is external, and must be labelled as coming from another commit.
    contrasts = "\n".join(aggregator.render_contrasts(results))
    assert aggregator.DEFAULT_REFERENCE in contrasts
    assert "DIFFERENT commit" in contrasts
    # An in-wave label is honoured when it is complete...
    in_wave = "\n".join(aggregator.render_contrasts(results, "condln_stats"))
    assert "reference: condln_stats" in in_wave
    # ...and falls back to the external reference when it is not.
    missing = "\n".join(aggregator.render_contrasts(results, "shared_E3k1_dro"))
    assert "not complete" in missing and aggregator.DEFAULT_REFERENCE in missing
    table = "\n".join(aggregator.render_table(manifest, results))
    assert "condln_stats" in table and "OOD test" in table
