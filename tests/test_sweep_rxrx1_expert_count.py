from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_expert_count as sweep
from scripts import sweep_rxrx1_extreme_expert_aux as extreme
from scripts.sweep_rxrx1_expert_count import (
    AUX_SETTINGS,
    EXPERT_COUNTS,
    PRESSURES,
    cells,
)
from scripts.sweep_rxrx1_router_aux import sharded_rows


def test_expert_count60_has_complete_unique_budget():
    rows = cells()
    assert len(rows) == len(PRESSURES) * len(EXPERT_COUNTS) * len(AUX_SETTINGS) == 8
    assert len({run_id for _, _, run_id in rows}) == 8
    assert sum(tag.startswith("route_") for tag, _, _ in rows) == 4
    assert sum(tag.startswith("canonical_") for tag, _, _ in rows) == 4
    assert sweep.WANDB_JOB_TYPE == "rxrx1_expert_count60"
    assert "expert-count60" in sweep.WANDB_TAGS


def test_expert_count60_changes_only_predeclared_mechanism_fields():
    for tag, overrides, run_id in cells():
        pressure = tag.split("_", 1)[0]
        n_experts = next(n for n in EXPERT_COUNTS if f"_E{n}_" in tag)
        assert "seed=0" in overrides
        assert "stage=1" in overrides
        assert "model.variant=moe" in overrides
        assert "model.placement=early" in overrides
        assert "model.routing_unit=token" in overrides
        assert "model.geometry=cosine" in overrides
        assert f"model.pressure={pressure}" in overrides
        assert f"model.n_experts={n_experts}" in overrides
        assert "model.top_k=1" in overrides
        assert "train.epochs=60" in overrides
        assert "train.milestone_epochs=[10,30,60]" in overrides
        assert "train.save_checkpoint_epochs=[10,30,60]" in overrides
        assert "analysis.run_mechanism=true" in overrides
        assert any(v.startswith("losses.balance_w=") for v in overrides)
        assert any(v.startswith("losses.zloss_w=") for v in overrides)
        assert "expert_count60" in run_id


def test_expert_count60_is_not_mislabeled_exact_total_parameter_matched():
    counts = {
        int(next(
            v.split("=", 1)[1]
            for v in reversed(overrides)
            if v.startswith("model.n_experts=")
        ))
        for _, overrides, _ in cells()
    }
    assert counts == {4, 16}


def test_expert_count60_shards_are_disjoint_and_exhaustive():
    rows = cells()
    shards = [sharded_rows(rows, index, 5) for index in range(5)]
    flattened = [run_id for shard in shards for _, _, run_id in shard]
    assert len(flattened) == len(rows)
    assert set(flattened) == {run_id for _, _, run_id in rows}
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_extreme_expert_aux_is_complete_bounded_and_disjoint():
    rows = extreme.cells()
    assert len(rows) == 8
    assert len({run_id for _, _, run_id in rows}) == 8
    assert {tag for tag, _, _ in rows} == {
        f"{pressure}_E{n_experts}_{aux_label}"
        for pressure in extreme.PRESSURES
        for n_experts in extreme.EXPERT_COUNTS
        for aux_label, _, _ in extreme.AUX_SETTINGS
    }
    assert {run_id for _, _, run_id in rows}.isdisjoint(
        {run_id for _, _, run_id in cells()}
    )
    assert "active-compute-matched" in extreme.WANDB_TAGS


def test_extreme_expert_aux_locks_low_temperature_and_schedule():
    for tag, overrides, run_id in extreme.cells():
        pressure, expert_label, aux_label = tag.split("_", 2)
        cfg = apply_overrides(load_config(extreme.CONFIG), overrides)
        assert cfg["seed"] == 0
        assert cfg["stage"] == 1
        assert cfg["train"]["epochs"] == 60
        assert cfg["train"]["milestone_epochs"] == [10, 30, 60]
        assert cfg["train"]["save_checkpoint_epochs"] == [10, 30, 60]
        assert cfg["model"]["variant"] == "moe"
        assert cfg["model"]["placement"] == "early"
        assert cfg["model"]["routing_unit"] == "token"
        assert cfg["model"]["geometry"] == "cosine"
        assert cfg["model"]["pressure"] == pressure
        assert cfg["model"]["n_experts"] == int(expert_label[1:])
        assert cfg["model"]["top_k"] == 1
        assert cfg["model"]["temperature"] == 0.03
        expected = next(row for row in extreme.AUX_SETTINGS if row[0] == aux_label)
        assert cfg["losses"]["balance_w"] == expected[1]
        assert cfg["losses"]["zloss_w"] == expected[2]
        assert "extreme_expert_aux60" in run_id


def test_extreme_expert_aux_shards_are_disjoint_and_exhaustive():
    rows = extreme.cells()
    shards = [sharded_rows(rows, index, 8) for index in range(8)]
    assert all(len(shard) == 1 for shard in shards)
    assert {
        run_id for shard in shards for _, _, run_id in shard
    } == {run_id for _, _, run_id in rows}
