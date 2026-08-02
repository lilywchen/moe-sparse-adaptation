from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_temperature_expert_count as sweep
from scripts import sweep_rxrx1_temperature_expert_count_e32 as sweep_e32
from scripts import sweep_rxrx1_temperature_expert_count_e64 as sweep_e64
from scripts.sweep_rxrx1_router_aux import sharded_rows


def test_temperature_expert_count_has_complete_unique_budget():
    rows = sweep.cells()
    assert len(rows) == 6
    assert len({run_id for _, _, run_id in rows}) == 6
    assert {tag for tag, _, _ in rows} == {
        "canonical_E2", "canonical_E4", "canonical_E16",
        "route_E2", "route_E4", "route_E16",
    }
    assert sweep.WANDB_GROUP == (
        "rxrx1-cell-dino-temperature-expert-count60-20260802"
    )
    assert "active-compute-matched" in sweep.WANDB_TAGS


def test_temperature_expert_count_locks_geometry_and_varies_only_bounded_fields():
    for tag, overrides, run_id in sweep.cells():
        pressure, expert_label = tag.split("_")
        n_experts = int(expert_label[1:])
        cfg = apply_overrides(load_config(sweep.CONFIG), overrides)
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
        assert cfg["model"]["n_experts"] == n_experts
        assert cfg["model"]["top_k"] == 1
        assert cfg["model"]["temperature"] == 0.03
        assert cfg["losses"]["balance_w"] == 0.0
        assert cfg["losses"]["zloss_w"] == 0.001
        assert "temperature_expert_count60" in run_id


def test_temperature_expert_count_shards_are_disjoint_and_exhaustive():
    rows = sweep.cells()
    shards = [sharded_rows(rows, index, 6) for index in range(6)]
    flattened = [run_id for shard in shards for _, _, run_id in shard]
    assert len(flattened) == len(rows)
    assert set(flattened) == {run_id for _, _, run_id in rows}
    assert all(len(shard) == 1 for shard in shards)


def test_temperature_expert_count_e32_extension_is_bounded_and_disjoint():
    rows = sweep_e32.cells()
    assert len(rows) == 2
    assert {tag for tag, _, _ in rows} == {"canonical_E32", "route_E32"}
    assert {run_id for _, _, run_id in rows}.isdisjoint(
        {run_id for _, _, run_id in sweep.cells()}
    )
    for tag, overrides, run_id in rows:
        pressure, _ = tag.split("_")
        cfg = apply_overrides(load_config(sweep.CONFIG), overrides)
        assert cfg["seed"] == 0
        assert cfg["train"]["epochs"] == 60
        assert cfg["train"]["save_checkpoint_epochs"] == [10, 30, 60]
        assert cfg["model"]["pressure"] == pressure
        assert cfg["model"]["n_experts"] == 32
        assert cfg["model"]["top_k"] == 1
        assert cfg["model"]["temperature"] == 0.03
        assert cfg["losses"]["balance_w"] == 0.0
        assert cfg["losses"]["zloss_w"] == 0.001
        assert "temperature_expert_count60" in run_id


def test_temperature_expert_count_e64_extension_is_bounded_and_disjoint():
    rows = sweep_e64.cells()
    assert len(rows) == 2
    assert {tag for tag, _, _ in rows} == {"canonical_E64", "route_E64"}
    existing = {
        run_id
        for registry in (sweep.cells(), sweep_e32.cells())
        for _, _, run_id in registry
    }
    assert {run_id for _, _, run_id in rows}.isdisjoint(existing)
    for tag, overrides, run_id in rows:
        pressure, _ = tag.split("_")
        cfg = apply_overrides(load_config(sweep.CONFIG), overrides)
        assert cfg["seed"] == 0
        assert cfg["train"]["epochs"] == 60
        assert cfg["train"]["save_checkpoint_epochs"] == [10, 30, 60]
        assert cfg["model"]["pressure"] == pressure
        assert cfg["model"]["n_experts"] == 64
        assert cfg["model"]["top_k"] == 1
        assert cfg["model"]["temperature"] == 0.03
        assert cfg["losses"]["balance_w"] == 0.0
        assert cfg["losses"]["zloss_w"] == 0.001
        assert "temperature_expert_count60" in run_id
