from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_extreme_expert_aux as low_temp
from scripts import sweep_rxrx1_extreme_temperature_aux as sweep
from scripts.sweep_rxrx1_router_aux import sharded_rows


def test_extreme_temperature_addendum_is_bounded_unique_and_disjoint():
    rows = sweep.cells()
    assert len(rows) == 8
    assert len({run_id for _, _, run_id in rows}) == 8
    assert {tag for tag, _, _ in rows} == {
        f"{pressure}_E{n}_temp01_{aux}"
        for pressure in sweep.PRESSURES
        for n in sweep.EXPERT_COUNTS
        for aux, _, _ in sweep.AUX_SETTINGS
    }
    assert {run_id for _, _, run_id in rows}.isdisjoint(
        {run_id for _, _, run_id in low_temp.cells()}
    )
    assert "optimization-screen" in sweep.WANDB_TAGS


def test_extreme_temperature_addendum_changes_only_predeclared_fields():
    for tag, overrides, run_id in sweep.cells():
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
        assert cfg["model"]["pressure"] in sweep.PRESSURES
        assert cfg["model"]["top_k"] == 1
        assert cfg["model"]["temperature"] == sweep.TEMPERATURE
        assert cfg["model"]["n_experts"] in sweep.EXPERT_COUNTS
        aux = "tail_safe" if tag.endswith("tail_safe") else "no_aux"
        expected = next(row for row in sweep.AUX_SETTINGS if row[0] == aux)
        assert cfg["losses"]["balance_w"] == expected[1]
        assert cfg["losses"]["zloss_w"] == expected[2]
        assert "extreme_temperature_aux60" in run_id


def test_extreme_temperature_addendum_shards_are_one_cell_and_exhaustive():
    rows = sweep.cells()
    shards = [sharded_rows(rows, index, 8) for index in range(8)]
    assert all(len(shard) == 1 for shard in shards)
    assert {
        run_id for shard in shards for _, _, run_id in shard
    } == {run_id for _, _, run_id in rows}
