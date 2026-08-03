from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_temperature03_aux as route_sweep
from scripts import sweep_rxrx1_temperature03_canonical_aux as sweep
from scripts.sweep_rxrx1_router_aux import sharded_rows


def test_temperature03_canonical_controls_are_bounded_unique_and_disjoint():
    rows = sweep.cells()
    assert len(rows) == 4
    assert len({run_id for _, _, run_id in rows}) == 4
    assert {run_id for _, _, run_id in rows}.isdisjoint(
        {run_id for _, _, run_id in route_sweep.cells()}
    )
    assert {tag for tag, _, _ in rows} == {
        f"canonical_E{n}_temp03_{aux}"
        for n in sweep.EXPERT_COUNTS
        for aux, _, _ in sweep.AUX_SETTINGS
    }
    assert "canonical-pressure" in sweep.WANDB_TAGS


def test_temperature03_canonical_controls_change_only_predeclared_fields():
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
        assert cfg["model"]["pressure"] == sweep.PRESSURE
        assert cfg["model"]["top_k"] == 1
        assert cfg["model"]["temperature"] == sweep.TEMPERATURE
        assert cfg["model"]["n_experts"] in sweep.EXPERT_COUNTS
        aux = "tail_safe" if tag.endswith("tail_safe") else "no_aux"
        expected = next(row for row in sweep.AUX_SETTINGS if row[0] == aux)
        assert cfg["losses"]["balance_w"] == expected[1]
        assert cfg["losses"]["zloss_w"] == expected[2]
        assert "temperature03_canonical_aux60" in run_id


def test_temperature03_canonical_controls_shard_one_cell_exhaustively():
    rows = sweep.cells()
    shards = [sharded_rows(rows, index, 4) for index in range(4)]
    assert all(len(shard) == 1 for shard in shards)
    assert {
        run_id for shard in shards for _, _, run_id in shard
    } == {run_id for _, _, run_id in rows}
