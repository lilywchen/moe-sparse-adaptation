from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_temperature03_moderate_bank_aux as anchors
from scripts import sweep_rxrx1_upcycling_noise as sweep
from scripts.sweep_rxrx1_router_aux import sharded_rows


def test_upcycling_noise_family_is_bounded_unique_and_not_anchor_duplicate():
    rows = sweep.cells()
    assert len(rows) == 8
    assert len({run_id for _, _, run_id in rows}) == 8
    anchor_ids = {run_id for _, _, run_id in anchors.cells()}
    assert {run_id for _, _, run_id in rows}.isdisjoint(anchor_ids)
    assert all("noise" in tag and tag.endswith("tail_safe") for tag, _, _ in rows)


def test_upcycling_noise_changes_only_predeclared_fields():
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
        assert cfg["model"]["top_k"] == 1
        assert cfg["model"]["pressure"] in {"route", "canonical"}
        assert cfg["model"]["temperature"] == sweep.TEMPERATURE
        assert cfg["model"]["n_experts"] in {4, 8, 16}
        assert cfg["model"]["sym_break_moe"] in {1.0e-3, 1.0e-2}
        assert cfg["losses"]["balance_w"] == sweep.BALANCE_W
        assert cfg["losses"]["zloss_w"] == sweep.ZLOSS_W
        assert "upcycling_noise60" in run_id


def test_upcycling_noise_family_shards_one_cell_exhaustively():
    rows = sweep.cells()
    shards = [sharded_rows(rows, index, 8) for index in range(8)]
    assert all(len(shard) == 1 for shard in shards)
    assert {run_id for shard in shards for _, _, run_id in shard} == {
        run_id for _, _, run_id in rows
    }
