from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_temperature_leader_confirm as sweep


def test_temperature_leader_confirmation_is_locked_and_uses_shared_dense_anchors():
    rows = sweep.cells()
    assert len(rows) == 2
    assert len({row[2] for row in rows}) == 2
    assert sweep.WANDB_GROUP == "rxrx1-cell-dino-temperature-leader-confirm60-20260802"
    assert sweep.WANDB_JOB_TYPE == "rxrx1_temperature_leader_confirm60"
    comparators = sweep.comparator_run_ids()
    assert set(comparators) == {1, 2}

    for tag, overrides, run_id, comparator in rows:
        seed = int(tag.rsplit("s", 1)[1])
        cfg = apply_overrides(load_config(sweep.CONFIG), overrides)
        assert cfg["seed"] == seed
        assert cfg["stage"] == 1
        assert cfg["train"]["epochs"] == 60
        assert cfg["train"]["milestone_epochs"] == [10, 30, 60]
        assert cfg["train"]["save_checkpoint_epochs"] == [10, 30, 60]
        assert cfg["model"]["variant"] == "moe"
        assert cfg["model"]["placement"] == "early"
        assert cfg["model"]["routing_unit"] == "token"
        assert cfg["model"]["geometry"] == "cosine"
        assert cfg["model"]["pressure"] == "canonical"
        assert cfg["model"]["temperature"] == 0.03
        assert cfg["losses"]["balance_w"] == 0.0
        assert cfg["losses"]["zloss_w"] == 0.001
        assert "temperature_leader_confirm60" in run_id
        assert comparator == comparators[seed]
        assert f"ep60_s{seed}_tail_safe_confirm60_dense_s{seed}" in comparator


def test_temperature_leader_launch_adapter_preserves_registry_identity():
    registered = [(tag, overrides, run_id) for tag, overrides, run_id, _ in sweep.cells()]
    assert sweep.launch_cells() == registered
