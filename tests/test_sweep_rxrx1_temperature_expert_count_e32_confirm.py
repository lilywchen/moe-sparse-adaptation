from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_temperature_expert_count_e32_confirm as sweep


def test_e32_confirmation_is_locked_and_uses_shared_e8_anchors():
    rows = sweep.cells()
    assert len(rows) == 2
    assert len({row[2] for row in rows}) == 2
    comparators = sweep.comparator_run_ids()
    assert set(comparators) == {1, 2}

    for tag, overrides, run_id, comparator in rows:
        cfg = apply_overrides(load_config(sweep.CONFIG), overrides)
        seed = cfg["seed"]
        assert tag == f"route_E32_s{seed}"
        assert cfg["model"]["n_experts"] == 32
        assert cfg["model"]["pressure"] == "route"
        assert cfg["model"]["temperature"] == 0.03
        assert cfg["losses"]["balance_w"] == 0.0
        assert cfg["losses"]["zloss_w"] == 0.001
        assert cfg["train"]["epochs"] == 60
        assert cfg["train"]["milestone_epochs"] == [10, 30, 60]
        assert cfg["train"]["save_checkpoint_epochs"] == [10, 30, 60]
        assert comparator == comparators[seed]
        assert f"route_E8k1_ep60_s{seed}_temperature_expert_count_confirm60" in comparator
        assert f"route_E32k1_ep60_s{seed}_temperature_expert_count_confirm60" in run_id


def test_e32_launch_adapter_preserves_registry_identity():
    registered = [(tag, overrides, run_id) for tag, overrides, run_id, _ in sweep.cells()]
    assert sweep.launch_cells() == registered
