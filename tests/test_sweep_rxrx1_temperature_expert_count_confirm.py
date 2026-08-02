from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_temperature_expert_count_confirm as sweep


def test_confirmation_registry_is_locked_paired_and_unique():
    rows = sweep.cells()
    assert len(rows) == 4
    assert len({row[2] for row in rows}) == 4
    assert sweep.WANDB_GROUP == (
        "rxrx1-cell-dino-temperature-expert-count-confirm60-20260802"
    )
    assert sweep.WANDB_JOB_TYPE == "rxrx1_temperature_expert_count_confirm60"

    by_seed = {}
    for tag, overrides, run_id, comparator in rows:
        cfg = apply_overrides(load_config(sweep.CONFIG), overrides)
        seed = cfg["seed"]
        n_experts = cfg["model"]["n_experts"]
        by_seed.setdefault(seed, {})[n_experts] = run_id
        assert seed in {1, 2}
        assert n_experts in {8, 16}
        assert cfg["stage"] == 1
        assert cfg["train"]["epochs"] == 60
        assert cfg["train"]["milestone_epochs"] == [10, 30, 60]
        assert cfg["train"]["save_checkpoint_epochs"] == [10, 30, 60]
        assert cfg["model"]["variant"] == "moe"
        assert cfg["model"]["placement"] == "early"
        assert cfg["model"]["routing_unit"] == "token"
        assert cfg["model"]["geometry"] == "cosine"
        assert cfg["model"]["pressure"] == "route"
        assert cfg["model"]["temperature"] == 0.03
        assert cfg["losses"]["balance_w"] == 0.0
        assert cfg["losses"]["zloss_w"] == 0.001
        assert "temperature_expert_count_confirm60" in run_id
        assert f"s{seed}" in tag
        assert comparator.endswith(
            f"ep60_s{seed}_temperature_expert_count_confirm60_route_E8_s{seed}_20260802"
        )

    assert set(by_seed) == {1, 2}
    assert all(set(seed_rows) == {8, 16} for seed_rows in by_seed.values())


def test_launch_adapter_preserves_registry_identity_and_seed1_pair_order():
    registered = [(tag, overrides, run_id) for tag, overrides, run_id, _ in sweep.cells()]
    assert sweep.launch_cells() == registered
    assert [row[0] for row in registered[:2]] == ["route_E8_s1", "route_E16_s1"]
