from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_expert_count_aux_confirm as sweep


def test_auxiliary_confirmation_registry_is_locked_paired_and_unique():
    rows = sweep.cells()
    assert len(rows) == 4
    assert len({row[2] for row in rows}) == 4
    assert sweep.WANDB_GROUP == (
        "rxrx1-cell-dino-expert-count-aux-confirm60-20260802"
    )
    assert sweep.HF_PREFIX == (
        "rxrx1/cell_dino_cp5/expert_count_aux_confirm60_20260802"
    )
    assert sweep.WANDB_JOB_TYPE == "rxrx1_expert_count_aux_confirm60"

    by_seed = {}
    for tag, overrides, run_id, comparator in rows:
        cfg = apply_overrides(load_config(sweep.CONFIG), overrides)
        seed = cfg["seed"]
        balance_w = cfg["losses"]["balance_w"]
        zloss_w = cfg["losses"]["zloss_w"]
        by_seed.setdefault(seed, set()).add((balance_w, zloss_w))
        assert seed in {1, 2}
        assert cfg["stage"] == 1
        assert cfg["train"]["epochs"] == 60
        assert cfg["train"]["milestone_epochs"] == [10, 30, 60]
        assert cfg["train"]["save_checkpoint_epochs"] == [10, 30, 60]
        assert cfg["model"]["variant"] == "moe"
        assert cfg["model"]["placement"] == "early"
        assert cfg["model"]["routing_unit"] == "token"
        assert cfg["model"]["geometry"] == "cosine"
        assert cfg["model"]["pressure"] == "canonical"
        assert cfg["model"]["n_experts"] == 4
        assert cfg["analysis"]["run_mechanism"] is True
        assert "expert_count_aux_confirm60" in run_id
        assert f"s{seed}" in tag
        assert comparator.endswith(
            f"ep60_s{seed}_expert_count_aux_confirm60_canonical_E4_zero_aux_s{seed}_20260802"
        )

    expected = {(0.01, 0.01), (0.0, 0.0)}
    assert by_seed == {1: expected, 2: expected}


def test_launch_adapter_preserves_pair_order():
    registered = [(tag, overrides, run_id) for tag, overrides, run_id, _ in sweep.cells()]
    assert sweep.launch_cells() == registered
    assert [row[0] for row in registered] == [
        "canonical_E4_tail_safe_s1",
        "canonical_E4_zero_aux_s1",
        "canonical_E4_tail_safe_s2",
        "canonical_E4_zero_aux_s2",
    ]
