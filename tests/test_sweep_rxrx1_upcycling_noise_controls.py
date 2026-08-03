from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_upcycling_noise_controls as sweep


def test_control_bank_is_unique_and_reuses_existing_learned_anchors():
    rows = sweep.cells()
    tags = [tag for tag, _, _ in rows]
    run_ids = [run_id for _, _, run_id in rows]

    assert len(rows) == 14
    assert len(tags) == len(set(tags))
    assert len(run_ids) == len(set(run_ids))
    assert sum(tag.startswith("learned_") for tag in tags) == 2
    assert sum(tag.startswith("frozen_") for tag in tags) == 6
    assert sum(tag.startswith("dense_") for tag in tags) == 6
    assert "learned_E4_temp03_noise0p001_tail_safe" in tags
    assert "learned_E16_temp03_noise0p001_tail_safe" in tags
    for shared in sweep.SHARED_LEARNED.values():
        assert not any(shared in tag for tag in tags)


def test_control_variants_and_noise_are_exact():
    for tag, overrides, run_id in sweep.cells():
        cfg = apply_overrides(load_config(sweep.CONFIG), overrides)
        assert cfg["model"]["temperature"] == 0.3
        assert cfg["losses"]["balance_w"] == 0.01
        assert cfg["losses"]["zloss_w"] == 0.01
        assert cfg["train"]["save_checkpoint_epochs"] == [10, 30, 60]
        assert "upcycling_noise_controls60" in run_id
        if tag.startswith("learned_"):
            assert cfg["model"]["variant"] == "moe"
            assert cfg["model"]["sym_break_moe"] in {0.001, 0.01}
        elif tag.startswith("frozen_"):
            assert cfg["model"]["variant"] == "moe_frozen"
            assert cfg["model"]["sym_break_moe"] in {0.001, 0.01}
        else:
            assert cfg["model"]["variant"] == "dense_wide"
            assert cfg["model"]["sym_break_moe"] == 0.0
            assert cfg["model"]["sym_break_wide"] in {0.001, 0.01}


def test_fourteen_shards_are_disjoint_and_exhaustive():
    rows = sweep.cells()
    shards = [sweep.base.refill.sharded_rows(rows, i, len(rows)) for i in range(len(rows))]
    assert all(len(shard) == 1 for shard in shards)
    assert {row[2] for shard in shards for row in shard} == {row[2] for row in rows}
