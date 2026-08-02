from scripts.sweep_rxrx1_cell_dino import cells, sharded_rows


def test_factorial60_has_complete_unique_cell_budget():
    rows = cells()
    assert len(rows) == 43
    assert len({run_id for _, _, run_id in rows}) == 43
    assert sum(tag.startswith("moe_") for tag, _, _ in rows) == 36
    assert sum(tag.startswith("dense_") for tag, _, _ in rows) == 6
    assert sum(tag == "original" for tag, _, _ in rows) == 1


def test_factorial60_is_native_cp5_seed0_and_ood_test_blind():
    for _, overrides, run_id in cells():
        assert "seed=0" in overrides
        assert "stage=1" in overrides
        assert "train.epochs=60" in overrides
        assert "train.milestone_epochs=[10,30,60]" in overrides
        assert "run_tag=factorial60_20260801" in overrides
        assert "factorial60_20260801" in run_id


def test_factorial60_checkpoint_policy_keeps_final_models_and_dense_anchors():
    for tag, overrides, _ in cells():
        if tag.startswith("moe_"):
            assert "train.save_checkpoint_epochs=[60]" in overrides
        else:
            assert "train.save_checkpoint_epochs=[10,30,60]" in overrides


def test_five_shards_are_disjoint_and_exhaustive():
    rows = cells()
    shards = [sharded_rows(rows, index, 5) for index in range(5)]
    flattened = [run_id for shard in shards for _, _, run_id in shard]
    assert len(flattened) == len(rows)
    assert set(flattened) == {run_id for _, _, run_id in rows}
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
