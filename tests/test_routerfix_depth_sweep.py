from scripts.sweep_rxrx1_routerfix_depth30 import (
    confirmation_cells, screen_cells, sharded_rows,
)


def _override_value(overrides, key):
    prefix = key + "="
    return next(value[len(prefix):] for value in overrides if value.startswith(prefix))


def test_screen_fills_eight_gpus_with_unique_corrected_runs():
    rows = screen_cells()
    assert len(rows) == 8
    assert len({run_id for _, _, run_id in rows}) == 8
    for _, overrides, run_id in rows:
        assert "model.routing_estimator=selected_st" in overrides
        assert "train.epochs=30" in overrides
        assert "train.milestone_epochs=[5,10,20,30]" in overrides
        assert "stage=1" in overrides
        if "_moe_" in run_id or "_moe_frozen_" in run_id:
            assert "selected_st" in run_id


def test_screen_has_locked_depth_and_control_matrix():
    rows = {label: overrides for label, overrides, _ in screen_cells()}
    assert set(rows) == {
        "original", "dense_last2", "learned_last1", "frozen_last1",
        "learned_last2", "frozen_last2", "learned_last4", "learned_all12",
    }
    assert _override_value(rows["dense_last2"], "model.ffn_block_indices") == "[10,11]"
    assert _override_value(rows["learned_last1"], "model.ffn_block_indices") == "[11]"
    assert _override_value(rows["learned_last2"], "model.ffn_block_indices") == "[10,11]"
    assert _override_value(rows["learned_last4"], "model.ffn_block_indices") == "[8,9,10,11]"
    assert _override_value(rows["learned_all12"], "model.ffn_block_indices") == (
        "[0,1,2,3,4,5,6,7,8,9,10,11]")


def test_four_container_shards_are_disjoint_and_two_jobs_each():
    rows = screen_cells()
    shards = [sharded_rows(rows, index, 4) for index in range(4)]
    assert [len(shard) for shard in shards] == [2, 2, 2, 2]
    assert {run_id for shard in shards for _, _, run_id in shard} == {
        run_id for _, _, run_id in rows}


def test_confirmation_is_two_fresh_seeds_by_four_exact_controls():
    rows = confirmation_cells((10, 11))
    assert len(rows) == 8
    assert len({run_id for _, _, run_id in rows}) == 8
    assert {int(_override_value(overrides, "seed")) for _, overrides, _ in rows} == {1, 2}
    for label, overrides, _ in rows:
        if not label.startswith("original"):
            assert _override_value(overrides, "model.ffn_block_indices") == "[10,11]"
