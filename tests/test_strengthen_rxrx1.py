from scripts.strengthen_rxrx1 import SCREEN_SPECS, screen_rows, sharded_rows


def test_hypothesis_matrix_has_ten_unique_seed0_runs():
    rows = screen_rows()
    assert len(rows) == len(SCREEN_SPECS) == 10
    assert len({rid for _, _, rid in rows}) == 10
    for _, overrides, _ in rows:
        assert "seed=0" in overrides
        assert "train.epochs=90" in overrides
        assert "train.milestone_epochs=[10,30,60,90]" in overrides
        assert "train.warmup_epochs=5" in overrides
        assert "analysis.run_mechanism=false" in overrides

    interventions = {tag: set(overrides) for tag, overrides, _ in rows}
    assert "model.variant=dense_wide" in interventions["dense_wide"]
    assert "model.routing_unit=image" in interventions["moe_image_top1"]
    assert "model.pressure=route" in interventions["moe_token_within_env"]
    assert "model.top_k=2" in interventions["moe_token_top2"]
    assert "model.freeze_backbone=true" in interventions["frozen_linear"]
    assert "model.unfreeze_last_n_blocks=4" in interventions["partial_last4"]
    assert "model.pressure=output" in interventions["output_invariant"]
    assert "train.objective=environment_balanced" in interventions["environment_balanced"]
    assert "train.save_checkpoint_epochs=[10,30,60,90]" in interventions["original_anchor"]


def test_five_shards_cover_screen_once_with_two_arms_each():
    rows = screen_rows()
    shards = [sharded_rows(rows, i, 5) for i in range(5)]
    assert all(len(shard) == 2 for shard in shards)
    assert {rid for shard in shards for _, _, rid in shard} == {rid for _, _, rid in rows}
    assert sum(len(shard) for shard in shards) == len(rows)
