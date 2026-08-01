from scripts.strengthen_rxrx1 import SCREEN_SPECS, screen_rows, sharded_rows


def test_strength_screen_has_ten_unique_seed0_runs():
    rows = screen_rows()
    assert len(rows) == len(SCREEN_SPECS) == 10
    assert len({rid for _, _, rid in rows}) == 10
    for _, overrides, _ in rows:
        assert "seed=0" in overrides
        assert "train.epochs=30" in overrides
        assert "train.warmup_epochs=3" in overrides
        assert "analysis.run_mechanism=false" in overrides


def test_five_shards_cover_screen_once_with_two_arms_each():
    rows = screen_rows()
    shards = [sharded_rows(rows, i, 5) for i in range(5)]
    assert all(len(shard) == 2 for shard in shards)
    assert {rid for shard in shards for _, _, rid in shard} == {rid for _, _, rid in rows}
    assert sum(len(shard) for shard in shards) == len(rows)
