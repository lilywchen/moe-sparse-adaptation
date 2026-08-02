from scripts.sweep_rxrx1_router_aux import AUX_SETTINGS, PRESSURES, cells, sharded_rows


def test_router_aux60_has_complete_unique_budget():
    rows = cells()
    assert len(rows) == len(PRESSURES) * len(AUX_SETTINGS) == 16
    assert len({run_id for _, _, run_id in rows}) == 16
    assert sum(tag.startswith("route_") for tag, _, _ in rows) == 8
    assert sum(tag.startswith("canonical_") for tag, _, _ in rows) == 8


def test_router_aux60_changes_only_predeclared_auxiliary_mechanism_fields():
    for tag, overrides, run_id in cells():
        pressure = tag.split("_", 1)[0]
        assert "seed=0" in overrides
        assert "stage=1" in overrides
        assert "model.variant=moe" in overrides
        assert "model.placement=early" in overrides
        assert "model.routing_unit=token" in overrides
        assert "model.geometry=cosine" in overrides
        assert f"model.pressure={pressure}" in overrides
        assert "model.n_experts=8" in overrides
        assert "model.top_k=1" in overrides
        assert "train.epochs=60" in overrides
        assert "train.milestone_epochs=[10,30,60]" in overrides
        assert "train.save_checkpoint_epochs=[10,30,60]" in overrides
        assert "analysis.run_mechanism=true" in overrides
        assert any(v.startswith("losses.balance_w=") for v in overrides)
        assert any(v.startswith("losses.zloss_w=") for v in overrides)
        assert "router_aux60" in run_id


def test_router_aux60_does_not_duplicate_frozen_factorial_default():
    assert ("bw1em2_z1em3", 1.0e-2, 1.0e-3) not in AUX_SETTINGS


def test_router_aux60_shards_are_disjoint_and_exhaustive():
    rows = cells()
    shards = [sharded_rows(rows, index, 5) for index in range(5)]
    flattened = [run_id for shard in shards for _, _, run_id in shard]
    assert len(flattened) == len(rows)
    assert set(flattened) == {run_id for _, _, run_id in rows}
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
