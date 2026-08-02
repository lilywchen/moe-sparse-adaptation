from scripts.sweep_rxrx1_router_aux import sharded_rows
from scripts import sweep_rxrx1_tail_safe_confirm as sweep
from scripts.sweep_rxrx1_tail_safe_confirm import (
    BALANCE_W,
    SEEDS,
    ZLOSS_W,
    cells,
)


def test_tail_safe_confirmation_is_locked_and_paired():
    rows = cells()
    assert len(rows) == 4
    assert len({run_id for _, _, run_id in rows}) == 4
    for seed in SEEDS:
        seed_rows = [(tag, overrides) for tag, overrides, _ in rows if tag.endswith(f"s{seed}")]
        assert {tag.split("_", 1)[0] for tag, _ in seed_rows} == {"moe", "dense"}
        assert all(f"seed={seed}" in overrides for _, overrides in seed_rows)
    assert sweep.WANDB_JOB_TYPE == "rxrx1_tail_safe_confirm60"
    assert "confirmatory" in sweep.WANDB_TAGS


def test_tail_safe_sparse_configuration_has_no_tunable_fields():
    for tag, overrides, run_id in cells():
        assert "stage=1" in overrides
        assert "train.epochs=60" in overrides
        assert "train.milestone_epochs=[10,30,60]" in overrides
        assert "train.save_checkpoint_epochs=[10,30,60]" in overrides
        assert "train.objective=erm" in overrides
        assert "analysis.run_mechanism=true" in overrides
        if tag.startswith("moe_"):
            assert "model.variant=moe" in overrides
            assert "model.placement=early" in overrides
            assert "model.routing_unit=token" in overrides
            assert "model.geometry=cosine" in overrides
            assert "model.pressure=canonical" in overrides
            assert "model.n_experts=8" in overrides
            assert "model.top_k=1" in overrides
            assert f"losses.balance_w={BALANCE_W}" in overrides
            assert f"losses.zloss_w={ZLOSS_W}" in overrides
        else:
            assert "model.variant=dense_wide" in overrides
            assert "model.placement=early" in overrides
            assert "model.pressure=canonical" in overrides
        assert "tail_safe_confirm60" in run_id


def test_tail_safe_confirmation_shards_are_disjoint_and_exhaustive():
    rows = cells()
    shards = [sharded_rows(rows, index, 4) for index in range(4)]
    flattened = [run_id for shard in shards for _, _, run_id in shard]
    assert len(flattened) == len(rows)
    assert set(flattened) == {run_id for _, _, run_id in rows}
