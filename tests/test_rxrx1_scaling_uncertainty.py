import pytest

from scripts.analyze_rxrx1_scaling_uncertainty import (
    hierarchical_batch_bootstrap,
    seed_bootstrap,
)


def test_seed_bootstrap_is_deterministic_and_centers_on_paired_mean():
    values = {1: -0.5, 2: -0.7, 5: 0.2}
    first = seed_bootstrap(values, n_boot=2000, rng_seed=7)
    second = seed_bootstrap(values, n_boot=2000, rng_seed=7)
    assert first == second
    assert first["mean"] == pytest.approx(-1.0 / 3.0)
    assert first["lo"] < first["mean"] < first["hi"]


def test_hierarchical_bootstrap_preserves_seed_and_batch_pairing():
    values = {
        1: {"a": 1.0, "b": 3.0, "c": 5.0},
        2: {"a": 2.0, "b": 4.0, "c": 6.0},
    }
    counts = {seed: {"a": 1, "b": 2, "c": 1} for seed in values}
    result = hierarchical_batch_bootstrap(values, counts, n_boot=3000, rng_seed=11)
    assert result["mean"] == pytest.approx(3.5)
    assert result["n_seeds"] == 2
    assert result["n_batches"] == 3
    assert result["lo"] < result["mean"] < result["hi"]


def test_hierarchical_bootstrap_rejects_misaligned_batches():
    values = {1: {"a": 1.0, "b": 2.0}, 2: {"a": 2.0, "c": 3.0}}
    counts = {1: {"a": 1, "b": 1}, 2: {"a": 1, "c": 1}}
    with pytest.raises(ValueError, match="batch keys differ"):
        hierarchical_batch_bootstrap(values, counts, n_boot=10)
