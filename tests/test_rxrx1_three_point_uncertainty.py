import pytest

from scripts.analyze_rxrx1_three_point_uncertainty import (
    _midpoint_deviation,
    _slope,
    summarize_contrast,
)


def _result(value, env_values):
    return {
        "acc_heldout": value / 100.0,
        "per_env_heldout": {key: score / 100.0 for key, score in env_values.items()},
        "per_env_n_heldout": {key: 10 for key in env_values},
    }


def test_slope_and_midpoint_deviation_are_log_scale_aware():
    xs = [3.0, 4.0, 5.044394119]
    assert _slope(xs, [1.0, 3.0, 5.088788238]) == pytest.approx(2.0)
    assert _midpoint_deviation(xs, [1.0, 3.0, 5.088788238]) == pytest.approx(0.0)


def test_paired_batch_slope_recovers_overall_slope():
    rows = {}
    scale_gaps = {"quarter": 1.0, "midpoint": 2.0, "full": 3.044394119}
    for seed in (1, 2):
        for scale, gap in scale_gaps.items():
            rows[(scale, "left", seed)] = _result(10.0 + gap, {"a": 10.0 + gap, "b": 10.0 + gap})
            rows[(scale, "right", seed)] = _result(10.0, {"a": 10.0, "b": 10.0})
    report = summarize_contrast(rows, "left", "right", (1, 2), n_boot=1000)
    assert report["slope_seed_bootstrap"]["mean"] == pytest.approx(1.0)
    assert report["slope_hierarchical_seed_batch_bootstrap"]["mean"] == pytest.approx(1.0)
    assert report["midpoint_deviation_seed_bootstrap"]["mean"] == pytest.approx(0.0, abs=1e-9)
