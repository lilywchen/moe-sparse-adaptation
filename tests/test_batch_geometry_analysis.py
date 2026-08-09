import numpy as np

from scripts.analyze_rxrx1_batch_geometry import _corr, _ece, _error_overlap, _linear_cka


def test_correlations_handle_rank_and_degenerate_inputs():
    assert _corr([1, 2, 3], [2, 4, 6]) > 0.999
    assert _corr([1, 2, 3], [30, 20, 10], rank=True) < -0.999
    assert _corr([1, 1, 1], [1, 2, 3]) is None


def test_linear_cka_is_one_for_identical_centered_geometry():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(20, 6))
    assert _linear_cka(x, x) > 0.999999


def test_ece_is_zero_for_two_perfectly_calibrated_bins():
    confidence = np.asarray([0.25] * 4 + [0.75] * 4)
    correct = np.asarray([1, 0, 0, 0, 1, 1, 1, 0], dtype=bool)
    assert abs(_ece(confidence, correct, bins=4)) < 1e-12


def test_error_overlap_reports_directional_rescues():
    base = {
        "labels": np.asarray([0, 1, 2, 3]), "envs": np.asarray([0, 0, 1, 1]),
        "cells": np.zeros(4), "confidence": np.ones(4),
    }
    predictions = {
        "dense": {split: {**base, "prediction": np.asarray([0, 9, 2, 9]),
                           "correct": np.asarray([1, 0, 1, 0], dtype=bool)}
                  for split in ("val", "test")},
        "shared": {split: {**base, "prediction": np.asarray([0, 1, 9, 9]),
                            "correct": np.asarray([1, 1, 0, 0], dtype=bool)}
                   for split in ("val", "test")},
    }
    row = _error_overlap(predictions)["val"]["dense__vs__shared"]
    assert row["shared_rescues_dense"] == 0.25
    assert row["dense_rescues_shared"] == 0.25
