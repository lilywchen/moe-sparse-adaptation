import numpy as np
import pandas as pd

from moe_shift.data.rxrx1_huvec_batch import (
    choose_difficulty_anchors,
    matched_pseudo_target_wells,
    source_compositions,
)
from scripts.prepare_rxrx1_huvec_batch_effect import planned_runs
from scripts.run_rxrx1_huvec_batch_effect import schedule


def synthetic_geometry(n_experiments=6, n_labels=5):
    rows, features = [], []
    for experiment in range(n_experiments):
        for label in range(n_labels):
            rows.append((experiment, label))
            features.append([1 + label, experiment / 10, (label + experiment) / 50])
    experiments, labels = np.asarray(rows).T
    return experiments, labels, np.asarray(features, np.float32)


def test_difficulty_anchors_are_unique_and_reproducible():
    experiments, labels, features = synthetic_geometry()
    first, audit = choose_difficulty_anchors(
        range(6), experiments, labels, features, count=4)
    second, _ = choose_difficulty_anchors(range(6), experiments, labels, features, count=4)
    assert first == second
    assert len(first) == len(set(first)) == 4
    assert len(audit) == 6


def test_source_compositions_have_equal_size_and_expected_extremes():
    order = list(range(6))
    distance = np.abs(np.subtract.outer(order, order)).astype(float)
    rows = source_compositions(0, range(6), order, distance, size=3)
    by_name = {row["composition"]: row for row in rows}
    assert by_name["near"]["source_experiments"] == [1, 2, 3]
    assert by_name["far"]["source_experiments"] == [3, 4, 5]
    assert all(len(row["source_experiments"]) == 3 for row in rows)


def test_pseudo_targets_match_label_and_site_count():
    target = pd.DataFrame([
        {"well_id": "t0", "experiment": 9, "label": 0, "n_sites": 2,
         "correct_top1": False},
        {"well_id": "t1", "experiment": 9, "label": 1, "n_sites": 2,
         "correct_top1": True},
    ])
    iid = pd.DataFrame([
        {"well_id": f"s{experiment}{label}", "experiment": experiment,
         "label": label, "n_sites": 2, "correct_top1": bool(label)}
        for experiment in range(3) for label in range(2)
    ])
    rows = matched_pseudo_target_wells(iid, target, 9, n_resamples=7)
    assert len(rows) == 14
    assert (rows.target_n_sites == rows.pseudo_n_sites).all()
    assert rows.groupby("resample").label.nunique().eq(2).all()


def test_frozen_wave_is_36_unique_runs_with_all_controls():
    anchors = [1, 4, 7, 10]
    splits = [
        {"split_id": f"loo_t{target}", "kind": "diagnostic_loo"}
        for target in range(16)
    ] + [
        {"split_id": f"composition_t{target}_{composition}",
         "kind": "source_composition"}
        for target in anchors for composition in ("near", "diverse", "far")
    ]
    rows = planned_runs(splits, anchors)
    assert len(rows) == len({row["run_id"] for row in rows}) == 36
    assert sum(row["model"] == "vit_tiny" for row in rows) == 28
    assert sum(row["model"] == "vit_tiny_moe" for row in rows) == 4
    assert sum(row["model"] == "vit_tiny_dense_matched" for row in rows) == 4
    assert all(row["selection"]["metric"] == "source_iid_site_top1" for row in rows)


class FakeOptimizer:
    def __init__(self):
        self.param_groups = [{"initial_lr": 1.0, "lr": 1.0}]


def test_learning_rate_resume_is_epoch_exact_not_run_length_dependent():
    recipe = {"warmup_epochs": 5, "schedule_epochs": 160, "min_lr_ratio": .02}
    whole, resumed = FakeOptimizer(), FakeOptimizer()
    for epoch in range(80):
        schedule(whole, epoch, recipe)
    schedule(resumed, 79, recipe)
    assert whole.param_groups[0]["lr"] == resumed.param_groups[0]["lr"]
    assert 0.02 < whole.param_groups[0]["lr"] < 1.0
