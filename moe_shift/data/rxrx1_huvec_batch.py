"""Frozen design helpers for the RxRx1 HUVEC batch-degradation study."""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
import pandas as pd

from .rxrx1_huvec import EXPECTED_TREATMENTS


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def target_difficulty(target, sources, experiments, labels, features):
    """Median same-perturbation target-to-source-centroid cosine displacement."""
    experiments = np.asarray(experiments, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    features = unit(features)
    rows = []
    for index in np.flatnonzero(experiments == int(target)):
        source = np.isin(experiments, list(map(int, sources))) & (labels == labels[index])
        if source.any():
            centroid = unit(features[source].mean(0, keepdims=True))[0]
            rows.append(1.0 - float(features[index] @ centroid))
    if not rows:
        raise ValueError(f"target {target} has no perturbations shared with its source set")
    return float(np.median(rows)), len(rows)


def choose_difficulty_anchors(source_experiments, experiments, labels, features, count=4):
    """Choose deterministic quantile anchors using only frozen Cell-DINO geometry."""
    source = sorted(map(int, source_experiments))
    if not 1 <= int(count) <= len(source):
        raise ValueError("anchor count must lie between one and the source experiment count")
    rows = []
    for target in source:
        difficulty, coverage = target_difficulty(
            target, [value for value in source if value != target],
            experiments, labels, features)
        rows.append((difficulty, target, coverage))
    rows.sort()
    indices = np.linspace(0, len(rows) - 1, int(count)).round().astype(int)
    anchors = [int(rows[index][1]) for index in indices]
    if len(set(anchors)) != int(count):
        raise RuntimeError("difficulty quantiles produced duplicate anchors")
    return anchors, {
        str(target): {"difficulty": float(difficulty), "matched_labels": int(coverage)}
        for difficulty, target, coverage in rows
    }


def source_compositions(target, source_experiments, experiment_order, distance,
                        size=8, candidate_limit=20000):
    """Return equal-size near, diverse, and far source-batch counterfactuals.

    Near/far use the frozen matched-perturbation distance to the target.  Diverse maximizes the
    median pairwise distance among candidate source sets and is deterministic.
    """
    source = sorted(set(map(int, source_experiments)) - {int(target)})
    size = int(size)
    if not 1 <= size <= len(source):
        raise ValueError("source composition size is incompatible with available experiments")
    order = list(map(int, experiment_order))
    lookup = {value: index for index, value in enumerate(order)}
    if int(target) not in lookup or not set(source) <= set(lookup):
        raise ValueError("composition experiments are absent from the distance matrix")
    target_index = lookup[int(target)]
    ranked = sorted(source, key=lambda value: (distance[target_index, lookup[value]], value))
    near, far = tuple(sorted(ranked[:size])), tuple(sorted(ranked[-size:]))

    all_candidates = combinations(source, size)
    best = None
    for index, candidate in enumerate(all_candidates):
        if index >= int(candidate_limit):
            raise RuntimeError(
                "candidate limit truncated the deterministic diverse-source search")
        candidate = tuple(sorted(candidate))
        if candidate in {near, far}:
            continue
        pairwise = [distance[lookup[left], lookup[right]]
                    for left, right in combinations(candidate, 2)]
        score = float(np.median(pairwise)) if pairwise else 0.0
        target_distance = float(np.median(
            [distance[target_index, lookup[value]] for value in candidate]))
        key = (score, target_distance, tuple(-value for value in candidate))
        if best is None or key > best[0]:
            best = (key, tuple(candidate), score, target_distance)
    assert best is not None
    diverse = tuple(sorted(best[1]))

    def record(name, values):
        values = list(map(int, values))
        return {
            "composition": name,
            "source_experiments": values,
            "target_distance": float(np.median(
                [distance[target_index, lookup[value]] for value in values])),
            "source_pairwise_diversity": float(np.median([
                distance[lookup[left], lookup[right]]
                for left, right in combinations(values, 2)
            ])) if len(values) > 1 else 0.0,
        }

    return [record("near", near), record("diverse", diverse), record("far", far)]


def role_label_coverage(assignment):
    expected = set(range(EXPECTED_TREATMENTS))
    output = {}
    for role in ("train", "iid_validation", "target"):
        observed = set(map(int, assignment.loc[assignment.role == role, "label"].unique()))
        output[role] = {
            "observed_labels": len(observed),
            "fraction": float(len(observed) / EXPECTED_TREATMENTS),
            "missing_labels": sorted(expected - observed),
        }
    return output


def matched_pseudo_target_wells(iid_predictions, target_predictions, target_experiment,
                                n_resamples=50, seed=20260815):
    """Choose class- and site-count-matched IID wells for a held-out experiment.

    The returned table contains one pseudo-target well per target label whenever a compatible IID
    well exists.  Selection is hash-ranked rather than RNG-order-dependent.
    """
    iid = pd.DataFrame(iid_predictions).copy()
    target = pd.DataFrame(target_predictions).copy()
    required = {"well_id", "experiment", "label", "n_sites", "correct_top1"}
    if not required <= set(iid) or not required <= set(target):
        raise ValueError(f"pseudo-target predictions require columns {sorted(required)}")
    target = target[target.experiment.astype(int) == int(target_experiment)].copy()
    if target.empty:
        raise ValueError(f"target experiment {target_experiment} has no well predictions")
    if target.duplicated("label").any():
        raise ValueError("expected at most one treatment well per target experiment and label")
    rows = []
    for resample in range(int(n_resamples)):
        for target_row in target.sort_values("label").itertuples(index=False):
            candidates = iid[(iid.label == int(target_row.label)) &
                             (iid.n_sites == int(target_row.n_sites))]
            if candidates.empty:
                candidates = iid[iid.label == int(target_row.label)]
            if candidates.empty:
                continue
            ranked = []
            for candidate in candidates.itertuples(index=False):
                token = (f"{int(seed)}|{int(target_experiment)}|{resample}|"
                         f"{int(target_row.label)}|{candidate.well_id}")
                ranked.append((hashlib.sha256(token.encode()).hexdigest(), candidate))
            candidate = min(ranked, key=lambda item: item[0])[1]
            rows.append({
                "target_experiment": int(target_experiment),
                "resample": int(resample), "label": int(target_row.label),
                "target_well_id": str(target_row.well_id),
                "pseudo_well_id": str(candidate.well_id),
                "target_correct_top1": bool(target_row.correct_top1),
                "pseudo_correct_top1": bool(candidate.correct_top1),
                "target_n_sites": int(target_row.n_sites),
                "pseudo_n_sites": int(candidate.n_sites),
            })
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("no label-matched pseudo-target wells could be constructed")
    return output
