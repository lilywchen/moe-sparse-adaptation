"""Predeclared, leakage-safe RxRx1 training-environment scaling design.

The axis in this module is the number of independent *training experiments*.  It is deliberately
not called generic sample scaling: increasing the prefix changes both the number of fields and the
number of acquisition environments.  Site-density and class-count curves must be separate studies.
"""
from __future__ import annotations

import hashlib
from collections import Counter


DESIGN_SEED = 20260810
TRAIN_EXPERIMENTS_BY_CELL = {
    0: (0, 1, 2, 3, 4, 5, 6),
    1: (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26),
    2: (35, 36, 37, 38, 39, 40, 41),
    3: (46, 47, 48),
}

# Hamilton-style proportional allocation with at least one experiment from each cell type.  The
# quarter-data point therefore holds cell-type coverage fixed while reducing independent batches.
QUARTER_QUOTAS = {0: 2, 1: 4, 2: 1, 3: 1}


def _stable_order(cell_type: int, experiments, seed: int = DESIGN_SEED):
    return tuple(sorted(
        (int(value) for value in experiments),
        key=lambda value: hashlib.sha256(
            f"{int(seed)}:{int(cell_type)}:{int(value)}".encode()).hexdigest(),
    ))


def quarter_environment_subset(seed: int = DESIGN_SEED):
    """Return the fixed eight-experiment nested prefix, independent of labels and outcomes."""
    selected = []
    for cell_type in sorted(TRAIN_EXPERIMENTS_BY_CELL):
        ordered = _stable_order(cell_type, TRAIN_EXPERIMENTS_BY_CELL[cell_type], seed)
        selected.extend(ordered[:QUARTER_QUOTAS[cell_type]])
    return tuple(selected)


def full_environment_subset():
    return tuple(
        value
        for cell_type in sorted(TRAIN_EXPERIMENTS_BY_CELL)
        for value in TRAIN_EXPERIMENTS_BY_CELL[cell_type]
    )


def subset_digest(environments):
    canonical = ",".join(str(value) for value in sorted(int(v) for v in environments))
    return hashlib.sha256(canonical.encode()).hexdigest()


def audit_environment_subset(dataset, environments):
    """Audit an RxRx1 WILDS train subset without loading pixels.

    Counts are made at experimental units.  A well is keyed by experiment, plate, and well so
    encoded well ids cannot collide across plates.  This function intentionally accepts the WILDS
    dataset object to keep dataset acquisition out of sweep planning and unit tests.
    """
    import torch

    subset = dataset.get_subset("train")
    indices = torch.as_tensor(subset.indices, dtype=torch.long)
    metadata = dataset.metadata_array[indices]
    labels = dataset.y_array[indices]
    fields = list(dataset.metadata_fields)
    exp_col = fields.index("experiment")
    cell_col = fields.index("cell_type")
    plate_col = fields.index("plate")
    well_col = fields.index("well")
    site_col = fields.index("site")

    wanted = tuple(int(value) for value in environments)
    mask = torch.zeros(len(indices), dtype=torch.bool)
    for environment in wanted:
        mask |= metadata[:, exp_col] == environment
    observed = set(int(value) for value in torch.unique(metadata[mask, exp_col]).tolist())
    missing = set(wanted) - observed
    if missing:
        raise ValueError(f"unknown/missing RxRx1 train environments: {sorted(missing)}")

    chosen_metadata = metadata[mask]
    chosen_labels = labels[mask].long()
    n_classes = int(dataset.n_classes)
    per_class = torch.bincount(chosen_labels, minlength=n_classes)
    environment_counts = Counter(int(value) for value in chosen_metadata[:, exp_col].tolist())
    cell_environment_counts = Counter()
    for environment in wanted:
        rows = chosen_metadata[:, exp_col] == environment
        cell = int(torch.unique(chosen_metadata[rows, cell_col]).item())
        cell_environment_counts[cell] += 1
    cell_sample_counts = Counter(int(value) for value in chosen_metadata[:, cell_col].tolist())
    wells = set(zip(
        chosen_metadata[:, exp_col].tolist(),
        chosen_metadata[:, plate_col].tolist(),
        chosen_metadata[:, well_col].tolist(),
    ))
    sites = set(zip(
        chosen_metadata[:, exp_col].tolist(),
        chosen_metadata[:, plate_col].tolist(),
        chosen_metadata[:, well_col].tolist(),
        chosen_metadata[:, site_col].tolist(),
    ))
    return {
        "environment_ids": list(wanted),
        "environment_sha256": subset_digest(wanted),
        "n_environments": len(wanted),
        "n_fields": int(mask.sum()),
        "n_wells": len(wells),
        "n_sites": len(sites),
        "n_classes_observed": int((per_class > 0).sum()),
        "n_classes_expected": n_classes,
        "examples_per_class_min": int(per_class.min()),
        "examples_per_class_median": float(per_class.float().median()),
        "examples_per_class_max": int(per_class.max()),
        "environment_field_counts": dict(sorted(environment_counts.items())),
        "cell_environment_counts": dict(sorted(cell_environment_counts.items())),
        "cell_field_counts": dict(sorted(cell_sample_counts.items())),
    }
