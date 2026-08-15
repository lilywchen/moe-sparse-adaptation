#!/usr/bin/env python
"""Freeze three custom experiment-held-out folds for full multi-cell RxRx1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_LABELS = 1_108
CELL_TYPES = ("HEPG2", "HUVEC", "RPE", "U2OS")
SPLIT_KEY = "rxrx1-full-threefold-v1"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _ranked_experiments(frame, cell_type):
    names = sorted(frame.loc[frame.cell_type == cell_type, "experiment_name"].unique())
    return sorted(names, key=lambda name: hashlib.sha256(
        f"{SPLIT_KEY}|{cell_type}|{name}".encode()).hexdigest())


def _coverage(frame):
    output = {}
    for role in ("train", "iid_validation", "target"):
        rows = frame[frame.role == role]
        observed = set(map(int, rows.label.unique()))
        output[role] = {
            "n_sites": len(rows), "n_wells": int(rows.well_id.nunique()),
            "n_labels": len(observed),
            "label_fraction": len(observed) / EXPECTED_LABELS,
            "by_cell": {
                cell: {
                    "n_sites": len(cell_rows),
                    "n_wells": int(cell_rows.well_id.nunique()),
                    "n_labels": int(cell_rows.label.nunique()),
                }
                for cell, cell_rows in rows.groupby("cell_type", sort=True)
            },
        }
    return output


def freeze(treatment_manifest, output_root):
    treatment_manifest = Path(treatment_manifest).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(treatment_manifest)
    required = {
        "global_index", "well_id", "cell_type", "experiment", "experiment_name",
        "label", "site", "relative_path",
    }
    if not required <= set(frame):
        raise ValueError(f"treatment manifest lacks columns: {sorted(required - set(frame))}")
    if set(frame.cell_type.unique()) != set(CELL_TYPES):
        raise ValueError("full split requires all four RxRx1 cell types")
    if set(map(int, frame.label.unique())) != set(range(EXPECTED_LABELS)):
        raise ValueError("full split requires treatment labels 0..1107")
    if frame.groupby("well_id").cell_type.nunique().max() != 1:
        raise ValueError("well identity crosses cell types")

    target_chunks = {}
    for cell in CELL_TYPES:
        target_chunks[cell] = [list(chunk) for chunk in np.array_split(
            np.asarray(_ranked_experiments(frame, cell), dtype=object), 3)]
        flattened = [name for chunk in target_chunks[cell] for name in chunk]
        if sorted(flattened) != sorted(frame.loc[
                frame.cell_type == cell, "experiment_name"].unique()):
            raise RuntimeError(f"threefold experiment partition failed for {cell}")

    registry = {
        "schema_version": 1, "split_key": SPLIT_KEY,
        "treatment_manifest": str(treatment_manifest),
        "treatment_manifest_sha256": _sha256(treatment_manifest),
        "folds": [],
    }
    for fold_index in range(3):
        split_id = f"full_fold{fold_index}"
        target_names = {
            name for cell in CELL_TYPES for name in target_chunks[cell][fold_index]
        }
        assignment = frame.copy()
        assignment["role"] = np.where(
            assignment.experiment_name.isin(target_names), "target", "train")
        source_wells = assignment[assignment.role == "train"].drop_duplicates("well_id")
        iid_wells = set()
        for (cell, label), rows in source_wells.groupby(["cell_type", "label"], sort=True):
            candidates = sorted(
                (hashlib.sha256(
                    f"{SPLIT_KEY}|{fold_index}|{cell}|{int(label)}|{row.well_id}".encode()
                 ).hexdigest(), str(row.well_id))
                for row in rows.itertuples(index=False)
            )
            if len(candidates) < 2:
                raise ValueError(
                    f"{split_id}: {cell} label {label} lacks train/IID source occurrences")
            iid_wells.add(candidates[0][1])
        assignment.loc[assignment.well_id.isin(iid_wells), "role"] = "iid_validation"
        if assignment.groupby("well_id").role.nunique().max() != 1:
            raise RuntimeError(f"{split_id}: sites from one well cross roles")
        if assignment.loc[assignment.role == "train", "label"].nunique() != EXPECTED_LABELS:
            raise ValueError(f"{split_id}: source training loses perturbation classes")
        if set(assignment.loc[assignment.role == "target", "experiment_name"]) != target_names:
            raise RuntimeError(f"{split_id}: target experiment assignment changed")

        assignment_path = output_root / f"{split_id}.parquet"
        assignment.sort_values(
            ["role", "cell_type", "experiment", "label", "well_id", "site"]
        ).to_parquet(assignment_path, index=False)
        source_names = sorted(set(frame.experiment_name) - target_names)
        registry["folds"].append({
            "split_id": split_id,
            "source_experiment_names": source_names,
            "target_experiment_names": sorted(target_names),
            "source_experiments": sorted(map(int, frame.loc[
                frame.experiment_name.isin(source_names), "experiment"].unique())),
            "target_experiments": sorted(map(int, frame.loc[
                frame.experiment_name.isin(target_names), "experiment"].unique())),
            "target_experiments_by_cell": {
                cell: sorted(target_chunks[cell][fold_index]) for cell in CELL_TYPES
            },
            "assignment": str(assignment_path),
            "assignment_sha256": _sha256(assignment_path),
            "coverage": _coverage(assignment),
        })
    _atomic_json(output_root / "split_registry.json", registry)
    print(json.dumps(registry, indent=2, sort_keys=True), flush=True)
    return registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--treatment-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    freeze(args.treatment_manifest, args.output_root)


if __name__ == "__main__":
    main()
