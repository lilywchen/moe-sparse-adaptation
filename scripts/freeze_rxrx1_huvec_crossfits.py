#!/usr/bin/env python
"""Freeze six target-complete HUVEC cross-fits and source-scaling variants.

Every HUVEC experiment is a target exactly once.  For each fold, four other
HUVEC experiments are reserved for source-only model selection, the remaining
16 HUVEC experiments are training sources, and the 17 official-training
non-HUVEC experiments are a fixed auxiliary source pool.  The target is never
used for normalization, fitting, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_KEY = "rxrx1-huvec-sixfold-all-sites-v1"
SCALES = (4, 8, 12, 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _rank(values, key):
    return sorted(values, key=lambda value: hashlib.sha256(
        f"{SPLIT_KEY}|{key}|{value}".encode()).hexdigest())


def _coverage(frame: pd.DataFrame) -> dict:
    output = {}
    for role, rows in frame.groupby("role", sort=True):
        output[str(role)] = {
            "n_sites": int(len(rows)),
            "n_wells": int(rows.well_id.nunique()),
            "n_experiments": int(rows.experiment_name.nunique()),
            "n_labels": int(rows.label.nunique()),
            "cell_types": sorted(map(str, rows.cell_type.unique())),
            "well_types": {str(k): int(v) for k, v in
                           rows.well_type.value_counts().sort_index().items()},
        }
    return output


def freeze(all_sites: str | Path, output_root: str | Path) -> dict:
    source_path = Path(all_sites).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source_path)
    required = {"global_index", "well_id", "cell_type", "dataset", "experiment",
                "experiment_name", "label", "site", "relative_path", "well_type"}
    if not required <= set(frame):
        raise ValueError(f"all-sites manifest lacks {sorted(required - set(frame))}")
    huvec_names = sorted(frame.loc[frame.cell_type.eq("HUVEC"),
                                   "experiment_name"].unique())
    if len(huvec_names) != 24:
        raise ValueError(f"expected 24 HUVEC experiments, found {len(huvec_names)}")
    aux_names = sorted(frame.loc[
        frame.dataset.eq("train") & ~frame.cell_type.eq("HUVEC"),
        "experiment_name"].unique())
    if len(aux_names) != 17:
        raise ValueError(f"expected 17 official-train auxiliary experiments, found {len(aux_names)}")

    ordered = _rank(huvec_names, "target-groups")
    groups = [list(map(str, chunk)) for chunk in np.array_split(
        np.asarray(ordered, dtype=object), 6)]
    registry = {
        "schema_version": 1,
        "split_key": SPLIT_KEY,
        "all_sites": str(source_path),
        "all_sites_sha256": _sha256(source_path),
        "target_policy": "target pixels are final-evaluation only",
        "fixed_auxiliary_experiment_names": aux_names,
        "huvec_target_groups": groups,
        "folds": [],
    }
    allowed = set(huvec_names) | set(aux_names)
    base = frame[frame.experiment_name.isin(allowed)].copy()
    for fold in range(6):
        target_names = set(groups[fold])
        validation_names = set(groups[(fold + 1) % 6])
        source_names = set(huvec_names) - target_names - validation_names
        ranked_source = _rank(source_names, f"fold-{fold}-source-prefix")
        if len(source_names) != 16:
            raise RuntimeError("cross-fit source inventory changed")
        fold_record = {
            "split_id": f"huvec_crossfit{fold}",
            "target_experiment_names": sorted(target_names),
            "selection_validation_experiment_names": sorted(validation_names),
            "huvec_training_experiment_names": sorted(source_names),
            "scales": [],
        }
        for count in SCALES:
            selected_huvec = set(ranked_source[:count])
            selected = base[
                base.experiment_name.isin(
                    target_names | validation_names | selected_huvec | set(aux_names))
            ].copy()
            selected["role"] = "train"
            selected.loc[selected.experiment_name.isin(validation_names), "role"] = (
                "selection_validation")
            selected.loc[selected.experiment_name.isin(target_names), "role"] = "target"
            if selected.groupby("well_id").role.nunique().max() != 1:
                raise RuntimeError("the two sites from a well crossed roles")
            if set(selected.loc[selected.role.eq("target"), "experiment_name"]) != target_names:
                raise RuntimeError("target experiment assignment changed")
            scale_id = f"huvec_crossfit{fold}_h{count}"
            path = root / f"{scale_id}.parquet"
            selected.sort_values(
                ["role", "cell_type", "experiment", "label", "well_id", "site"]
            ).to_parquet(path, index=False)
            fold_record["scales"].append({
                "scale_id": scale_id,
                "n_huvec_training_experiments": count,
                "huvec_training_experiment_names": sorted(selected_huvec),
                "assignment": str(path),
                "assignment_sha256": _sha256(path),
                "coverage": _coverage(selected),
            })
        registry["folds"].append(fold_record)
    _atomic_json(root / "huvec_crossfit_registry.json", registry)
    print(json.dumps(registry, indent=2, sort_keys=True), flush=True)
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-sites", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    freeze(args.all_sites, args.output_root)


if __name__ == "__main__":
    main()
