#!/usr/bin/env python
"""Freeze official full-RxRx1 metadata into audited six-channel site manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


EXPECTED_CELL_EXPERIMENTS = {"HEPG2": 11, "HUVEC": 24, "RPE": 11, "U2OS": 5}
EXPECTED_SITES = 125_510
EXPECTED_WELLS = 62_755
EXPECTED_TREATMENT_SITES = 112_824
EXPECTED_TREATMENTS = 1_108


def _native_channel_paths(raw_root, composite_relative_path):
    """Expand one site path into the six official RxRx1 channel PNGs.

    Keep this small metadata utility independent of the PyTorch data stack so
    manifests can also be audited on CPU-only transfer and login nodes.
    """
    relative = Path(str(composite_relative_path))
    return tuple(
        Path(raw_root) / relative.parent / f"{relative.stem}_w{channel}.png"
        for channel in range(1, 7)
    )


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


def build(metadata_csv, raw_root, output_root, verify_paths=True):
    metadata_csv = Path(metadata_csv).expanduser().resolve()
    raw_root = Path(raw_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(metadata_csv)
    required = {
        "site_id", "well_id", "cell_type", "dataset", "experiment", "plate",
        "well", "site", "well_type", "sirna", "sirna_id",
    }
    if set(frame) != required:
        raise ValueError(
            f"official metadata schema changed: missing={sorted(required - set(frame))}, "
            f"unexpected={sorted(set(frame) - required)}")
    if len(frame) != EXPECTED_SITES or frame.site_id.nunique() != EXPECTED_SITES:
        raise ValueError("official RxRx1 site count or uniqueness changed")
    if frame.well_id.nunique() != EXPECTED_WELLS:
        raise ValueError("official RxRx1 well count changed")
    experiments = frame.groupby("cell_type").experiment.nunique().to_dict()
    if experiments != EXPECTED_CELL_EXPERIMENTS:
        raise ValueError(f"official cell/experiment inventory changed: {experiments}")
    treatment = frame.well_type.eq("treatment")
    if int(treatment.sum()) != EXPECTED_TREATMENT_SITES:
        raise ValueError("official treatment-site count changed")
    treatment_labels = sorted(map(int, frame.loc[treatment, "sirna_id"].unique()))
    if treatment_labels != list(range(EXPECTED_TREATMENTS)):
        raise ValueError("treatment labels must be the contiguous range 0..1107")
    if frame.groupby("well_id").site.nunique().max() > 2:
        raise ValueError("a well contains more than two microscope sites")

    frame.insert(0, "global_index", range(len(frame)))
    frame["experiment_name"] = frame.experiment.astype(str)
    frame["relative_path"] = (
        "images/" + frame.experiment_name + "/Plate"
        + frame.plate.astype(str) + "/" + frame.well.astype(str)
        + "_s" + frame.site.astype(str) + ".png"
    )
    experiment_order = sorted(frame.experiment_name.unique())
    experiment_map = {experiment: index for index, experiment in enumerate(experiment_order)}
    frame["experiment"] = frame.experiment_name.map(experiment_map).astype("int64")
    frame["label"] = frame.sirna_id.astype("int64")

    if verify_paths:
        missing = []
        for row in frame.itertuples(index=False):
            for path in _native_channel_paths(raw_root, row.relative_path):
                if not path.is_file():
                    missing.append(str(path))
                    if len(missing) >= 20:
                        break
            if len(missing) >= 20:
                break
        if missing:
            raise FileNotFoundError(f"full RxRx1 extraction is incomplete: {missing}")

    all_path = output_root / "all_sites.parquet"
    treatment_path = output_root / "treatment_sites.parquet"
    inventory_path = output_root / "experiment_inventory.csv"
    frame.to_parquet(all_path, index=False)
    frame.loc[treatment].to_parquet(treatment_path, index=False)
    inventory = (frame.groupby(["cell_type", "experiment_name", "experiment", "dataset"])
                 .agg(n_sites=("site_id", "size"), n_wells=("well_id", "nunique"),
                      n_treatment_sites=("well_type", lambda values: int((values == "treatment").sum())),
                      n_treatment_labels=("sirna_id", lambda values: int(
                          frame.loc[values.index].loc[lambda rows: rows.well_type == "treatment",
                                                      "sirna_id"].nunique())))
                 .reset_index())
    inventory.to_csv(inventory_path, index=False)
    summary = {
        "schema_version": 1,
        "source": "official Recursion RxRx1 metadata.csv",
        "metadata_csv": str(metadata_csv),
        "metadata_sha256": _sha256(metadata_csv),
        "raw_root": str(raw_root),
        "paths_verified": bool(verify_paths),
        "n_sites": len(frame), "n_wells": int(frame.well_id.nunique()),
        "n_treatment_sites": int(treatment.sum()),
        "n_treatment_labels": len(treatment_labels),
        "cell_experiment_counts": experiments,
        "all_sites": str(all_path), "all_sites_sha256": _sha256(all_path),
        "treatment_sites": str(treatment_path),
        "treatment_sites_sha256": _sha256(treatment_path),
        "experiment_inventory": str(inventory_path),
        "experiment_inventory_sha256": _sha256(inventory_path),
    }
    _atomic_json(output_root / "full_manifest.summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--skip-path-verification", action="store_true")
    args = parser.parse_args()
    build(args.metadata_csv, args.raw_root, args.output_root,
          verify_paths=not args.skip_path_verification)


if __name__ == "__main__":
    main()
