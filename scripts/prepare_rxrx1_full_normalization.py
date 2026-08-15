#!/usr/bin/env python
"""Cache exact channel moments and freeze launch-ready full-RxRx1 registries."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


CHANNELS = 6


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _atomic_parquet(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _channel_paths(raw_root, relative_path):
    relative = Path(str(relative_path))
    return tuple(
        Path(raw_root) / relative.parent / f"{relative.stem}_w{channel}.png"
        for channel in range(1, CHANNELS + 1)
    )


def _site_moments(task):
    global_index, relative_path, raw_root = task
    row = {"global_index": int(global_index)}
    expected_shape = None
    for channel, path in enumerate(_channel_paths(raw_root, relative_path)):
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            values = np.asarray(image, dtype=np.float64) / 255.0
        if values.ndim != 2:
            raise ValueError(f"expected a grayscale RxRx1 channel, got {values.shape}: {path}")
        if expected_shape is None:
            expected_shape = values.shape
        elif values.shape != expected_shape:
            raise ValueError(f"channel shapes differ within site {relative_path}")
        row[f"c{channel}_sum"] = float(values.sum(dtype=np.float64))
        row[f"c{channel}_sum_squares"] = float(np.square(values).sum(dtype=np.float64))
        row[f"c{channel}_pixels"] = int(values.size)
    return row


def extract_shard(manifest, raw_root, output_root, shard_index, num_shards, workers):
    manifest = Path(manifest).expanduser().resolve()
    raw_root = Path(raw_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    shard_index = int(shard_index); num_shards = int(num_shards)
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard index must lie in [0, num_shards)")
    frame = pd.read_parquet(manifest, columns=["global_index", "relative_path"])
    shard = frame[frame.global_index.astype("int64") % num_shards == shard_index]
    tasks = [
        (int(row.global_index), str(row.relative_path), str(raw_root))
        for row in shard.itertuples(index=False)
    ]
    if not tasks:
        raise ValueError(f"empty channel-moment shard {shard_index}/{num_shards}")
    rows = []
    with mp.get_context("spawn").Pool(processes=int(workers)) as pool:
        for count, row in enumerate(
                pool.imap_unordered(_site_moments, tasks, chunksize=8), start=1):
            rows.append(row)
            if count == 1 or count % 500 == 0 or count == len(tasks):
                print(
                    f"[moments] shard={shard_index}/{num_shards} "
                    f"sites={count}/{len(tasks)}", flush=True)
    output = output_root / f"moment_shard{shard_index:02d}-of-{num_shards:02d}.parquet"
    result = pd.DataFrame(rows).sort_values("global_index").reset_index(drop=True)
    if len(result) != len(tasks) or result.global_index.nunique() != len(tasks):
        raise RuntimeError("channel-moment shard lost or duplicated sites")
    _atomic_parquet(output, result)
    marker = {
        "schema_version": 1, "state": "complete", "shard_index": shard_index,
        "num_shards": num_shards, "n_sites": len(result), "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest), "raw_root": str(raw_root),
        "output": str(output), "output_sha256": _sha256(output),
    }
    _atomic_json(output.with_suffix(".summary.json"), marker)
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)


def _normalization(rows):
    means, stds, pixels = [], [], []
    for channel in range(CHANNELS):
        count = int(rows[f"c{channel}_pixels"].sum())
        total = float(rows[f"c{channel}_sum"].sum())
        squares = float(rows[f"c{channel}_sum_squares"].sum())
        mean = total / count
        variance = max(squares / count - mean * mean, 1e-12)
        means.append(mean); stds.append(variance ** 0.5); pixels.append(count)
    return {"mean": means, "std": stds, "pixels_per_channel": pixels}


def finalize(manifest, treatment_manifest, raw_root, split_registry, output_root, num_shards):
    manifest = Path(manifest).expanduser().resolve()
    treatment_manifest = Path(treatment_manifest).expanduser().resolve()
    raw_root = Path(raw_root).expanduser().resolve()
    split_registry = Path(split_registry).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    shards = [
        output_root / f"moment_shard{index:02d}-of-{int(num_shards):02d}.parquet"
        for index in range(int(num_shards))
    ]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"channel-moment shards are incomplete: {missing}")
    all_sites = pd.read_parquet(manifest, columns=["global_index"])
    moments = pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True)
    moments = moments.sort_values("global_index").reset_index(drop=True)
    if moments.global_index.duplicated().any():
        raise ValueError("channel-moment shards overlap")
    if set(map(int, moments.global_index)) != set(map(int, all_sites.global_index)):
        raise ValueError("channel-moment shards do not cover the full site manifest")
    moment_path = output_root / "channel_moments.parquet"
    _atomic_parquet(moment_path, moments)

    source_registry = json.loads(split_registry.read_text())
    normalization_rows = []
    study_splits = []
    for fold in source_registry["folds"]:
        assignment_path = Path(fold["assignment"]).expanduser().resolve()
        if _sha256(assignment_path) != fold["assignment_sha256"]:
            raise ValueError(f"split assignment checksum changed: {assignment_path}")
        assignment = pd.read_parquet(
            assignment_path, columns=["global_index", "role", "experiment"])
        train_indices = assignment.loc[assignment.role == "train", ["global_index"]]
        train_moments = train_indices.merge(
            moments, on="global_index", how="left", validate="one_to_one")
        if train_moments.isna().any().any():
            raise ValueError(f"missing moments for {fold['split_id']} source training sites")
        normalization = _normalization(train_moments)
        normalization_rows.append({
            "split_id": fold["split_id"], "n_training_sites": len(train_moments),
            **normalization,
        })
        study_splits.append({
            "split_id": fold["split_id"],
            "source_experiments": fold["source_experiments"],
            "target_experiments": fold["target_experiments"],
            "source_experiment_names": fold["source_experiment_names"],
            "target_experiment_names": fold["target_experiment_names"],
            "target_experiments_by_cell": fold["target_experiments_by_cell"],
            "assignment": str(assignment_path),
            "assignment_sha256": fold["assignment_sha256"],
            "coverage": fold["coverage"], "normalization": normalization,
            "physical_target_exclusion": False,
        })

    normalization_path = output_root / "normalization_registry.json"
    _atomic_json(normalization_path, {
        "schema_version": 1, "moment_manifest": str(manifest),
        "moment_manifest_sha256": _sha256(manifest),
        "channel_moments": str(moment_path),
        "channel_moments_sha256": _sha256(moment_path),
        "folds": normalization_rows,
    })
    study_registry_path = output_root / "study_registry.json"
    study_registry = {
        "schema_version": 1, "dataset": "RxRx1 full native six-channel",
        "cell_types": ["HEPG2", "HUVEC", "RPE", "U2OS"],
        "n_sites": 125_510, "n_treatment_sites": 112_824,
        "n_labels": 1_108, "n_experiments": 51,
        "site_manifest": str(treatment_manifest),
        "site_manifest_sha256": _sha256(treatment_manifest),
        "pretraining_site_manifest": str(manifest),
        "pretraining_site_manifest_sha256": _sha256(manifest),
        "raw_root": str(raw_root), "main_training_splits": study_splits,
        "primary_splits": study_splits,
        "split_registry": str(split_registry),
        "split_registry_sha256": _sha256(split_registry),
        "normalization_registry": str(normalization_path),
        "normalization_registry_sha256": _sha256(normalization_path),
        "target_policy": (
            "target experiments are excluded from supervised training, source-IID selection, "
            "and primary source-only MAE pretraining"
        ),
    }
    _atomic_json(study_registry_path, study_registry)
    ready = {
        "schema_version": 1, "state": "ready", "study_registry": str(study_registry_path),
        "study_registry_sha256": _sha256(study_registry_path),
        "channel_moments": str(moment_path), "folds": len(study_splits),
        "sites": len(moments),
    }
    _atomic_json(output_root / "FULL_RXRX1_READY.json", ready)
    print(json.dumps(ready, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--treatment-manifest")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--split-registry")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        if args.shard_index is not None:
            parser.error("--finalize and --shard-index are mutually exclusive")
        if not args.treatment_manifest or not args.split_registry:
            parser.error("finalization requires --treatment-manifest and --split-registry")
        finalize(
            args.manifest, args.treatment_manifest, args.raw_root,
            args.split_registry, args.output_root, args.num_shards)
    else:
        if args.shard_index is None:
            parser.error("extraction requires --shard-index")
        extract_shard(
            args.manifest, args.raw_root, args.output_root,
            args.shard_index, args.num_shards, args.workers)


if __name__ == "__main__":
    main()
