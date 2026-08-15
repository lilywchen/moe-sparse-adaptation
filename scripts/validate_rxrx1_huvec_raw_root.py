#!/usr/bin/env python
"""Validate every frozen six-channel HUVEC site before a remote sweep is released."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from moe_shift.data.rxrx1_huvec import _native_channel_paths


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-manifest", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--allow-extra-folders", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.site_manifest).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve()
    marker = Path(args.marker).expanduser().resolve()
    sites = pd.read_parquet(manifest)
    required = {"global_index", "relative_path", "experiment", "well_id", "site"}
    missing_columns = required - set(sites)
    if missing_columns:
        raise ValueError(f"site manifest is missing columns: {sorted(missing_columns)}")
    if sites.global_index.duplicated().any():
        raise ValueError("global_index must identify one frozen site")

    folders = sorted({Path(str(value)).parts[1] for value in sites.relative_path})
    available = sorted(path.name for path in (raw_root / "images").glob("HUVEC-*")
                       if path.is_dir())
    missing_folders = set(folders) - set(available)
    unexpected_folders = set(available) - set(folders)
    if missing_folders or (unexpected_folders and not args.allow_extra_folders):
        raise ValueError(
            f"HUVEC folder mismatch: missing={sorted(missing_folders)}, "
            f"unexpected={sorted(unexpected_folders)}")

    missing_paths = []
    referenced_channels = 0
    for row in sites.itertuples(index=False):
        paths = _native_channel_paths(raw_root, Path(str(row.relative_path)))
        referenced_channels += len(paths)
        for path in paths:
            if not path.is_file():
                missing_paths.append(str(path))
                if len(missing_paths) >= 20:
                    break
        if len(missing_paths) >= 20:
            break
    if missing_paths:
        raise FileNotFoundError(
            f"frozen HUVEC raw root is incomplete; first missing channels: {missing_paths}")

    payload = {
        "schema_version": 1,
        "state": "complete",
        "site_manifest": str(manifest),
        "site_manifest_sha256": _sha256(manifest),
        "raw_root": str(raw_root),
        "n_sites": int(len(sites)),
        "n_wells": int(sites.well_id.nunique()),
        "n_experiments": int(sites.experiment.nunique()),
        "experiment_folders": folders,
        "extra_folders_allowed": bool(args.allow_extra_folders),
        "referenced_channels_checked": int(referenced_channels),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(marker.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, marker)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
