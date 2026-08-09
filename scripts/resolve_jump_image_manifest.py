#!/usr/bin/env python3
"""Join a JUMP source/plate/well manifest to official load_data.csv image URLs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


CHANNELS = ("DNA", "ER", "RNA", "AGP", "Mito")
URL_COLUMNS = tuple(f"URL_Orig{channel}" for channel in CHANNELS)
KEY_COLUMNS = ("source", "plate", "well")


def _rank(seed, *values):
    payload = "\x1f".join((str(seed), *map(str, values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_manifest(manifest_path, load_data_root, output_path, seed=0):
    manifest_path, load_data_root, output_path = (
        Path(manifest_path), Path(load_data_root), Path(output_path)
    )
    with open(manifest_path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        input_fields = tuple(reader.fieldnames or ())
        missing = set(KEY_COLUMNS) - set(input_fields)
        if missing:
            raise ValueError(f"manifest lacks required columns: {sorted(missing)}")
        manifest_rows = list(reader)
    wanted = {tuple(row[column] for column in KEY_COLUMNS) for row in manifest_rows}
    chosen = {}
    scanned_files = 0
    for path in sorted(load_data_root.rglob("load_data.csv")):
        scanned_files += 1
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_Site",
                *URL_COLUMNS,
            }
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
            for row in reader:
                key = (row["Metadata_Source"], row["Metadata_Plate"], row["Metadata_Well"])
                if key not in wanted:
                    continue
                if any(not row.get(column) for column in URL_COLUMNS):
                    continue
                rank = _rank(seed, *key, row["Metadata_Site"])
                if key not in chosen or rank < chosen[key][0]:
                    chosen[key] = (rank, row)
    unresolved = sorted(wanted - set(chosen))
    if unresolved:
        preview = ", ".join("/".join(key) for key in unresolved[:5])
        raise ValueError(f"{len(unresolved)} manifest wells lack five-channel image rows: {preview}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_fields = (*input_fields, "site", *(f"url_{channel}" for channel in CHANNELS))
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, delimiter="\t")
        writer.writeheader()
        for manifest_row in manifest_rows:
            key = tuple(manifest_row[column] for column in KEY_COLUMNS)
            image_row = chosen[key][1]
            writer.writerow({
                **manifest_row,
                "site": image_row["Metadata_Site"],
                **{f"url_{channel}": image_row[f"URL_Orig{channel}"] for channel in CHANNELS},
            })
    summary = {
        "schema_version": 1,
        "input_manifest": str(manifest_path),
        "output_manifest": str(output_path),
        "seed": seed,
        "resolved_rows": len(manifest_rows),
        "resolved_wells": len(wanted),
        "image_urls": len(manifest_rows) * len(CHANNELS),
        "load_data_files_scanned": scanned_files,
        "cell_dino_channel_order": list(CHANNELS),
        "image_index_ready": True,
        "pixels_local": False,
        "training_ready": False,
        "next_gate": "audit object existence and exact bytes, then stage pixels with checksums",
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--load-data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(resolve_manifest(
        args.manifest, args.load_data_root, args.output, args.seed
    ), indent=2))


if __name__ == "__main__":
    main()
