#!/usr/bin/env python3
"""Validate RxRx3-core image shards and six-channel joins before training."""

import argparse
import csv
import io
import json
import re
from pathlib import Path

from PIL import Image


KEY_PATTERN = re.compile(
    r"^(?P<experiment>[^/]+)/Plate(?P<plate>[0-9]+)/"
    r"(?P<address>[A-Z]{1,2}[0-9]{2})_s1_(?P<channel>[1-6])$"
)
FULL_CHANNEL_MASK = (1 << 6) - 1


def _manifest_wells(manifest_paths):
    wells = set()
    summaries = []
    for path in sorted(map(Path, manifest_paths)):
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"split", "well_id", "gene", "experiment_name", "plate", "address"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} lacks manifest columns: {sorted(missing)}")
            rows = list(reader)
        manifest_wells = {row["well_id"] for row in rows}
        wells.update(manifest_wells)
        summaries.append({
            "path": path.name,
            "rows": len(rows),
            "unique_wells": len(manifest_wells),
            "splits": sorted({row["split"] for row in rows}),
        })
    if not summaries:
        raise ValueError("at least one manifest is required")
    return wells, summaries


def _well_and_channel(key):
    match = KEY_PATTERN.fullmatch(key)
    if match is None:
        return None, None
    values = match.groupdict()
    well_id = f"{values['experiment']}_{int(values['plate'])}_{values['address']}"
    return well_id, int(values["channel"])


def audit_images(data_dir, manifest_paths, expected_shards=35,
                 expected_bytes=17_390_577_507, decode_per_shard=1):
    """Return a strict audit dictionary without mutating the dataset."""
    import pyarrow.parquet as pq

    data_dir = Path(data_dir)
    shards = sorted(data_dir.glob("*.parquet"))
    selected_wells, manifests = _manifest_wells(manifest_paths)
    channel_masks = {}
    row_count = 0
    duplicate_channels = 0
    malformed_keys = []
    decoded = []
    schema_errors = []

    for shard in shards:
        parquet = pq.ParquetFile(shard)
        if set(parquet.schema_arrow.names) != {"__key__", "jp2"}:
            schema_errors.append({"path": shard.name, "columns": parquet.schema_arrow.names})
            continue
        for row_group in range(parquet.metadata.num_row_groups):
            keys = parquet.read_row_group(row_group, columns=["__key__"]).column(0).to_pylist()
            row_count += len(keys)
            for key in keys:
                well_id, channel = _well_and_channel(key)
                if well_id is None:
                    if len(malformed_keys) < 20:
                        malformed_keys.append(key)
                    continue
                bit = 1 << (channel - 1)
                previous = channel_masks.get(well_id, 0)
                if previous & bit:
                    duplicate_channels += 1
                channel_masks[well_id] = previous | bit

        for sample_index in range(min(decode_per_shard, parquet.metadata.num_rows)):
            table = parquet.read_row_group(sample_index, columns=["__key__", "jp2"])
            key = table.column("__key__")[0].as_py()
            payload = table.column("jp2")[0].as_py()
            raw = payload.get("bytes") if isinstance(payload, dict) else None
            if not raw:
                decoded.append({"shard": shard.name, "key": key, "error": "missing bytes"})
                continue
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    decoded.append({
                        "shard": shard.name, "key": key, "size": list(image.size),
                        "mode": image.mode, "format": image.format,
                    })
            except Exception as error:  # pragma: no cover - exercised by corrupted real data
                decoded.append({"shard": shard.name, "key": key, "error": repr(error)})

    incomplete = [well for well, mask in channel_masks.items() if mask != FULL_CHANNEL_MASK]
    missing_selected = sorted(selected_wells - set(channel_masks))
    incomplete_selected = sorted(
        well for well in selected_wells
        if well in channel_masks and channel_masks[well] != FULL_CHANNEL_MASK
    )
    decode_errors = [item for item in decoded if "error" in item]
    decode_shape_errors = [
        item for item in decoded
        if "error" not in item and (item["size"] != [512, 512] or item["mode"] != "L")
    ]
    actual_bytes = sum(path.stat().st_size for path in shards)
    checks = {
        "shard_count": len(shards) == expected_shards,
        "shard_bytes": actual_bytes == expected_bytes,
        "schema": not schema_errors,
        "key_parse": not malformed_keys,
        "unique_channel_rows": duplicate_channels == 0,
        "all_wells_have_six_channels": not incomplete and row_count == 6 * len(channel_masks),
        "all_manifest_wells_present": not missing_selected,
        "manifest_wells_have_six_channels": not incomplete_selected,
        "decoded_samples": len(decoded) == len(shards) * decode_per_shard,
        "decoded_pixels": not decode_errors and not decode_shape_errors,
    }
    return {
        "schema_version": 1,
        "data_dir": str(data_dir),
        "shards": len(shards),
        "shard_bytes": actual_bytes,
        "channel_rows": row_count,
        "unique_wells": len(channel_masks),
        "manifest_union_wells": len(selected_wells),
        "manifests": manifests,
        "decoded_samples": decoded,
        "failures": {
            "schema_errors": schema_errors,
            "malformed_keys_first20": malformed_keys,
            "duplicate_channels": duplicate_channels,
            "incomplete_wells": len(incomplete),
            "incomplete_wells_first20": sorted(incomplete)[:20],
            "missing_manifest_wells": len(missing_selected),
            "missing_manifest_wells_first20": missing_selected[:20],
            "incomplete_manifest_wells": len(incomplete_selected),
            "incomplete_manifest_wells_first20": incomplete_selected[:20],
            "decode_errors": decode_errors,
            "decode_shape_errors": decode_shape_errors,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-shards", type=int, default=35)
    parser.add_argument("--expected-bytes", type=int, default=17_390_577_507)
    parser.add_argument("--decode-per-shard", type=int, default=1)
    args = parser.parse_args()
    report = audit_images(
        args.data_dir, args.manifest, args.expected_shards, args.expected_bytes,
        args.decode_per_shard,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
