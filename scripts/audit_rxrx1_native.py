#!/usr/bin/env python
"""Audit native RxRx1 channel coverage without touching the OOD test split."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.data.rxrx1 import _native_channel_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="WILDS root containing rxrx1_v1.0")
    ap.add_argument("--raw-root", required=True, help="official archive root containing images/")
    ap.add_argument("--output", required=True)
    ap.add_argument("--image-smoke-count", type=int, default=64)
    args = ap.parse_args()

    from wilds import get_dataset

    ds = get_dataset(dataset="rxrx1", root_dir=args.data_root, download=False)
    splits = ("train", "id_test", "val")  # OOD test is intentionally not constructed or audited.
    records = {}
    missing = []
    total_samples = 0
    smoke_paths = []
    for split in splits:
        subset = ds.get_subset(split)
        split_missing = 0
        for global_idx in subset.indices:
            paths = _native_channel_paths(args.raw_root, ds._input_array[int(global_idx)])
            absent = [str(path) for path in paths if not path.is_file()]
            if absent:
                split_missing += 1
                if len(missing) < 20:
                    missing.append({"split": split, "paths": absent})
            elif len(smoke_paths) < args.image_smoke_count:
                smoke_paths.extend(paths)
        records[split] = {"samples": len(subset), "samples_missing_any_channel": split_missing}
        total_samples += len(subset)

    modes, sizes = set(), set()
    digest = hashlib.sha256()
    for path in smoke_paths[: args.image_smoke_count * 6]:
        with Image.open(path) as image:
            modes.add(image.mode)
            sizes.add(tuple(image.size))
            digest.update(image.tobytes())

    report = {
        "schema_version": 1,
        "raw_root": str(Path(args.raw_root).resolve()),
        "selection_splits": list(splits),
        "test_evaluated": False,
        "channels_per_sample": 6,
        "total_selection_samples": total_samples,
        "splits": records,
        "missing_examples": missing,
        "smoke": {
            "files_read": min(len(smoke_paths), args.image_smoke_count * 6),
            "modes": sorted(modes),
            "sizes": sorted(list(sizes)),
            "pixel_digest": digest.hexdigest(),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if any(row["samples_missing_any_channel"] for row in records.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
