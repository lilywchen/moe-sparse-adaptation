#!/usr/bin/env python3
"""Build deterministic JUMP-CP compound-classification manifests from metadata only.

The output deliberately stops at source/plate/well identifiers. Image paths and
site identifiers must be joined from an official load-data index before training.
"""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


DEFAULT_TRAIN_SOURCES = ("source_1", "source_2", "source_3")
DEFAULT_VAL_SOURCE = "source_8"
DEFAULT_TEST_SOURCE = "source_10"
CELL_DINO_CHANNELS = ("DNA", "ER", "RNA", "AGP", "Mito")


def _open_csv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, "rt", newline="")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(seed, *values):
    payload = "\x1f".join((str(seed), *map(str, values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metadata_file(root, stem):
    for name in (f"{stem}.csv.gz", f"{stem}.csv"):
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing {stem}.csv[.gz] under {root}")


def _read_ids(path, column):
    with _open_csv(path) as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path} lacks required column {column}")
        return {row[column] for row in reader if row.get(column)}


def build_rows(metadata_root, train_sources=DEFAULT_TRAIN_SOURCES,
               val_source=DEFAULT_VAL_SOURCE, test_source=DEFAULT_TEST_SOURCE, seed=0):
    metadata_root = Path(metadata_root)
    compound_path = _metadata_file(metadata_root, "compound")
    control_path = _metadata_file(metadata_root, "perturbation_control")
    well_path = _metadata_file(metadata_root, "well")
    compounds = _read_ids(compound_path, "Metadata_JCP2022")
    controls = _read_ids(control_path, "Metadata_JCP2022")
    eligible = compounds - controls

    sources = (*train_sources, val_source, test_source)
    if len(sources) != len(set(sources)):
        raise ValueError("train/validation/test sources must be disjoint")
    selected = {source: {} for source in sources}
    with _open_csv(well_path) as handle:
        reader = csv.DictReader(handle)
        required = {"Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{well_path} lacks required columns: {sorted(missing)}")
        for row in reader:
            source = row["Metadata_Source"]
            class_id = row["Metadata_JCP2022"]
            if source not in selected or class_id not in eligible:
                continue
            candidate = {
                "source": source,
                "plate": row["Metadata_Plate"],
                "well": row["Metadata_Well"],
                "class_id": class_id,
            }
            key = _rank(seed, source, class_id, candidate["plate"], candidate["well"])
            current = selected[source].get(class_id)
            if current is None or key < current[0]:
                selected[source][class_id] = (key, candidate)

    common = set.intersection(*(set(selected[source]) for source in sources))
    ranked_classes = sorted(common, key=lambda class_id: (_rank(seed, "class", class_id), class_id))
    return ranked_classes, selected


def _parse_class_counts(value):
    parsed = []
    for token in value.split(","):
        token = token.strip().lower()
        if not token:
            continue
        item = "all" if token == "all" else int(token)
        if item != "all" and item < 1:
            raise ValueError("class counts must be positive")
        if item not in parsed:
            parsed.append(item)
    if not parsed:
        raise ValueError("at least one class count is required")
    return parsed


def _write_tsv(path, class_ids, selected, train_sources, val_source, test_source):
    split_for = {source: "train" for source in train_sources}
    split_for[val_source] = "ood_val"
    split_for[test_source] = "ood_test"
    fields = ("split", "label", "class_id", "class_rank", "source", "plate", "well")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for label, class_id in enumerate(class_ids):
            for source in (*train_sources, val_source, test_source):
                row = selected[source][class_id][1]
                writer.writerow({
                    "split": split_for[source], "label": label, "class_id": class_id,
                    "class_rank": label, "source": source, "plate": row["plate"],
                    "well": row["well"],
                })


def build_manifests(metadata_root, output_dir, class_counts=(1024, 4096, "all"),
                    train_sources=DEFAULT_TRAIN_SOURCES, val_source=DEFAULT_VAL_SOURCE,
                    test_source=DEFAULT_TEST_SOURCE, seed=0):
    metadata_root, output_dir = Path(metadata_root), Path(output_dir)
    ranked_classes, selected = build_rows(
        metadata_root, train_sources, val_source, test_source, seed
    )
    if not ranked_classes:
        raise ValueError("no non-control compound is present in every requested source")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for requested in class_counts:
        count = len(ranked_classes) if requested == "all" else int(requested)
        if count > len(ranked_classes):
            raise ValueError(f"requested {count} classes but only {len(ranked_classes)} are common")
        suffix = "all" if requested == "all" else str(count)
        path = output_dir / f"jump_compound_cross_source_{suffix}.tsv"
        _write_tsv(path, ranked_classes[:count], selected, train_sources, val_source, test_source)
        outputs.append({"class_count": count, "row_count": count * (len(train_sources) + 2),
                        "path": path.name, "sha256": _sha256(path)})

    metadata_files = {
        stem: _metadata_file(metadata_root, stem)
        for stem in ("compound", "perturbation_control", "well")
    }
    summary = {
        "schema_version": 1,
        "task": "compound_perturbation_identification",
        "selection_split": "ood_val",
        "test_readout": "descriptive_fixed_arm",
        "seed": seed,
        "source_split": {
            "train": list(train_sources), "ood_val": val_source, "ood_test": test_source,
        },
        "common_noncontrol_compounds": len(ranked_classes),
        "sampling": "one deterministic well per compound per source; site unresolved",
        "cell_dino_channel_order": list(CELL_DINO_CHANNELS),
        "training_ready": False,
        "blocker": "join source/plate/well to an official image load-data index and choose a site",
        "manifests": outputs,
        "metadata": {
            stem: {"path": path.name, "sha256": _sha256(path)}
            for stem, path in metadata_files.items()
        },
    }
    summary_path = output_dir / "jump_compound_cross_source_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-sources", default=",".join(DEFAULT_TRAIN_SOURCES))
    parser.add_argument("--val-source", default=DEFAULT_VAL_SOURCE)
    parser.add_argument("--test-source", default=DEFAULT_TEST_SOURCE)
    parser.add_argument("--class-counts", default="1024,4096,all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = build_manifests(
        args.metadata_root, args.output_dir, _parse_class_counts(args.class_counts),
        tuple(source.strip() for source in args.train_sources.split(",") if source.strip()),
        args.val_source, args.test_source, args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
