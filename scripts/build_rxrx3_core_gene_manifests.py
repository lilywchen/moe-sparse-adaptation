#!/usr/bin/env python3
"""Audit RxRx3-core metadata and build deterministic gene-classification curves.

The builder never reads or writes image pixels.  It pairs the two experiments
with identical query-gene sets, assigns one experiment to training and its mate
to OOD test, reserves one complete training plate for ID validation, and emits
two separate nested training curves:

* plate count: 1/2/4/8 training plates with four guides fixed;
* guide count: 1/2/4 guides with eight training plates fixed.

All evaluation rows are identical within a curve.  Pixel staging and the exact
six-channel join remain explicit launch gates.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


REQUIRED_COLUMNS = {
    "well_id", "experiment_name", "plate", "address", "gene", "treatment",
    "perturbation_type", "cell_type", "well_type_label",
}
CHANNEL_ORDER = ("Hoechst", "ConA", "Phalloidin", "Syto14", "MitoTracker", "WGA")
CELL_DINO_MAP = ("w1", "w2", "w4", "mean(w3,w6)", "w5")


def _rank(seed, *values):
    payload = "\x1f".join((str(seed), *map(str, values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_counts(value):
    counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not counts or any(item < 1 for item in counts) or tuple(sorted(set(counts))) != counts:
        raise ValueError("counts must be unique positive integers in increasing order")
    return counts


def _read_rows(metadata_path):
    metadata_path = Path(metadata_path)
    with open(metadata_path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{metadata_path} lacks required columns: {sorted(missing)}")
        rows = list(reader)
    well_ids = [row["well_id"] for row in rows]
    if len(well_ids) != len(set(well_ids)):
        raise ValueError("metadata contains duplicate well_id values")
    return rows


def _per_class_stats(rows):
    counts = sorted(Counter(row["gene"] for row in rows).values())
    return {
        "rows": len(rows),
        "classes": len(counts),
        "per_class_min": min(counts) if counts else 0,
        "per_class_median": median(counts) if counts else 0,
        "per_class_max": max(counts) if counts else 0,
    }


def _write_manifest(path, rows, label_for):
    fields = (
        "split", "label", "gene", "experiment_name", "plate", "address", "well_id",
        "guide", "perturbation_type", "cell_type",
    )
    rows = sorted(
        rows,
        key=lambda row: (
            {"train": 0, "id_val": 1, "ood_test": 2}[row["split"]],
            label_for[row["gene"]], row["experiment_name"], int(row["plate"]),
            row["address"], row["treatment"],
        ),
    )
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "split": row["split"],
                "label": label_for[row["gene"]],
                "gene": row["gene"],
                "experiment_name": row["experiment_name"],
                "plate": row["plate"],
                "address": row["address"],
                "well_id": row["well_id"],
                "guide": row["treatment"],
                "perturbation_type": row["perturbation_type"],
                "cell_type": row["cell_type"],
            })


def build_manifests(metadata_path, output_dir, seed=0, plate_counts=(1, 2, 4, 8),
                    guide_counts=(1, 2, 4)):
    """Build manifests and return a complete metadata/split audit summary."""
    plate_counts, guide_counts = tuple(plate_counts), tuple(guide_counts)
    if tuple(sorted(set(plate_counts))) != plate_counts or min(plate_counts) < 1:
        raise ValueError("plate_counts must be unique positive integers in increasing order")
    if tuple(sorted(set(guide_counts))) != guide_counts or min(guide_counts) < 1:
        raise ValueError("guide_counts must be unique positive integers in increasing order")
    max_plates, max_guides = max(plate_counts), max(guide_counts)
    rows = _read_rows(metadata_path)
    query = [row for row in rows if row["well_type_label"] == "Query guides"]
    if not query:
        raise ValueError("metadata contains no Query guides rows")
    if {row["perturbation_type"] for row in query} != {"CRISPR"}:
        raise ValueError("Query guides must all have perturbation_type=CRISPR")
    if any(not row["gene"] or not row["treatment"] for row in query):
        raise ValueError("Query guides rows require non-empty gene and treatment")

    genes_by_experiment = defaultdict(set)
    guides_by_gene_experiment = defaultdict(set)
    plates_by_gene_experiment_guide = defaultdict(set)
    all_plates_by_experiment = defaultdict(set)
    for row in query:
        experiment, gene, guide, plate = (
            row["experiment_name"], row["gene"], row["treatment"], row["plate"]
        )
        genes_by_experiment[experiment].add(gene)
        guides_by_gene_experiment[(gene, experiment)].add(guide)
        plates_by_gene_experiment_guide[(gene, experiment, guide)].add(plate)
        all_plates_by_experiment[experiment].add(plate)

    experiments_by_gene_set = defaultdict(list)
    for experiment, genes in genes_by_experiment.items():
        experiments_by_gene_set[tuple(sorted(genes))].append(experiment)
    candidate_pairs = [
        (gene_set, tuple(sorted(experiments)))
        for gene_set, experiments in experiments_by_gene_set.items()
        if len(experiments) == 2
    ]
    candidate_pairs.sort(key=lambda item: item[1])
    candidate_experiments = {experiment for _, pair in candidate_pairs for experiment in pair}

    experiments_by_gene = defaultdict(set)
    for gene_set, pair in candidate_pairs:
        for gene in gene_set:
            experiments_by_gene[gene].update(pair)
    common_guides = {}
    for gene, experiments in experiments_by_gene.items():
        common_guides[gene] = set.intersection(*(
            guides_by_gene_experiment[(gene, experiment)] for experiment in sorted(experiments)
        ))
    initially_eligible = {
        gene for gene, guides in common_guides.items() if len(guides) >= max_guides
    }
    selected_guides = {}
    for gene in initially_eligible:
        experiments = experiments_by_gene[gene]
        selected_guides[gene] = tuple(sorted(
            common_guides[gene],
            key=lambda guide: (
                -min(len(plates_by_gene_experiment_guide[(gene, experiment, guide)])
                     for experiment in experiments),
                _rank(seed, "guide", gene, guide), guide,
            ),
        )[:max_guides])

    rows_by_experiment_plate = defaultdict(list)
    labels_by_experiment_plate = defaultdict(set)
    eligible_plates_by_experiment = defaultdict(set)
    for row in query:
        experiment, gene = row["experiment_name"], row["gene"]
        if experiment not in candidate_experiments or gene not in initially_eligible:
            continue
        if row["treatment"] not in selected_guides[gene]:
            continue
        key = (experiment, row["plate"])
        rows_by_experiment_plate[key].append(row)
        labels_by_experiment_plate[key].add(gene)
        eligible_plates_by_experiment[experiment].add(row["plate"])

    # Requiring two individually class-complete plates makes the one-plate train point and
    # one-plate ID validation honest. Two of 87 real pairs fail this QC-completeness rule.
    valid_pairs = []
    excluded_pairs = []
    for gene_set, pair in candidate_pairs:
        target_genes = set(gene_set) & initially_eligible
        complete_plates = {
            experiment: [
                plate for plate in eligible_plates_by_experiment[experiment]
                if target_genes <= labels_by_experiment_plate[(experiment, plate)]
            ]
            for experiment in pair
        }
        if target_genes and all(len(complete_plates[experiment]) >= 2 for experiment in pair):
            valid_pairs.append((gene_set, pair, complete_plates))
        else:
            excluded_pairs.append({
                "experiments": list(pair),
                "eligible_classes": len(target_genes),
                "complete_plate_counts": {
                    experiment: len(complete_plates[experiment]) for experiment in pair
                },
            })

    final_genes = {
        gene for gene_set, _, _ in valid_pairs for gene in gene_set
        if gene in initially_eligible
    }
    if not final_genes:
        raise ValueError("no experiment pair survives the guide and plate-completeness gates")
    ranked_genes = sorted(final_genes, key=lambda gene: (_rank(seed, "class", gene), gene))
    label_for = {gene: index for index, gene in enumerate(ranked_genes)}

    roles = {}
    train_plate_order = {}
    validation_plate = {}
    for gene_set, pair, complete_plates in valid_pairs:
        # Prefer the member with more QC-retained plates for training, then break ties by hash.
        train_experiment = min(
            pair,
            key=lambda experiment: (
                -len(all_plates_by_experiment[experiment]),
                _rank(seed, "train-role", *gene_set, experiment), experiment,
            ),
        )
        test_experiment = pair[1] if pair[0] == train_experiment else pair[0]
        full = sorted(
            complete_plates[train_experiment],
            key=lambda plate: (_rank(seed, "complete-plate", train_experiment, plate), plate),
        )
        validation_plate[train_experiment] = full[0]
        first_train = full[1]
        remaining = sorted(
            eligible_plates_by_experiment[train_experiment] - {full[0], full[1]},
            key=lambda plate: (_rank(seed, "train-plate", train_experiment, plate), plate),
        )
        train_plate_order[train_experiment] = (first_train, *remaining)
        if len(train_plate_order[train_experiment]) < max_plates:
            raise ValueError(
                f"{train_experiment} has only {len(train_plate_order[train_experiment])} "
                f"training plates after validation; requested {max_plates}"
            )
        roles[train_experiment] = "train"
        roles[test_experiment] = "ood_test"

    train_experiments = sorted(experiment for experiment, role in roles.items() if role == "train")
    test_experiments = sorted(experiment for experiment, role in roles.items() if role == "ood_test")

    def evaluation_rows():
        output = []
        for experiment in train_experiments:
            for row in rows_by_experiment_plate[(experiment, validation_plate[experiment])]:
                if row["gene"] in final_genes:
                    output.append({**row, "split": "id_val"})
        for experiment in test_experiments:
            for plate in sorted(eligible_plates_by_experiment[experiment], key=int):
                for row in rows_by_experiment_plate[(experiment, plate)]:
                    if row["gene"] in final_genes:
                        output.append({**row, "split": "ood_test"})
        return output

    fixed_evaluation = evaluation_rows()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []

    for count in plate_counts:
        train_rows = []
        for experiment in train_experiments:
            for plate in train_plate_order[experiment][:count]:
                for row in rows_by_experiment_plate[(experiment, plate)]:
                    if row["gene"] in final_genes:
                        train_rows.append({**row, "split": "train"})
        path = output_dir / f"rxrx3_core_gene_plate_{count}.tsv"
        _write_manifest(path, train_rows + fixed_evaluation, label_for)
        stats = _per_class_stats(train_rows)
        if stats["classes"] != len(final_genes):
            raise ValueError(f"plate-{count} training point loses class coverage: {stats}")
        manifests.append({
            "axis": "train_plate_count", "scale": count, "path": path.name,
            "sha256": _sha256(path), "train": stats,
        })

    selected_prefix = {
        gene: tuple(selected_guides[gene]) for gene in final_genes
    }
    for count in guide_counts:
        allowed = {gene: set(guides[:count]) for gene, guides in selected_prefix.items()}
        train_rows = []
        for experiment in train_experiments:
            for plate in train_plate_order[experiment][:max_plates]:
                for row in rows_by_experiment_plate[(experiment, plate)]:
                    if row["gene"] in final_genes and row["treatment"] in allowed[row["gene"]]:
                        train_rows.append({**row, "split": "train"})
        path = output_dir / f"rxrx3_core_gene_guides_{count}.tsv"
        _write_manifest(path, train_rows + fixed_evaluation, label_for)
        stats = _per_class_stats(train_rows)
        if stats["classes"] != len(final_genes):
            raise ValueError(f"guide-{count} training point loses class coverage: {stats}")
        manifests.append({
            "axis": "train_guide_count", "scale": count, "path": path.name,
            "sha256": _sha256(path), "train": stats,
        })

    evaluation_stats = {
        split: _per_class_stats([row for row in fixed_evaluation if row["split"] == split])
        for split in ("id_val", "ood_test")
    }
    if any(stats["classes"] != len(final_genes) for stats in evaluation_stats.values()):
        raise ValueError(f"evaluation split loses class coverage: {evaluation_stats}")

    query_experiment_counts = Counter()
    for gene in {row["gene"] for row in query}:
        query_experiment_counts[len({
            row["experiment_name"] for row in query if row["gene"] == gene
        })] += 1
    summary = {
        "schema_version": 1,
        "task": "crispr_gene_perturbation_identification",
        "atomic_unit": "well (six channel files joined by well_id)",
        "seed": seed,
        "metadata": {"path": Path(metadata_path).name, "sha256": _sha256(metadata_path)},
        "dataset": {
            "rows": len(rows), "unique_wells": len({row["well_id"] for row in rows}),
            "query_guide_wells": len(query),
            "experiments": len(genes_by_experiment),
            "experiment_plate_pairs": len({
                (row["experiment_name"], row["plate"]) for row in rows
            }),
            "cell_types": sorted({row["cell_type"] for row in rows}),
            "query_genes": len({row["gene"] for row in query}),
            "query_gene_experiment_count_histogram": {
                str(key): value for key, value in sorted(query_experiment_counts.items())
            },
        },
        "split": {
            "candidate_identical_gene_set_pairs": len(candidate_pairs),
            "valid_pairs": len(valid_pairs),
            "excluded_pairs": excluded_pairs,
            "classes": len(final_genes),
            "train_experiments": len(train_experiments),
            "ood_test_experiments": len(test_experiments),
            "id_validation_plates": len(validation_plate),
            "train_ood_experiment_overlap": sorted(set(train_experiments) & set(test_experiments)),
            "evaluation": evaluation_stats,
        },
        "curves": {
            "train_plate_count": list(plate_counts),
            "train_guide_count": list(guide_counts),
            "fixed_selected_guides_per_class": max_guides,
            "fixed_train_plates_for_guide_curve": max_plates,
            "interpretation": {
                "train_plate_count": "plate-batch count and examples per class co-vary",
                "train_guide_count": "guide diversity/examples per class vary; plate domains fixed",
                "experiment_count": "not identifiable at fixed labels in RxRx3-core",
            },
        },
        "channels": {
            "rxrx3_raw_order": list(CHANNEL_ORDER),
            "cell_dino_6_to_5": list(CELL_DINO_MAP),
        },
        "manifests": manifests,
        "training_ready": False,
        "blocker": (
            "stage the 35 Hugging Face image parquet shards; verify exactly six 512x512 uint8 "
            "JP2 channel rows per selected well and resolve channel keys before training"
        ),
    }
    summary_path = output_dir / "rxrx3_core_gene_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plate-counts", default="1,2,4,8")
    parser.add_argument("--guide-counts", default="1,2,4")
    args = parser.parse_args()
    summary = build_manifests(
        args.metadata, args.output_dir, args.seed,
        _parse_counts(args.plate_counts), _parse_counts(args.guide_counts),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
