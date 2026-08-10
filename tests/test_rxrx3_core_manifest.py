import csv

import pytest

from scripts.build_rxrx3_core_gene_manifests import build_manifests


FIELDS = (
    "well_id", "experiment_name", "plate", "address", "gene", "treatment", "SMILES",
    "concentration", "perturbation_type", "cell_type", "well_type_label",
)


def _write_fixture(path, duplicate=False):
    rows = []
    for pair_index, experiments in enumerate((("gene-001", "gene-101"), ("gene-002", "gene-102"))):
        genes = (f"G{pair_index}A", f"G{pair_index}B")
        for experiment in experiments:
            for plate in range(1, 4):
                for gene in genes:
                    for guide in range(1, 3):
                        rows.append({
                            "well_id": f"{experiment}_{plate}_{gene}_{guide}",
                            "experiment_name": experiment,
                            "plate": str(plate),
                            "address": f"A{len(rows):02d}",
                            "gene": gene,
                            "treatment": f"{gene}_guide_{guide}",
                            "SMILES": "", "concentration": "",
                            "perturbation_type": "CRISPR", "cell_type": "HUVEC",
                            "well_type_label": "Query guides",
                        })
    if duplicate:
        rows.append(dict(rows[0]))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_rxrx3_core_curves_are_nested_disjoint_and_keep_evaluation_fixed(tmp_path):
    metadata = tmp_path / "metadata.csv"
    _write_fixture(metadata)
    output_a, output_b = tmp_path / "a", tmp_path / "b"
    first = build_manifests(
        metadata, output_a, seed=7, plate_counts=(1, 2), guide_counts=(1, 2)
    )
    second = build_manifests(
        metadata, output_b, seed=7, plate_counts=(1, 2), guide_counts=(1, 2)
    )

    assert first["split"]["classes"] == 4
    assert first["split"]["valid_pairs"] == 2
    assert first["split"]["train_experiments"] == 2
    assert first["split"]["ood_test_experiments"] == 2
    assert first["split"]["train_ood_experiment_overlap"] == []
    assert first["curves"]["interpretation"]["experiment_count"].startswith("not identifiable")
    assert first["channels"]["cell_dino_6_to_5"] == [
        "w1", "w2", "w4", "mean(w3,w6)", "w5",
    ]
    assert first["training_ready"] is False

    def read(name, root=output_a):
        with open(root / name, newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    plate_1 = read("rxrx3_core_gene_plate_1.tsv")
    plate_2 = read("rxrx3_core_gene_plate_2.tsv")
    guide_1 = read("rxrx3_core_gene_guides_1.tsv")
    guide_2 = read("rxrx3_core_gene_guides_2.tsv")
    evaluation = lambda rows: {
        tuple(sorted(row.items())) for row in rows if row["split"] != "train"
    }
    assert evaluation(plate_1) == evaluation(plate_2) == evaluation(guide_1) == evaluation(guide_2)
    assert {row["well_id"] for row in plate_1 if row["split"] == "train"} < {
        row["well_id"] for row in plate_2 if row["split"] == "train"
    }
    assert {row["well_id"] for row in guide_1 if row["split"] == "train"} < {
        row["well_id"] for row in guide_2 if row["split"] == "train"
    }
    assert [item["sha256"] for item in first["manifests"]] == [
        item["sha256"] for item in second["manifests"]
    ]


def test_rxrx3_core_rejects_duplicate_wells(tmp_path):
    metadata = tmp_path / "metadata.csv"
    _write_fixture(metadata, duplicate=True)
    with pytest.raises(ValueError, match="duplicate well_id"):
        build_manifests(metadata, tmp_path / "out", plate_counts=(1, 2), guide_counts=(1, 2))
