import csv
import gzip
import json

from scripts.build_jump_compound_cross_source_manifest import build_manifests


def _write_csv(path, fieldnames, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_cross_source_manifest_is_deterministic_and_excludes_controls(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    compounds = ["JCP_A", "JCP_B", "JCP_C", "JCP_CONTROL"]
    _write_csv(metadata / "compound.csv.gz", ["Metadata_JCP2022"],
               [{"Metadata_JCP2022": item} for item in compounds])
    _write_csv(metadata / "perturbation_control.csv",
               ["Metadata_JCP2022", "Metadata_pert_type"],
               [{"Metadata_JCP2022": "JCP_CONTROL", "Metadata_pert_type": "negcon"}])
    sources = ("source_1", "source_2", "source_3", "source_8", "source_10")
    wells = []
    for source in sources:
        for index, class_id in enumerate(("JCP_A", "JCP_B", "JCP_CONTROL")):
            wells.append({"Metadata_Source": source, "Metadata_Plate": f"P{index}",
                          "Metadata_Well": f"A0{index + 1}", "Metadata_JCP2022": class_id})
    for source in sources[:-1]:
        wells.append({"Metadata_Source": source, "Metadata_Plate": "PC",
                      "Metadata_Well": "B01", "Metadata_JCP2022": "JCP_C"})
    wells.append({"Metadata_Source": "source_1", "Metadata_Plate": "PZ",
                  "Metadata_Well": "Z99", "Metadata_JCP2022": "JCP_A"})
    _write_csv(metadata / "well.csv.gz",
               ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"], wells)

    output_a, output_b = tmp_path / "a", tmp_path / "b"
    first = build_manifests(metadata, output_a, class_counts=(1, "all"), seed=7)
    second = build_manifests(metadata, output_b, class_counts=(1, "all"), seed=7)
    assert first["common_noncontrol_compounds"] == 2
    assert first["source_split"] == {
        "train": ["source_1", "source_2", "source_3"],
        "ood_val": "source_8", "ood_test": "source_10",
    }
    assert first["cell_dino_channel_order"] == ["DNA", "ER", "RNA", "AGP", "Mito"]
    assert first["training_ready"] is False
    manifest_a = (output_a / "jump_compound_cross_source_all.tsv").read_text()
    manifest_b = (output_b / "jump_compound_cross_source_all.tsv").read_text()
    assert manifest_a == manifest_b
    rows = list(csv.DictReader(manifest_a.splitlines(), delimiter="\t"))
    assert len(rows) == 10
    assert {row["class_id"] for row in rows} == {"JCP_A", "JCP_B"}
    assert {row["split"] for row in rows} == {"train", "ood_val", "ood_test"}
    assert json.loads((output_a / "jump_compound_cross_source_summary.json").read_text())[
        "manifests"
    ][1]["row_count"] == 10
