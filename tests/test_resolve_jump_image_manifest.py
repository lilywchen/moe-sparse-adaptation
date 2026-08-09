import csv
import json

from scripts.resolve_jump_image_manifest import resolve_manifest


def _write(path, fields, rows, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_resolver_uses_stain_semantics_and_is_deterministic(tmp_path):
    manifest = tmp_path / "manifest.tsv"
    _write(manifest, ["split", "label", "class_id", "source", "plate", "well"], [{
        "split": "train", "label": "0", "class_id": "JCP_A",
        "source": "source_1", "plate": "P1", "well": "A01",
    }], delimiter="\t")
    fields = ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_Site",
              "URL_OrigBrightfield", "URL_OrigRNA", "URL_OrigMito", "URL_OrigER",
              "URL_OrigDNA", "URL_OrigAGP"]
    rows = []
    for site in (1, 2):
        rows.append({
            "Metadata_Source": "source_1", "Metadata_Plate": "P1",
            "Metadata_Well": "A01", "Metadata_Site": site,
            "URL_OrigBrightfield": f"s3://bucket/bf-{site}",
            "URL_OrigRNA": f"s3://bucket/rna-{site}",
            "URL_OrigMito": f"s3://bucket/mito-{site}",
            "URL_OrigER": f"s3://bucket/er-{site}",
            "URL_OrigDNA": f"s3://bucket/dna-{site}",
            "URL_OrigAGP": f"s3://bucket/agp-{site}",
        })
    load_data = tmp_path / "indices" / "batch" / "P1" / "load_data.csv"
    _write(load_data, fields, rows)
    out_a, out_b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    first = resolve_manifest(manifest, tmp_path / "indices", out_a, seed=9)
    second = resolve_manifest(manifest, tmp_path / "indices", out_b, seed=9)
    assert out_a.read_text() == out_b.read_text()
    resolved = next(csv.DictReader(out_a.read_text().splitlines(), delimiter="\t"))
    site = resolved["site"]
    assert [resolved[f"url_{channel}"] for channel in ("DNA", "ER", "RNA", "AGP", "Mito")] == [
        f"s3://bucket/dna-{site}", f"s3://bucket/er-{site}",
        f"s3://bucket/rna-{site}", f"s3://bucket/agp-{site}",
        f"s3://bucket/mito-{site}",
    ]
    assert {key: value for key, value in first.items() if key != "output_manifest"} == {
        key: value for key, value in second.items() if key != "output_manifest"
    }
    assert first["image_urls"] == 5
    assert first["training_ready"] is False
    assert json.loads(out_a.with_suffix(".tsv.summary.json").read_text())["pixels_local"] is False
