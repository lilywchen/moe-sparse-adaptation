import csv
import io

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from scripts.audit_rxrx3_core_images import audit_images


def _image_bytes():
    handle = io.BytesIO()
    Image.new("L", (512, 512), color=17).save(handle, format="PNG")
    return handle.getvalue()


def _write_fixture(root, missing_channel=False):
    data = root / "data"
    data.mkdir()
    payload = _image_bytes()
    keys, images = [], []
    wells = (("gene-001", 1, "A01"), ("gene-101", 2, "B02"))
    for experiment, plate, address in wells:
        channels = range(1, 6 if missing_channel and experiment == "gene-101" else 7)
        for channel in channels:
            keys.append(f"{experiment}/Plate{plate}/{address}_s1_{channel}")
            images.append({"bytes": payload, "path": None})
    table = pa.table({
        "__key__": keys,
        "jp2": pa.array(images, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    })
    shard = data / "train-00000-of-00001.parquet"
    pq.write_table(table, shard, row_group_size=6)

    manifest = root / "manifest.tsv"
    fields = ("split", "well_id", "gene", "experiment_name", "plate", "address")
    with open(manifest, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow({
            "split": "train", "well_id": "gene-001_1_A01", "gene": "G1",
            "experiment_name": "gene-001", "plate": 1, "address": "A01",
        })
        writer.writerow({
            "split": "ood_test", "well_id": "gene-101_2_B02", "gene": "G1",
            "experiment_name": "gene-101", "plate": 2, "address": "B02",
        })
    return data, manifest, shard.stat().st_size


def test_image_audit_validates_six_channel_join_and_pixels(tmp_path):
    data, manifest, size = _write_fixture(tmp_path)
    report = audit_images(data, [manifest], expected_shards=1, expected_bytes=size)
    assert report["passed"] is True
    assert report["channel_rows"] == 12
    assert report["unique_wells"] == 2
    assert report["manifest_union_wells"] == 2
    assert report["decoded_samples"][0]["size"] == [512, 512]
    assert report["decoded_samples"][0]["mode"] == "L"


def test_image_audit_rejects_missing_selected_channel(tmp_path):
    data, manifest, size = _write_fixture(tmp_path, missing_channel=True)
    report = audit_images(data, [manifest], expected_shards=1, expected_bytes=size)
    assert report["passed"] is False
    assert report["failures"]["incomplete_wells"] == 1
    assert report["failures"]["incomplete_manifest_wells"] == 1
