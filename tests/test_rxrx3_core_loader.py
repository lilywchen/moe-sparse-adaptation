import csv
import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pyarrow")
import pyarrow as pa
import pyarrow.parquet as pq

from moe_shift.data import make_loaders, make_val_loader
from moe_shift.data.rxrx3_core import (
    ParquetLocalitySampler,
    RxRx3CoreDataset,
    build_selected_well_index,
    read_rxrx3_manifest,
)


def _image_bytes(channel, corrupt=False):
    if corrupt:
        return b"not-an-image"
    grid = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16)
    image = Image.fromarray((grid + channel * 7).astype(np.uint8), mode="L")
    handle = io.BytesIO()
    image.save(handle, format="PNG")
    return handle.getvalue()


def _rows():
    output = []
    specifications = [
        ("train", "ExpA", 1, "A01", 0, "G0"),
        ("train", "ExpA", 1, "A02", 1, "G1"),
        ("id_val", "ExpA", 2, "A01", 0, "G0"),
        ("id_val", "ExpA", 2, "A02", 1, "G1"),
        ("ood_test", "ExpB", 1, "A01", 0, "G0"),
        ("ood_test", "ExpB", 1, "A02", 1, "G1"),
    ]
    for split, experiment, plate, address, label, gene in specifications:
        output.append({
            "split": split, "label": label, "gene": gene,
            "experiment_name": experiment, "plate": plate, "address": address,
            "well_id": f"{experiment}_{plate}_{address}", "guide": f"sg{gene}",
            "perturbation_type": "CRISPR", "cell_type": "HUVEC",
        })
    return output


def _write_manifest(path, rows=None):
    rows = list(_rows() if rows is None else rows)
    fields = list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_shard(data_dir, rows=None, omit=None, duplicate=None, corrupt=None):
    rows = list(_rows() if rows is None else rows)
    keys, payloads = [], []
    for row in rows:
        for channel in range(1, 7):
            pair = (row["well_id"], channel)
            if pair == omit:
                continue
            key = (
                f"{row['experiment_name']}/Plate{row['plate']}/"
                f"{row['address']}_s1_{channel}"
            )
            keys.append(key)
            payloads.append({"bytes": _image_bytes(channel, pair == corrupt), "path": None})
            if pair == duplicate:
                keys.append(key)
                payloads.append({"bytes": _image_bytes(channel), "path": None})
    data_dir.mkdir()
    table = pa.table({
        "__key__": pa.array(keys, type=pa.string()),
        "jp2": pa.array(payloads, type=pa.struct([
            pa.field("bytes", pa.binary()), pa.field("path", pa.string()),
        ])),
    })
    pq.write_table(table, data_dir / "train-00000-of-00001.parquet", row_group_size=5)
    return data_dir


def test_manifest_and_index_are_split_safe_and_complete(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.tsv")
    data_dir = _write_shard(tmp_path / "data")
    records, summary = read_rxrx3_manifest(manifest)
    positions, index_summary = build_selected_well_index(
        data_dir, [row["well_id"] for row in records], tmp_path / "index"
    )
    assert summary["classes"] == 2
    assert summary["train_experiments"] == 1
    assert summary["ood_test_experiments"] == 1
    assert all(len(channels) == 6 for channels in positions.values())
    assert index_summary["cache_hit"] is False
    cached, cached_summary = build_selected_well_index(
        data_dir, [row["well_id"] for row in records], tmp_path / "index"
    )
    assert cached == positions
    assert cached_summary["cache_hit"] is True


def test_dataset_reads_six_channels_and_emits_raw_domain(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.tsv")
    data_dir = _write_shard(tmp_path / "data")
    records, summary = read_rxrx3_manifest(manifest)
    positions, _ = build_selected_well_index(
        data_dir, [row["well_id"] for row in records], tmp_path / "index"
    )
    train = [row for row in records if row["split"] == "train"]
    dataset = RxRx3CoreDataset(
        train, positions, data_dir, transform=None,
        environment_map=summary["environment_map"], site_map=summary["site_map"],
        cell_map=summary["cell_map"],
    )
    x, y, site, environment, cell = dataset[0]
    assert tuple(x.shape) == (6, 16, 16)
    assert (y, site, environment, cell) == (0, 0, summary["environment_map"]["ExpA"], 0)
    assert float(x[1, 0, 0]) > float(x[0, 0, 0])
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    sampled = list(ParquetLocalitySampler(dataset))
    assert sorted(sampled) == list(range(len(dataset)))


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_selected_index_rejects_missing_or_duplicate_channels(tmp_path, kind):
    rows = _rows()
    manifest = _write_manifest(tmp_path / "manifest.tsv", rows)
    selected = (rows[0]["well_id"], 3)
    data_dir = _write_shard(
        tmp_path / "data", rows,
        omit=selected if kind == "missing" else None,
        duplicate=selected if kind == "duplicate" else None,
    )
    records, _ = read_rxrx3_manifest(manifest)
    with pytest.raises(ValueError, match=kind):
        build_selected_well_index(
            data_dir, [row["well_id"] for row in records], tmp_path / "index"
        )


def test_corrupt_selected_payload_fails_at_decode(tmp_path):
    rows = _rows()
    manifest = _write_manifest(tmp_path / "manifest.tsv", rows)
    selected = (rows[0]["well_id"], 2)
    data_dir = _write_shard(tmp_path / "data", rows, corrupt=selected)
    records, summary = read_rxrx3_manifest(manifest)
    positions, _ = build_selected_well_index(
        data_dir, [row["well_id"] for row in records], tmp_path / "index"
    )
    dataset = RxRx3CoreDataset(
        [records[0]], positions, data_dir, transform=None,
        environment_map=summary["environment_map"], site_map=summary["site_map"],
        cell_map=summary["cell_map"],
    )
    with pytest.raises(ValueError, match="cannot decode"):
        dataset[0]


def test_dispatcher_maps_six_channels_to_five_and_reuses_id_validation(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.tsv")
    data_dir = _write_shard(tmp_path / "data")
    cfg = {
        "dataset": "rxrx3_core", "rxrx3_manifest": str(manifest),
        "rxrx3_data_dir": str(data_dir), "rxrx3_index_dir": str(tmp_path / "index"),
        "img_size": 12, "model": {"num_classes": 2},
        "train": {"batch_size": 2, "num_workers": 0,
                  "rxrx3_channel_layout": "cell_dino_native_cp5"},
    }
    train, id_val, ood_test, audit = make_loaders(cfg)
    validation = make_val_loader(cfg)
    assert audit is id_val and validation is id_val
    assert validation.selection_split_name == "id_val"
    assert cfg["sites"] == {"K": 1, "n_cell_types": 1}
    assert tuple(next(iter(train))[0].shape) == (2, 5, 12, 12)
    assert set(next(iter(ood_test))[2].tolist()) == {-1}


def test_manifest_rejects_split_leakage_and_incomplete_class_coverage(tmp_path):
    rows = _rows()
    rows[-1]["experiment_name"] = "ExpA"
    rows[-1]["well_id"] = "ExpA_3_A02"
    path = _write_manifest(tmp_path / "leak.tsv", rows)
    with pytest.raises(ValueError, match="overlap"):
        read_rxrx3_manifest(path)

    rows = _rows()
    rows.pop()
    path = _write_manifest(tmp_path / "coverage.tsv", rows)
    with pytest.raises(ValueError, match="complete fixed label set"):
        read_rxrx3_manifest(path)
