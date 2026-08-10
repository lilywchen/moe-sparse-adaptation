"""Parquet-backed six-channel loader for the frozen RxRx3-core task.

The dataset stores one JPEG-2000 payload per channel row.  This module scans only
the lightweight ``__key__`` column to build a manifest-scoped random-access
index, then reads the row groups needed by each selected well on demand.  Image
bytes are never materialized globally.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from .rxrx1 import _rxrx1_raw_transform


KEY_PATTERN = re.compile(
    r"^(?P<experiment>[^/]+)/Plate(?P<plate>[0-9]+)/"
    r"(?P<address>[A-Z]{1,2}[0-9]{2})_s1_(?P<channel>[1-6])$"
)
REQUIRED_MANIFEST_COLUMNS = {
    "split", "label", "gene", "experiment_name", "plate", "address",
    "well_id", "guide", "cell_type",
}
SPLITS = ("train", "id_val", "ood_test")
_CONTEXT_CACHE = {}
_LOADER_CACHE = {}


def _sha256_values(values):
    digest = hashlib.sha256()
    for value in sorted(map(str, values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _well_and_channel(key):
    match = KEY_PATTERN.fullmatch(str(key))
    if match is None:
        return None, None
    values = match.groupdict()
    return (
        f"{values['experiment']}_{int(values['plate'])}_{values['address']}",
        int(values["channel"]),
    )


def read_rxrx3_manifest(path):
    """Read and fail-closed audit one frozen RxRx3 manifest."""
    path = Path(path)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} lacks RxRx3 manifest columns: {sorted(missing)}")
        records = [dict(row) for row in reader]
    if not records:
        raise ValueError(f"{path} is empty")

    wells = [row["well_id"] for row in records]
    if len(wells) != len(set(wells)):
        raise ValueError("RxRx3 manifest contains duplicate well_id rows")
    observed_splits = {row["split"] for row in records}
    if observed_splits != set(SPLITS):
        raise ValueError(
            f"RxRx3 manifest splits must be exactly {SPLITS}, got {sorted(observed_splits)}")

    label_to_gene, gene_to_label = {}, {}
    for row in records:
        try:
            label = int(row["label"])
            plate = int(row["plate"])
        except ValueError as error:
            raise ValueError("RxRx3 labels and plates must be integers") from error
        if label < 0 or plate < 1:
            raise ValueError("RxRx3 labels must be non-negative and plates positive")
        gene = row["gene"]
        if label in label_to_gene and label_to_gene[label] != gene:
            raise ValueError(f"label {label} maps to multiple genes")
        if gene in gene_to_label and gene_to_label[gene] != label:
            raise ValueError(f"gene {gene} maps to multiple labels")
        label_to_gene[label], gene_to_label[gene] = gene, label
        row["label"] = label
        row["plate"] = plate

    labels = sorted(label_to_gene)
    if labels != list(range(len(labels))):
        raise ValueError("RxRx3 labels must be contiguous 0..C-1")
    all_labels = set(labels)
    per_split_labels = {
        split: {row["label"] for row in records if row["split"] == split}
        for split in SPLITS
    }
    if any(values != all_labels for values in per_split_labels.values()):
        raise ValueError("every RxRx3 split must retain the complete fixed label set")

    train_experiments = sorted({
        row["experiment_name"] for row in records if row["split"] == "train"
    })
    id_experiments = sorted({
        row["experiment_name"] for row in records if row["split"] == "id_val"
    })
    test_experiments = sorted({
        row["experiment_name"] for row in records if row["split"] == "ood_test"
    })
    if set(train_experiments) != set(id_experiments):
        raise ValueError("RxRx3 ID validation must cover exactly the train experiments")
    if set(train_experiments) & set(test_experiments):
        raise ValueError("RxRx3 train and OOD-test experiments overlap")

    experiment_names = sorted(set(train_experiments) | set(test_experiments))
    environment_map = {name: index for index, name in enumerate(experiment_names)}
    site_map = {name: index for index, name in enumerate(train_experiments)}
    cell_types = sorted({row["cell_type"] for row in records})
    cell_map = {name: index for index, name in enumerate(cell_types)}
    split_counts = {split: sum(row["split"] == split for row in records) for split in SPLITS}
    summary = {
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(records),
        "unique_wells": len(wells),
        "classes": len(labels),
        "split_counts": split_counts,
        "train_experiments": len(train_experiments),
        "ood_test_experiments": len(test_experiments),
        "cell_types": cell_types,
        "well_set_sha256": _sha256_values(wells),
        "environment_map": environment_map,
        "site_map": site_map,
        "cell_map": cell_map,
    }
    return records, summary


def _shard_fingerprint(shards):
    return [{"name": path.name, "bytes": path.stat().st_size} for path in shards]


def _load_index_cache(path, selected_hash, fingerprint):
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") != 1
        or payload.get("selected_wells_sha256") != selected_hash
        or payload.get("shards") != fingerprint
    ):
        return None
    return payload.get("positions")


def build_selected_well_index(data_dir, well_ids, cache_dir=None, lock_timeout=1800):
    """Resolve every selected well to six ``(shard,row_group,row_offset)`` triples.

    A gzip cache under the persistent dataset directory prevents eight simultaneous
    training jobs from rescanning the key columns.  The cache is a derived index only;
    it contains no image bytes or labels.
    """
    import pyarrow.parquet as pq

    data_dir = Path(data_dir)
    selected = set(map(str, well_ids))
    if not selected:
        raise ValueError("RxRx3 index requires at least one selected well")
    shards = sorted(data_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no RxRx3 Parquet shards found under {data_dir}")
    fingerprint = _shard_fingerprint(shards)
    selected_hash = _sha256_values(selected)
    cache_root = Path(cache_dir) if cache_dir else data_dir.parent / "index_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"selected_wells_{selected_hash[:20]}.json.gz"
    cached = _load_index_cache(cache_path, selected_hash, fingerprint)
    if cached is not None:
        return cached, {"cache": str(cache_path), "cache_hit": True, "shards": len(shards)}

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_fd = None
    started = time.time()
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            cached = _load_index_cache(cache_path, selected_hash, fingerprint)
            if cached is not None:
                return cached, {
                    "cache": str(cache_path), "cache_hit": True, "shards": len(shards)
                }
            if time.time() - started > lock_timeout:
                raise TimeoutError(f"timed out waiting for RxRx3 index lock {lock_path}")
            time.sleep(1.0)

    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("utf-8"))
        positions = {well: [None] * 6 for well in selected}
        duplicate = []
        for shard_index, shard in enumerate(shards):
            parquet = pq.ParquetFile(shard)
            if "__key__" not in parquet.schema_arrow.names or "jp2" not in parquet.schema_arrow.names:
                raise ValueError(f"{shard} does not have __key__ and jp2 columns")
            for row_group in range(parquet.metadata.num_row_groups):
                keys = parquet.read_row_group(row_group, columns=["__key__"]).column(0)
                for row_offset, key in enumerate(keys.to_pylist()):
                    well_id, channel = _well_and_channel(key)
                    if well_id not in positions:
                        continue
                    slot = channel - 1
                    if positions[well_id][slot] is not None:
                        if len(duplicate) < 10:
                            duplicate.append((well_id, channel))
                    positions[well_id][slot] = [shard_index, row_group, row_offset]
        missing = {
            well: [index + 1 for index, value in enumerate(channels) if value is None]
            for well, channels in positions.items()
            if any(value is None for value in channels)
        }
        if duplicate:
            raise ValueError(f"duplicate RxRx3 selected channel rows: {duplicate[:10]}")
        if missing:
            first = list(sorted(missing.items()))[:10]
            raise ValueError(f"missing RxRx3 selected channels for {len(missing)} wells: {first}")
        payload = {
            "schema_version": 1,
            "selected_wells_sha256": selected_hash,
            "shards": fingerprint,
            "positions": positions,
        }
        tmp = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(tmp, cache_path)
        return positions, {"cache": str(cache_path), "cache_hit": False, "shards": len(shards)}
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


class RxRx3CoreDataset(Dataset):
    """A split view over manifest rows with lazy Parquet/JPEG-2000 reads."""

    def __init__(self, records, positions, data_dir, transform, environment_map, site_map, cell_map):
        self.records = list(records)
        self.positions = positions
        self.shards = sorted(Path(data_dir).glob("*.parquet"))
        self.transform = transform
        self.environment_map = dict(environment_map)
        self.site_map = dict(site_map)
        self.cell_map = dict(cell_map)
        self.environment_ids = np.asarray([
            self.environment_map[row["experiment_name"]] for row in self.records
        ], dtype=np.int64)
        self.targets = [int(row["label"]) for row in self.records]
        self._parquet_files = {}
        self._row_groups = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_parquet_files"] = {}
        state["_row_groups"] = {}
        return state

    def __len__(self):
        return len(self.records)

    def locality_key(self, index):
        """Shard/row-group of channel 1, used only to order training I/O."""
        location = self.positions[self.records[index]["well_id"]][0]
        return int(location[0]), int(location[1])

    def _parquet(self, shard_index):
        import pyarrow.parquet as pq
        if shard_index not in self._parquet_files:
            self._parquet_files[shard_index] = pq.ParquetFile(self.shards[shard_index])
        return self._parquet_files[shard_index]

    def _row_group(self, shard_index, row_group):
        key = (shard_index, row_group)
        if key not in self._row_groups:
            # Bound per-worker decoded Arrow memory while preserving the common case in which
            # all six consecutive channels share one or two row groups.
            if len(self._row_groups) >= 8:
                self._row_groups.pop(next(iter(self._row_groups)))
            self._row_groups[key] = self._parquet(shard_index).read_row_group(
                row_group, columns=["jp2"]
            ).column(0)
        return self._row_groups[key]

    @staticmethod
    def _decode(payload, well_id, channel):
        value = payload.as_py() if hasattr(payload, "as_py") else payload
        raw = value.get("bytes") if isinstance(value, dict) else value
        if not isinstance(raw, (bytes, bytearray, memoryview)) or not raw:
            raise ValueError(f"RxRx3 {well_id} channel {channel} has no image bytes")
        try:
            with Image.open(io.BytesIO(bytes(raw))) as image:
                image.load()
                array = np.asarray(image, dtype=np.float32).copy()
        except Exception as error:
            raise ValueError(
                f"cannot decode RxRx3 {well_id} channel {channel}: {error}") from error
        if array.ndim != 2:
            raise ValueError(
                f"RxRx3 {well_id} channel {channel} is not grayscale: {array.shape}")
        return torch.from_numpy(array) / 255.0

    def __getitem__(self, index):
        row = self.records[index]
        well_id = row["well_id"]
        locations = self.positions.get(well_id)
        if locations is None or len(locations) != 6:
            raise KeyError(f"RxRx3 index has no complete entry for {well_id}")
        channels = []
        for channel, (shard, row_group, row_offset) in enumerate(locations, start=1):
            payload = self._row_group(int(shard), int(row_group))[int(row_offset)]
            channels.append(self._decode(payload, well_id, channel))
        x = torch.stack(channels, dim=0)
        if self.transform is not None:
            x = self.transform(x)
        raw_environment = self.environment_map[row["experiment_name"]]
        site = self.site_map.get(row["experiment_name"], -1)
        cell = self.cell_map[row["cell_type"]]
        return x, int(row["label"]), int(site), int(raw_environment), int(cell)


class ParquetLocalitySampler(Sampler):
    """Shuffle row groups and rows while keeping nearby wells adjacent.

    RxRx3 row groups contain 100 channel payloads (roughly 16 wells).  Global
    random sampling would reread a ~1.3 MB row group for nearly every well.
    Shuffling the groups and their internal order preserves stochastic training
    while allowing each worker's bounded row-group cache to serve all nearby
    wells in a batch.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        groups = {}
        for index in range(len(dataset)):
            groups.setdefault(dataset.locality_key(index), []).append(index)
        self.groups = list(groups.values())

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        if not self.groups:
            return iter(())
        order = torch.randperm(len(self.groups)).tolist()
        indices = []
        for group_index in order:
            group = self.groups[group_index]
            inner = torch.randperm(len(group)).tolist()
            indices.extend(group[position] for position in inner)
        return iter(indices)


def _context_key(cfg):
    manifest = Path(cfg["rxrx3_manifest"]).resolve()
    data_dir = Path(cfg["rxrx3_data_dir"]).resolve()
    cache_dir = Path(cfg.get("rxrx3_index_dir", data_dir.parent / "index_cache")).resolve()
    img_size = int(cfg.get("img_size") or cfg["model"].get("vit", {}).get("img", 256))
    return str(manifest), str(data_dir), str(cache_dir), img_size


def _make_context(cfg):
    key = _context_key(cfg)
    if key in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[key]
    manifest, data_dir, cache_dir, img_size = key
    layout = cfg["train"].get("rxrx3_channel_layout", "cell_dino_native_cp5")
    if layout != "cell_dino_native_cp5":
        raise ValueError("RxRx3 Cell-DINO runs require cell_dino_native_cp5 channel layout")
    records, summary = read_rxrx3_manifest(manifest)
    expected_classes = int(cfg["model"]["num_classes"])
    if expected_classes != summary["classes"]:
        raise ValueError(
            f"model.num_classes={expected_classes} but RxRx3 manifest has {summary['classes']}")
    positions, index_summary = build_selected_well_index(
        data_dir, [row["well_id"] for row in records], cache_dir=cache_dir
    )
    transforms = {
        "train": _rxrx1_raw_transform(img_size, True, layout),
        "eval": _rxrx1_raw_transform(img_size, False, layout),
    }
    common = (
        positions, data_dir, summary["environment_map"], summary["site_map"], summary["cell_map"]
    )
    datasets = {}
    for split in SPLITS:
        split_records = [row for row in records if row["split"] == split]
        datasets[split] = RxRx3CoreDataset(
            split_records, common[0], common[1],
            transforms["train" if split == "train" else "eval"],
            common[2], common[3], common[4],
        )
    context = {"datasets": datasets, "summary": {**summary, "index": index_summary}}
    _CONTEXT_CACHE[key] = context
    return context


def _loader_key(cfg):
    return _context_key(cfg) + (
        int(cfg["train"]["batch_size"]), int(cfg["train"]["num_workers"]),
    )


def _make_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        sampler=(ParquetLocalitySampler(dataset) if shuffle else None), num_workers=num_workers,
        pin_memory=True, drop_last=shuffle, persistent_workers=(num_workers > 0),
    )


def _loader_bundle(cfg):
    key = _loader_key(cfg)
    if key in _LOADER_CACHE:
        return _LOADER_CACHE[key]
    context = _make_context(cfg)
    summary = context["summary"]
    cfg.setdefault("sites", {})["K"] = summary["train_experiments"]
    cfg["sites"]["n_cell_types"] = len(summary["cell_types"])
    batch_size, workers = key[-2:]
    train = _make_loader(context["datasets"]["train"], batch_size, workers, True)
    id_val = _make_loader(context["datasets"]["id_val"], batch_size, workers, False)
    ood_test = _make_loader(context["datasets"]["ood_test"], batch_size, workers, False)
    id_val.selection_split_name = "id_val"
    print(
        f"[rxrx3_core] {summary['classes']} classes; "
        f"{summary['train_experiments']} train / {summary['ood_test_experiments']} OOD experiments; "
        f"|train|={len(train.dataset)} |id_val|={len(id_val.dataset)} "
        f"|ood_test|={len(ood_test.dataset)}; index_cache_hit={summary['index']['cache_hit']}"
    )
    _LOADER_CACHE[key] = train, id_val, ood_test
    return _LOADER_CACHE[key]


def make_rxrx3_core_loaders(cfg):
    """Return train, ID validation, OOD test, and audit loaders."""
    train, id_val, ood_test = _loader_bundle(cfg)
    return train, id_val, ood_test, id_val


def make_rxrx3_core_val_loader(cfg):
    """Return the frozen ID-validation plates used for checkpoint diagnostics."""
    return _loader_bundle(cfg)[1]
