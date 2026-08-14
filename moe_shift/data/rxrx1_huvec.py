"""Split-safe six-channel RxRx1 HUVEC study utilities.

The WILDS package is used only as the authoritative metadata/index layer.  Pixels are read from
the official six-channel archive, and every custom split is made at experiment and well level.
The module deliberately keeps the two microscope sites from a well together.
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .rxrx1 import _native_channel_paths

EXPECTED_HUVEC_EXPERIMENTS = 24
EXPECTED_TREATMENTS = 1108


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field_index(fields, *names):
    normalized = {str(name).lower(): index for index, name in enumerate(fields)}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


_SITE_RE = re.compile(r"(?:^|[_-])s(?:ite)?[_-]?(\d+)$", re.IGNORECASE)


def _path_site(relative_path) -> int:
    stem = Path(str(relative_path)).stem
    match = _SITE_RE.search(stem)
    if not match:
        raise ValueError(f"cannot infer RxRx1 site from {relative_path!r}")
    return int(match.group(1))


def _path_well_key(relative_path) -> str:
    stem = Path(str(relative_path)).stem
    stem = re.sub(r"(?:^|[_-])s(?:ite)?[_-]?\d+$", "", stem, flags=re.IGNORECASE)
    return str(Path(str(relative_path)).parent / stem)


def build_huvec_manifest(data_root, raw_root, output_path, verify_paths=True):
    """Build and audit the native-six-channel HUVEC treatment manifest.

    Treatment labels are identified from the experimental design rather than an assumed numeric
    range: a treatment occurs in at most one well per experiment, whereas the 30 positive controls
    occur once on each of four plates.  The resulting set must contain exactly 1,108 labels.
    """
    from wilds import get_dataset

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_root = Path(raw_root).expanduser().resolve()
    dataset = get_dataset(dataset="rxrx1", root_dir=str(data_root), download=False)
    fields = list(dataset.metadata_fields)
    exp_col = _field_index(fields, "experiment")
    cell_col = _field_index(fields, "cell_type", "cell")
    plate_col = _field_index(fields, "plate")
    well_col = _field_index(fields, "well", "well_id")
    site_col = _field_index(fields, "site")
    if exp_col is None or cell_col is None:
        raise ValueError(f"RxRx1 metadata lacks experiment/cell_type fields: {fields}")

    metadata = np.asarray(dataset.metadata_array)
    labels = np.asarray(dataset.y_array).reshape(-1)
    paths = np.asarray(dataset._input_array).reshape(-1)
    cell_to_experiments = {}
    for cell in np.unique(metadata[:, cell_col]):
        mask = metadata[:, cell_col] == cell
        cell_to_experiments[int(cell)] = sorted(
            set(map(int, metadata[mask, exp_col].tolist())))
    huvec_candidates = [cell for cell, exps in cell_to_experiments.items()
                        if len(exps) == EXPECTED_HUVEC_EXPERIMENTS]
    if len(huvec_candidates) != 1:
        raise ValueError(
            "expected exactly one 24-experiment HUVEC cell code, got "
            f"{[(cell, len(exps)) for cell, exps in cell_to_experiments.items()]}")
    huvec_code = huvec_candidates[0]
    indices = np.flatnonzero(metadata[:, cell_col] == huvec_code)

    rows = []
    missing_channels = []
    for global_index in indices.tolist():
        meta = metadata[global_index]
        relative = str(paths[global_index])
        experiment = int(meta[exp_col])
        plate = str(meta[plate_col]) if plate_col is not None else Path(relative).parent.name
        well = str(meta[well_col]) if well_col is not None else _path_well_key(relative)
        site = int(meta[site_col]) if site_col is not None else _path_site(relative)
        well_id = f"{experiment}|{plate}|{well}"
        if verify_paths:
            channel_paths = _native_channel_paths(raw_root, relative)
            missing = [str(path) for path in channel_paths if not path.is_file()]
            if missing:
                missing_channels.extend(missing[:1])
        rows.append({
            "global_index": int(global_index), "cell_type": int(huvec_code),
            "experiment": experiment, "plate": plate, "well": well, "site": site,
            "well_id": well_id, "original_label": int(labels[global_index]),
            "relative_path": relative,
        })
    if missing_channels:
        raise FileNotFoundError(
            f"{len(missing_channels)} HUVEC sites have missing native channels; first: "
            f"{missing_channels[:5]}")

    frame = pd.DataFrame(rows)
    well_rows = frame.drop_duplicates(["well_id", "original_label"])
    per_label_experiment = (well_rows.groupby(["original_label", "experiment"])
                             .well_id.nunique().rename("wells").reset_index())
    label_design = per_label_experiment.groupby("original_label").wells.agg(
        n_experiments="size", min_wells="min", max_wells="max", total_wells="sum")
    treatment_labels = sorted(map(int, label_design.index[label_design.max_wells <= 1].tolist()))
    if len(treatment_labels) != EXPECTED_TREATMENTS:
        counts = label_design.groupby("max_wells").size().to_dict()
        raise ValueError(
            f"experimental-design filter found {len(treatment_labels)} treatment labels, "
            f"expected {EXPECTED_TREATMENTS}; max-wells histogram={counts}")
    label_map = {label: index for index, label in enumerate(treatment_labels)}
    frame = frame[frame.original_label.isin(treatment_labels)].copy()
    frame["label"] = frame.original_label.map(label_map).astype(np.int64)
    frame = frame.sort_values(
        ["experiment", "label", "plate", "well", "site", "global_index"]
    ).reset_index(drop=True)

    experiments = sorted(map(int, frame.experiment.unique().tolist()))
    site_counts = frame.groupby("well_id").site.nunique()
    if len(experiments) != EXPECTED_HUVEC_EXPERIMENTS:
        raise ValueError(f"HUVEC treatment manifest has {len(experiments)} experiments")
    if frame.label.nunique() != EXPECTED_TREATMENTS:
        raise ValueError("HUVEC manifest lost treatment classes")
    if int(site_counts.max()) > 2 or int(site_counts.min()) < 1:
        raise ValueError("a treatment well must contain one or two microscope sites")
    if frame.duplicated(["global_index"]).any():
        raise ValueError("duplicate WILDS indices in HUVEC manifest")

    frame.to_parquet(output_path, index=False)
    summary = {
        "schema_version": 1,
        "manifest": str(output_path),
        "manifest_sha256": _sha256(output_path),
        "huvec_cell_code": int(huvec_code),
        "n_experiments": len(experiments),
        "experiments": experiments,
        "n_classes": int(frame.label.nunique()),
        "n_wells": int(frame.well_id.nunique()),
        "n_sites": len(frame),
        "wells_with_one_site": int((site_counts == 1).sum()),
        "wells_with_two_sites": int((site_counts == 2).sum()),
        "original_label_map": {str(key): int(value) for key, value in label_map.items()},
        "metadata_fields": fields,
        "raw_root": str(raw_root),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return frame, summary


def deterministic_split(frame, source_experiments, target_experiments, split_key,
                        validation_occurrences=2):
    """Return a site manifest with train/iid_validation/target roles.

    Validation assignment is made per perturbation at the well level.  Exactly
    ``validation_occurrences`` source-experiment wells are held out whenever coverage permits.
    """
    source = set(map(int, source_experiments))
    target = set(map(int, target_experiments))
    if source & target:
        raise ValueError("source and target experiments overlap")
    available = set(map(int, frame.experiment.unique().tolist()))
    if not source or not target or not (source | target) <= available:
        raise ValueError("split names missing or unknown experiments")
    selected = frame[frame.experiment.isin(source | target)].copy()
    selected["role"] = "target"
    wells = selected[selected.experiment.isin(source)].drop_duplicates("well_id")
    validation_wells = set()
    for label, rows in wells.groupby("label", sort=True):
        candidates = []
        for row in rows.itertuples(index=False):
            token = f"{split_key}|{int(label)}|{int(row.experiment)}|{row.well_id}"
            candidates.append((hashlib.sha256(token.encode()).hexdigest(), row.well_id))
        candidates.sort()
        holdout_count = min(int(validation_occurrences), max(len(candidates) - 1, 0))
        if holdout_count < int(validation_occurrences):
            warnings.warn(
                f"{split_key}: label {label} has only {len(candidates)} source occurrences; "
                f"using {holdout_count} for IID validation and preserving one for training",
                RuntimeWarning,
                stacklevel=2,
            )
        validation_wells.update(well for _, well in candidates[:holdout_count])
    source_mask = selected.experiment.isin(source)
    selected.loc[source_mask, "role"] = "train"
    selected.loc[selected.well_id.isin(validation_wells), "role"] = "iid_validation"

    for role in ("train", "iid_validation", "target"):
        rows = selected[selected.role == role]
        if rows.empty:
            raise ValueError(f"{split_key}: {role} role is empty")
        observed = int(rows.label.nunique())
        if observed != EXPECTED_TREATMENTS:
            warnings.warn(
                f"{split_key}: {role} contains {observed} of {EXPECTED_TREATMENTS} labels; "
                "continuing with coverage recorded in the frozen registry",
                RuntimeWarning,
                stacklevel=2,
            )
    for well_id, roles in selected.groupby("well_id").role.nunique().items():
        if int(roles) != 1:
            raise ValueError(f"well {well_id} crosses split roles")
    return selected.sort_values(["role", "experiment", "label", "well_id", "site"])


def normalization_from_qc(site_frame, qc_frame):
    """Recover global channel moments from cached per-image means/stds."""
    joined = site_frame[["global_index"]].merge(qc_frame, on="global_index", validate="one_to_one")
    means, stds = [], []
    for channel in range(6):
        mean = joined[f"c{channel}_mean"].to_numpy(np.float64)
        std = joined[f"c{channel}_std"].to_numpy(np.float64)
        global_mean = float(mean.mean())
        global_var = float(np.maximum((std * std + mean * mean).mean() - global_mean ** 2, 1e-8))
        means.append(global_mean)
        stds.append(global_var ** 0.5)
    return means, stds


class Native6SiteDataset(Dataset):
    """Native six-channel site images with split-specific global normalization."""

    def __init__(self, frame, raw_root, img_size, mean, std, train=False):
        self.frame = frame.reset_index(drop=True)
        self.raw_root = Path(raw_root)
        self.img_size = int(img_size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(6, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(6, 1, 1)
        self.train = bool(train)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[int(index)]
        channels = []
        paths = _native_channel_paths(self.raw_root, row.relative_path)
        for path in paths:
            with Image.open(path) as image:
                channels.append(torch.from_numpy(
                    np.asarray(image, dtype=np.float32).copy()) / 255.0)
        x = torch.stack(channels)
        if self.train:
            angle = (0, 90, 180, 270)[int(torch.randint(0, 4, ()).item())]
            if angle:
                x = TF.rotate(x, angle)
            if bool(torch.rand(()) < 0.5):
                x = TF.hflip(x)
        x = TF.resize(x, [self.img_size, self.img_size], antialias=True)
        x = (x - self.mean) / self.std
        return {
            "image": x,
            "label": int(row.label),
            "experiment": int(row.experiment),
            "well_id": str(row.well_id),
            "site": int(row.site),
            "global_index": int(row.global_index),
        }
