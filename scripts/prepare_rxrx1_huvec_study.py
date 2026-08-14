#!/usr/bin/env python
"""Prepare, audit, visualize, and cheaply probe the RxRx1 HUVEC study.

The expensive extraction step is independently shardable.  Finalization fails closed unless all
shards cover the frozen manifest exactly once, then writes the immutable split registry consumed
by every raw-image run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.model import build_ccas
from moe_shift.data.rxrx1 import _native_channel_paths, _rxrx1_raw_transform
from moe_shift.data.rxrx1_huvec import (
    EXPECTED_TREATMENTS,
    build_huvec_manifest,
    deterministic_split,
    normalization_from_qc,
)
from moe_shift.utils.config import load_config

DEFAULT_RESULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/huvec_systematic_fast_20260814"
)
DEFAULT_CONFIG = "configs/ccas_rxrx1_cell_dino_native.yaml"


def stable_bucket(value, buckets):
    return int(hashlib.sha256(str(value).encode()).hexdigest(), 16) % int(buckets)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def build_manifest(result_root, config=DEFAULT_CONFIG, verify_paths=True):
    root = Path(result_root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config)
    frame, summary = build_huvec_manifest(
        cfg["data_root"], cfg["rxrx1_raw_root"], root / "data" / "huvec_sites.parquet",
        verify_paths=verify_paths,
    )
    summary["cell_dino_config"] = str(config)
    atomic_json(root / "data" / "huvec_sites.summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return frame, summary


def _qc(raw):
    small = F.interpolate(raw[None], size=(128, 128), mode="bilinear", align_corners=False)[0]
    values = {}
    flattened = small.flatten(1)
    for channel in range(6):
        row = flattened[channel]
        quantiles = torch.quantile(row, torch.tensor([0.1, 0.5, 0.9]))
        values.update({
            f"c{channel}_mean": float(row.mean()),
            f"c{channel}_std": float(row.std()),
            f"c{channel}_p10": float(quantiles[0]),
            f"c{channel}_p50": float(quantiles[1]),
            f"c{channel}_p90": float(quantiles[2]),
            f"c{channel}_zero": float((row <= 1.0 / 255.0).float().mean()),
            f"c{channel}_sat": float((row >= 254.0 / 255.0).float().mean()),
            f"c{channel}_focus": float(
                (small[channel, 1:] - small[channel, :-1]).abs().mean()
                + (small[channel, :, 1:] - small[channel, :, :-1]).abs().mean()),
        })
    centered = flattened - flattened.mean(1, keepdim=True)
    normalized = centered / centered.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
    corr = normalized @ normalized.T / normalized.shape[1]
    for left in range(6):
        for right in range(left + 1, 6):
            values[f"corr_{left}_{right}"] = float(corr[left, right])
    return values


class ExtractionDataset(Dataset):
    def __init__(self, frame, raw_root):
        self.frame = frame.reset_index(drop=True)
        self.raw_root = Path(raw_root)
        self.transform = _rxrx1_raw_transform(128, False, "cell_dino_native_cp5")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[int(index)]
        channels = []
        for path in _native_channel_paths(self.raw_root, row.relative_path):
            with Image.open(path) as image:
                channels.append(torch.from_numpy(
                    np.asarray(image, dtype=np.float32).copy()) / 255.0)
        raw = torch.stack(channels)
        return {
            "image": self.transform(raw), "qc": _qc(raw),
            "global_index": int(row.global_index), "well_id": str(row.well_id),
            "experiment": int(row.experiment), "label": int(row.label),
            "site": int(row.site),
        }


@torch.no_grad()
def extract_shard(result_root, shard_index, num_shards, config=DEFAULT_CONFIG,
                  batch_size=128, num_workers=8):
    root = Path(result_root)
    manifest_path = root / "data" / "huvec_sites.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"frozen HUVEC manifest missing: {manifest_path}")
    frame = pd.read_parquet(manifest_path)
    keep = frame.well_id.map(lambda value: stable_bucket(value, num_shards)) == int(shard_index)
    shard = frame[keep].copy()
    if shard.empty:
        raise ValueError(f"empty extraction shard {shard_index}/{num_shards}")
    cfg = load_config(config)
    model = build_ccas(cfg).cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loader = DataLoader(
        ExtractionDataset(shard, cfg["rxrx1_raw_root"]), batch_size=int(batch_size),
        shuffle=False, num_workers=int(num_workers), pin_memory=True,
        persistent_workers=(int(num_workers) > 0),
    )
    rows = []
    started = time.time()
    for batch_index, batch in enumerate(loader):
        features = model.forward_features(batch["image"].cuda(non_blocking=True)).float().cpu()
        batch_size_actual = len(features)
        for index in range(batch_size_actual):
            qc = {key: float(value[index]) for key, value in batch["qc"].items()}
            rows.append({
                "global_index": int(batch["global_index"][index]),
                "well_id": batch["well_id"][index],
                "experiment": int(batch["experiment"][index]),
                "label": int(batch["label"][index]),
                "site": int(batch["site"][index]),
                "embedding": features[index].numpy().astype(np.float32).tolist(),
                **qc,
            })
        if (batch_index + 1) % 25 == 0:
            print(f"[extract] shard={shard_index}/{num_shards} "
                  f"sites={len(rows)}/{len(shard)}", flush=True)
    output = root / "cache" / f"cell_dino_qc_shard{int(shard_index):02d}-of-{int(num_shards):02d}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    summary = {
        "shard_index": int(shard_index), "num_shards": int(num_shards),
        "n_sites": len(rows), "n_wells": int(pd.DataFrame(rows).well_id.nunique()),
        "embedding_dim": len(rows[0]["embedding"]), "elapsed_seconds": time.time() - started,
        "output": str(output),
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def _unit(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _matched_distance(experiments, labels, features, experiment_order):
    by_key = {(int(exp), int(label)): feature for exp, label, feature
              in zip(experiments, labels, features)}
    matrix = np.zeros((len(experiment_order), len(experiment_order)), dtype=np.float64)
    shared_counts = np.zeros_like(matrix, dtype=np.int64)
    for index, experiment in enumerate(experiment_order):
        shared_counts[index, index] = int(np.unique(labels[experiments == experiment]).size)
    for left_index, left in enumerate(experiment_order):
        for right_index in range(left_index + 1, len(experiment_order)):
            right = experiment_order[right_index]
            shared = sorted(set(labels[experiments == left]) & set(labels[experiments == right]))
            shared_counts[left_index, right_index] = shared_counts[right_index, left_index] = len(
                shared)
            if not shared:
                warnings.warn(
                    f"experiments {left} and {right} have no shared treatment labels; "
                    "their distance is undefined",
                    RuntimeWarning,
                    stacklevel=2,
                )
                matrix[left_index, right_index] = matrix[right_index, left_index] = np.nan
                continue
            a = _unit(np.stack([by_key[(left, label)] for label in shared]))
            b = _unit(np.stack([by_key[(right, label)] for label in shared]))
            value = float(np.median(1.0 - (a * b).sum(1)))
            matrix[left_index, right_index] = matrix[right_index, left_index] = value
    return matrix, shared_counts


def _target_difficulty(target, sources, experiments, labels, features):
    target_mask = experiments == int(target)
    target_labels = labels[target_mask]
    target_features = features[target_mask]
    rows = []
    for label, target_feature in zip(target_labels, target_features):
        source_mask = np.isin(experiments, list(sources)) & (labels == label)
        if source_mask.any():
            source_centroid = features[source_mask].mean(0)
            rows.append(1.0 - float(_unit(target_feature[None])[0] @ _unit(source_centroid[None])[0]))
    if not rows:
        raise ValueError(f"target {target} has no treatment labels shared with its source set")
    return float(np.median(rows)), len(rows)


def _role_label_coverage(assignment):
    coverage = {}
    expected = set(range(EXPECTED_TREATMENTS))
    for role in ("train", "iid_validation", "target"):
        observed = set(map(int, assignment.loc[assignment.role == role, "label"].unique()))
        coverage[role] = {
            "observed_labels": len(observed),
            "fraction": float(len(observed) / EXPECTED_TREATMENTS),
            "missing_labels": sorted(expected - observed),
        }
    coverage["target_by_experiment"] = {}
    for experiment, rows in assignment[assignment.role == "target"].groupby("experiment"):
        observed = set(map(int, rows.label.unique()))
        coverage["target_by_experiment"][str(int(experiment))] = {
            "observed_labels": len(observed),
            "fraction": float(len(observed) / EXPECTED_TREATMENTS),
            "missing_labels": sorted(expected - observed),
        }
    return coverage


def _folds(experiment_order, distance):
    # NaN on the diagonal excludes the trivial self-distance.
    centrality = np.nanmedian(np.where(np.eye(len(distance)), np.nan, distance), axis=1)
    ranked = [experiment_order[index] for index in np.argsort(centrality, kind="mergesort")]
    folds = [[], [], []]
    pattern = [0, 1, 2, 2, 1, 0]
    for index, experiment in enumerate(ranked):
        folds[pattern[index % len(pattern)]].append(int(experiment))
    if sorted(map(len, folds)) != [8, 8, 8]:
        raise ValueError(f"fold balancing failed: {list(map(len, folds))}")
    return folds, {str(exp): float(centrality[experiment_order.index(exp)]) for exp in experiment_order}


def _candidate_source_sets(target, experiment_order, distance, seed=20260814,
                           size=12, candidates=500):
    rng = np.random.default_rng(int(seed) + int(target))
    available = np.asarray([exp for exp in experiment_order if exp != target], dtype=np.int64)
    target_index = experiment_order.index(target)
    seen = set()
    rows = []
    while len(rows) < int(candidates):
        values = tuple(sorted(map(int, rng.choice(available, size=int(size), replace=False))))
        if values in seen:
            continue
        seen.add(values)
        score = float(np.median([distance[target_index, experiment_order.index(exp)]
                                 for exp in values]))
        rows.append((score, values))
    rows.sort(key=lambda item: (item[0], item[1]))
    selected = []
    for tier, quantile in (("low", 0.1), ("medium", 0.5), ("high", 0.9)):
        center = round(quantile * (len(rows) - 1))
        alternatives = []
        for offset in range(len(rows)):
            for index in (center - offset, center + offset):
                if not 0 <= index < len(rows):
                    continue
                candidate = rows[index]
                if all(len(set(candidate[1]) & set(old[1])) / size < 0.9
                       for old in alternatives):
                    alternatives.append(candidate)
                if len(alternatives) == 3:
                    break
            if len(alternatives) == 3:
                break
        for resample, (score, sources) in enumerate(alternatives):
            selected.append({
                "kind": "controlled", "target_experiments": [int(target)],
                "source_experiments": list(sources), "difficulty_tier": tier,
                "resample": int(resample), "selection_distance": score,
                "split_id": f"controlled_t{int(target)}_{tier}_r{resample}",
            })
    return selected


def _split_arrays(well_meta, features, spec):
    assignment = deterministic_split(
        well_meta, spec["source_experiments"], spec["target_experiments"], spec["split_id"])
    by_well = assignment.drop_duplicates("well_id")[["well_id", "role"]]
    metadata = well_meta.merge(by_well, on="well_id", validate="one_to_one")
    indices = {role: np.flatnonzero(metadata.role.to_numpy() == role)
               for role in ("train", "iid_validation", "target")}
    return metadata, indices


def _centroid_probe(metadata, indices, features):
    train = indices["train"]
    centroids = np.zeros((EXPECTED_TREATMENTS, features.shape[1]), dtype=np.float32)
    for label in range(EXPECTED_TREATMENTS):
        centroids[label] = features[train][metadata.label.to_numpy()[train] == label].mean(0)
    centroids = _unit(centroids)
    output = {}
    for role, selected in indices.items():
        prediction = (_unit(features[selected]) @ centroids.T).argmax(1)
        truth = metadata.label.to_numpy()[selected]
        output[role] = {"accuracy": float((prediction == truth).mean()), "n": len(selected)}
        if role == "target":
            output[role]["per_experiment"] = {
                str(int(exp)): float((prediction[metadata.experiment.to_numpy()[selected] == exp]
                                      == truth[metadata.experiment.to_numpy()[selected] == exp]).mean())
                for exp in sorted(set(map(int, metadata.experiment.to_numpy()[selected].tolist())))
            }
    return output


def _linear_probe(metadata, indices, features, device, epochs=25):
    label_values = metadata.label.to_numpy(np.int64)
    tensors = {role: (
        torch.from_numpy(features[index]).float(), torch.from_numpy(label_values[index]).long())
        for role, index in indices.items()
    }
    torch.manual_seed(20260814)
    model = torch.nn.Linear(features.shape[1], EXPECTED_TREATMENTS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    best, best_state, wait = -1.0, None, 0
    x_train, y_train = tensors["train"]
    for epoch in range(int(epochs)):
        model.train()
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), 1024):
            index = order[start:start + 1024]
            logits = model(x_train[index].to(device))
            loss = F.cross_entropy(logits, y_train[index].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            x_val, y_val = tensors["iid_validation"]
            value = float((model(x_val.to(device)).argmax(1).cpu() == y_val).float().mean())
        if value > best + 1e-5:
            best, best_state, wait = value, {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, 0
        else:
            wait += 1
        if epoch >= 7 and wait >= 5:
            break
    model.load_state_dict(best_state)
    model.eval()
    output = {}
    with torch.no_grad():
        for role, (x, y) in tensors.items():
            logits = model(x.to(device)).cpu()
            prediction = logits.argmax(1)
            output[role] = {
                "accuracy": float((prediction == y).float().mean()), "n": len(y),
                "loss": float(F.cross_entropy(logits, y)),
            }
            if role == "target":
                exp_values = metadata.experiment.to_numpy()[indices[role]]
                output[role]["per_experiment"] = {
                    str(int(exp)): float((prediction[exp_values == exp] == y[exp_values == exp])
                                         .float().mean())
                    for exp in sorted(set(map(int, exp_values.tolist())))
                }
    return output


def _plot_preparation(root, experiment_order, cell_distance, qc_distance, well_meta,
                      features, probe_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    figures = root / "analysis" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for name, matrix in (("cell_dino_distance", cell_distance), ("raw_qc_distance", qc_distance)):
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(matrix, cmap="magma")
        ax.set_xticks(range(len(experiment_order)), experiment_order, rotation=90, fontsize=6)
        ax.set_yticks(range(len(experiment_order)), experiment_order, fontsize=6)
        ax.set_title(name.replace("_", " ").title())
        fig.colorbar(image, ax=ax, shrink=0.75)
        fig.tight_layout(); fig.savefig(figures / f"{name}.png", dpi=180); plt.close(fig)

    rng = np.random.default_rng(20260814)
    take = rng.choice(len(features), size=min(12000, len(features)), replace=False)
    coordinates = PCA(n_components=2, random_state=20260814).fit_transform(features[take])
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1],
                         c=well_meta.experiment.to_numpy()[take], s=4, alpha=0.5, cmap="tab20")
    ax.set_title("Frozen Cell-DINO well embeddings by experiment")
    fig.colorbar(scatter, ax=ax, label="experiment code")
    fig.tight_layout(); fig.savefig(figures / "cell_dino_pca_experiment.png", dpi=180); plt.close(fig)

    rows = pd.DataFrame(probe_rows)
    if not rows.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, group in rows.groupby("method"):
            ax.scatter(group.cell_dino_difficulty, group.target_accuracy, label=method, alpha=0.75)
        ax.set_xlabel("Cell-DINO matched target difficulty")
        ax.set_ylabel("Well-level perturbation accuracy")
        ax.legend(); fig.tight_layout()
        fig.savefig(figures / "probe_accuracy_vs_difficulty.png", dpi=180); plt.close(fig)


def finalize(result_root, num_shards=6, device="cuda"):
    root = Path(result_root)
    site_manifest = pd.read_parquet(root / "data" / "huvec_sites.parquet")
    paths = [root / "cache" / f"cell_dino_qc_shard{i:02d}-of-{int(num_shards):02d}.parquet"
             for i in range(int(num_shards))]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing extraction shards: {missing}")
    extracted = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if extracted.global_index.duplicated().any():
        raise ValueError("duplicate site indices across extraction shards")
    if set(map(int, extracted.global_index)) != set(map(int, site_manifest.global_index)):
        raise ValueError("extraction shards do not exactly cover the frozen site manifest")
    extracted = extracted.sort_values("global_index").reset_index(drop=True)
    qc_columns = [column for column in extracted if column.startswith(("c", "corr_"))]
    site_qc = extracted[["global_index", *qc_columns]].copy()
    site_qc.to_parquet(root / "cache" / "site_qc.parquet", index=False)

    embedding = np.stack(extracted.embedding.map(np.asarray).to_list()).astype(np.float32)
    embedding = _unit(embedding)
    extracted_meta = extracted[["global_index", "well_id", "experiment", "label", "site"]]
    well_rows, well_embeddings, well_qc = [], [], []
    qc_values = extracted[qc_columns].to_numpy(np.float32)
    for well_id, indices in extracted_meta.groupby("well_id", sort=True).groups.items():
        index = np.asarray(list(indices), dtype=np.int64)
        first = extracted_meta.iloc[index[0]]
        well_rows.append({"well_id": well_id, "experiment": int(first.experiment),
                          "label": int(first.label), "n_sites": len(index)})
        well_embeddings.append(embedding[index].mean(0))
        well_qc.append(qc_values[index].mean(0))
    well_meta = pd.DataFrame(well_rows)
    well_embeddings = _unit(np.stack(well_embeddings))
    well_qc = np.stack(well_qc)
    well_meta.to_parquet(root / "cache" / "well_metadata.parquet", index=False)
    np.save(root / "cache" / "well_cell_dino.npy", well_embeddings.astype(np.float32))
    np.save(root / "cache" / "well_qc.npy", well_qc.astype(np.float32))

    experiments = well_meta.experiment.to_numpy(np.int64)
    labels = well_meta.label.to_numpy(np.int64)
    experiment_order = sorted(map(int, np.unique(experiments).tolist()))
    cell_distance, shared_label_counts = _matched_distance(
        experiments, labels, well_embeddings, experiment_order)
    standardized_qc = (well_qc - np.median(well_qc, axis=0)) / np.maximum(
        np.quantile(well_qc, 0.75, axis=0) - np.quantile(well_qc, 0.25, axis=0), 1e-6)
    qc_distance, qc_shared_label_counts = _matched_distance(
        experiments, labels, standardized_qc, experiment_order)
    if not np.array_equal(shared_label_counts, qc_shared_label_counts):
        raise RuntimeError("Cell-DINO and QC experiment-overlap audits disagree")
    folds, centrality = _folds(experiment_order, cell_distance)
    primary = [{
        "kind": "primary", "fold": index, "split_id": f"primary_fold{index}",
        "target_experiments": sorted(targets),
        "source_experiments": sorted(set(experiment_order) - set(targets)),
        "difficulty_tier": "natural", "resample": 0,
    } for index, targets in enumerate(folds)]
    ordered = sorted(experiment_order, key=lambda value: (centrality[str(value)], value))
    anchors = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    controlled = []
    for target in anchors:
        controlled.extend(_candidate_source_sets(target, experiment_order, cell_distance))
    all_specs = primary + controlled
    main_specs = primary + [spec for spec in controlled if spec["resample"] == 0]

    difficulty_rows = []
    for spec in all_specs:
        spec["target_difficulty"] = {}
        spec["raw_qc_target_difficulty"] = {}
        spec["target_label_coverage"] = {}
        for target in spec["target_experiments"]:
            cell_value, cell_count = _target_difficulty(
                target, spec["source_experiments"], experiments, labels, well_embeddings)
            qc_value, qc_count = _target_difficulty(
                target, spec["source_experiments"], experiments, labels, standardized_qc)
            if cell_count != qc_count:
                raise ValueError(f"target {target} Cell-DINO/QC label coverage disagrees")
            observed = int(np.unique(labels[experiments == int(target)]).size)
            spec["target_difficulty"][str(target)] = cell_value
            spec["raw_qc_target_difficulty"][str(target)] = qc_value
            spec["target_label_coverage"][str(target)] = {
                "observed_labels": observed,
                "source_matched_labels": int(cell_count),
                "fraction": float(cell_count / EXPECTED_TREATMENTS),
            }
        assignment = deterministic_split(
            site_manifest, spec["source_experiments"], spec["target_experiments"], spec["split_id"])
        spec["role_label_coverage"] = _role_label_coverage(assignment)
        means, stds = normalization_from_qc(
            assignment[assignment.role == "train"], site_qc)
        spec["normalization"] = {"mean": means, "std": stds}
        difficulty_rows.extend({
            "split_id": spec["split_id"], "target_experiment": int(target),
            "cell_dino_difficulty": spec["target_difficulty"][str(target)],
            "raw_qc_difficulty": spec["raw_qc_target_difficulty"][str(target)],
            "observed_target_labels": spec["target_label_coverage"][str(target)][
                "observed_labels"],
            "source_matched_labels": spec["target_label_coverage"][str(target)][
                "source_matched_labels"],
            "target_label_fraction": spec["target_label_coverage"][str(target)]["fraction"],
        } for target in spec["target_experiments"])

    probe_rows = []
    probe_dir = root / "probes"; probe_dir.mkdir(parents=True, exist_ok=True)
    for spec in all_specs:
        metadata, indices = _split_arrays(well_meta, well_embeddings, spec)
        for method, result in (
            ("centroid", _centroid_probe(metadata, indices, well_embeddings)),
            ("linear", _linear_probe(metadata, indices, well_embeddings, device)),
        ):
            atomic_json(probe_dir / f"{spec['split_id']}_{method}.json", result)
            for target in spec["target_experiments"]:
                probe_rows.append({
                    "split_id": spec["split_id"], "kind": spec["kind"],
                    "difficulty_tier": spec["difficulty_tier"], "resample": spec["resample"],
                    "target_experiment": int(target), "method": method,
                    "train_accuracy": result["train"]["accuracy"],
                    "iid_accuracy": result["iid_validation"]["accuracy"],
                    "target_accuracy": result["target"]["per_experiment"][str(target)],
                    "cell_dino_difficulty": spec["target_difficulty"][str(target)],
                    "raw_qc_difficulty": spec["raw_qc_target_difficulty"][str(target)],
                    "target_label_fraction": spec["target_label_coverage"][str(target)][
                        "fraction"],
                })
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(probe_rows).to_csv(root / "analysis" / "probe_results.csv", index=False)
    pd.DataFrame(difficulty_rows).to_csv(root / "analysis" / "target_difficulty.csv", index=False)
    np.save(root / "analysis" / "cell_dino_experiment_distance.npy", cell_distance)
    np.save(root / "analysis" / "raw_qc_experiment_distance.npy", qc_distance)
    np.save(root / "analysis" / "experiment_shared_label_counts.npy", shared_label_counts)
    registry = {
        "schema_version": 1, "study": "rxrx1_huvec_systematic_fast",
        "site_manifest": str(root / "data" / "huvec_sites.parquet"),
        "site_qc": str(root / "cache" / "site_qc.parquet"),
        "raw_root": str(load_config(DEFAULT_CONFIG)["rxrx1_raw_root"]),
        "experiments": experiment_order, "folds": folds, "centrality": centrality,
        "controlled_anchor_experiments": anchors,
        "primary_splits": primary, "controlled_splits": controlled,
        "main_training_splits": main_specs,
        "training_unit": "site", "evaluation_unit": "well",
        "target_class_policy": {
            "description": "score each target experiment on its observed treatment wells",
            "denominator": EXPECTED_TREATMENTS,
            "hard_minimum": None,
            "coverage_handling": "record missing labels; do not stop a valid nonempty split",
        },
        "target_is_excluded_from": [
            "normalization", "training", "iid_validation", "checkpoint_selection",
            "masked_autoencoder_pretraining"],
    }
    atomic_json(root / "study_registry.json", registry)
    _plot_preparation(root, experiment_order, cell_distance, qc_distance,
                      well_meta, well_embeddings, probe_rows)
    marker = {"completed_at": time.time(), "registry": str(root / "study_registry.json"),
              "n_probe_rows": len(probe_rows)}
    atomic_json(root / "PREPARED.json", marker)
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)
    return registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--skip-path-verification", action="store_true")
    parser.add_argument("--extract-shard", type=int)
    parser.add_argument("--num-extraction-shards", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    selected = sum((args.build_manifest, args.extract_shard is not None, args.finalize))
    if selected != 1:
        parser.error("select exactly one of --build-manifest, --extract-shard, or --finalize")
    if args.build_manifest:
        build_manifest(args.result_root, args.config, not args.skip_path_verification)
    elif args.extract_shard is not None:
        extract_shard(args.result_root, args.extract_shard, args.num_extraction_shards,
                      args.config, args.batch_size, args.num_workers)
    else:
        finalize(args.result_root, args.num_extraction_shards)


if __name__ == "__main__":
    main()
