#!/usr/bin/env python
"""Frozen-encoder RxRx1 evaluation for standalone HUVEC MAE runs.

The encoder checkpoint was selected only by reconstruction validation.  This script introduces
perturbation and batch labels afterward, freezes the encoder, and reports prototype retrieval,
a closed-form ridge linear probe, and source-batch separability.  Target labels never tune a
hyperparameter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.data.rxrx1_huvec import (
    EXPECTED_TREATMENTS,
    Native6SiteDataset,
    deterministic_split,
)
from moe_shift.models.huvec import build_study_model


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _unit(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


@torch.no_grad()
def _extract(encoder, frame, raw_root, normalization, normalization_mode,
             image_size, batch_size, workers, device):
    dataset = Native6SiteDataset(
        frame, raw_root, image_size, normalization["mean"], normalization["std"],
        train=False, normalization_mode=normalization_mode)
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(workers),
        pin_memory=True, persistent_workers=int(workers) > 0)
    features, labels, experiments, wells, sites = [], [], [], [], []
    encoder.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            output = encoder.forward_features(images)
        features.append(output.float().cpu().numpy())
        labels.extend(map(int, batch["label"])); experiments.extend(map(int, batch["experiment"]))
        wells.extend(map(str, batch["well_id"])); sites.extend(map(int, batch["site"]))
        if batch_index == 0 or (batch_index + 1) % 25 == 0:
            print(f"[embed] sites={sum(len(value) for value in features)}/{len(dataset)}", flush=True)
    metadata = pd.DataFrame({
        "well_id": wells, "site": sites, "label": labels, "experiment": experiments})
    values = _unit(np.concatenate(features))
    rows, well_values = [], []
    for well_id, indices in metadata.groupby("well_id", sort=True).groups.items():
        index = np.asarray(list(indices), dtype=np.int64)
        first = metadata.iloc[index[0]]
        if metadata.iloc[index].label.nunique() != 1 or metadata.iloc[index].experiment.nunique() != 1:
            raise ValueError(f"inconsistent site metadata in well {well_id}")
        rows.append({
            "well_id": well_id, "label": int(first.label),
            "experiment": int(first.experiment), "n_sites": len(index)})
        well_values.append(values[index].mean(0))
    return pd.DataFrame(rows), _unit(np.stack(well_values))


def _score_matrix(scores, labels):
    labels = np.asarray(labels, dtype=np.int64)
    truth = scores[np.arange(len(labels)), labels]
    ranks = 1 + (scores > truth[:, None]).sum(1)
    prediction = scores.argmax(1)
    return {
        "n": len(labels), "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()), "mean_rank": float(ranks.mean()),
        "mean_reciprocal_rank": float((1.0 / ranks).mean()),
    }, prediction, ranks


def _per_experiment_scores(scores, metadata):
    output = {}
    experiments = metadata.experiment.to_numpy(np.int64)
    labels = metadata.label.to_numpy(np.int64)
    for experiment in sorted(map(int, np.unique(experiments))):
        selected = experiments == experiment
        output[str(experiment)] = _score_matrix(scores[selected], labels[selected])[0]
    return output


def _prototype_scores(train_features, train_labels, query_features):
    prototypes = np.zeros((EXPECTED_TREATMENTS, train_features.shape[1]), dtype=np.float32)
    counts = np.zeros(EXPECTED_TREATMENTS, dtype=np.int64)
    for label in range(EXPECTED_TREATMENTS):
        selected = train_features[np.asarray(train_labels) == label]
        if not len(selected):
            raise ValueError(f"prototype training lacks perturbation {label}")
        prototypes[label] = selected.mean(0); counts[label] = len(selected)
    prototypes = _unit(prototypes)
    return query_features @ prototypes.T, counts


def _ridge_fit(features, labels, classes, penalty):
    x = np.asarray(features, dtype=np.float64)
    x = np.concatenate((x, np.ones((len(x), 1), dtype=np.float64)), axis=1)
    gram = x.T @ x
    regularizer = np.eye(x.shape[1], dtype=np.float64) * float(penalty)
    regularizer[-1, -1] = 0.0
    target = np.zeros((x.shape[1], int(classes)), dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    for label in range(int(classes)):
        target[:, label] = x[labels == label].sum(0)
    return np.linalg.solve(gram + regularizer, target).astype(np.float32)


def _ridge_scores(features, weights):
    x = np.asarray(features, dtype=np.float32)
    x = np.concatenate((x, np.ones((len(x), 1), dtype=np.float32)), axis=1)
    return x @ weights


def evaluate(args):
    started = time.time()
    output = Path(args.output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    registry = json.loads(Path(args.registry).read_text())
    if args.random_init:
        split_specs = {row["split_id"]: row for row in registry["main_training_splits"]}
        if args.split_id not in split_specs:
            raise KeyError(f"unknown frozen split {args.split_id!r}")
        torch.manual_seed(int(args.seed)); np.random.seed(int(args.seed))
        checkpoint_path = None
        config = {
            "model": args.model, "image_size": int(args.image_size),
            "split_id": args.split_id, "seed": int(args.seed),
            "pretraining": "none_random_initialization",
        }
        pretrain_audit = {
            "normalization": split_specs[args.split_id]["normalization"],
            "target_images_used_for_initialization": False,
        }
    else:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        checkpoint = _torch_load(checkpoint_path)
        config = checkpoint["config"]; pretrain_audit = checkpoint["audit"]
    split_specs = {row["split_id"]: row for row in registry["main_training_splits"]}
    split_spec = split_specs[config["split_id"]]
    sites = pd.read_parquet(args.site_manifest)
    assignment = deterministic_split(
        sites, split_spec["source_experiments"], split_spec["target_experiments"],
        split_spec["split_id"])
    frames = {role: assignment[assignment.role == role].copy()
              for role in ("train", "iid_validation", "target")}
    if set(map(int, frames["target"].experiment)) & set(map(int, frames["train"].experiment)):
        raise ValueError("target experiment leakage in evaluation registry")

    device = torch.device("cuda")
    encoder, model_audit = build_study_model(
        config["model"], EXPECTED_TREATMENTS, config["image_size"])
    if not args.random_init:
        encoder.load_state_dict(checkpoint["encoder"], strict=True)
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    normalization = pretrain_audit["normalization"]
    normalization_mode = config.get("normalization_mode", "frozen_global")
    metadata, features = {}, {}
    for role in ("train", "iid_validation", "target"):
        print(f"[role] extracting {role}", flush=True)
        metadata[role], features[role] = _extract(
            encoder, frames[role], args.raw_root, normalization, normalization_mode,
            config["image_size"], args.batch_size, args.workers, device)

    train_labels = metadata["train"].label.to_numpy(np.int64)
    prototype_rows = {}
    predictions = []
    for role in ("iid_validation", "target"):
        scores, counts = _prototype_scores(features["train"], train_labels, features[role])
        metrics, prediction, ranks = _score_matrix(scores, metadata[role].label)
        metrics["per_experiment"] = _per_experiment_scores(scores, metadata[role])
        prototype_rows[role] = metrics
        predictions.append(metadata[role].assign(
            role=role, prototype_prediction=prediction, prototype_true_rank=ranks))
    if not np.array_equal(counts, counts[0] * np.ones_like(counts)):
        prototype_count_range = [int(counts.min()), int(counts.max())]
    else:
        prototype_count_range = [int(counts[0]), int(counts[0])]

    penalties = (0.01, 0.1, 1.0, 10.0, 100.0)
    probe_candidates = []
    for penalty in penalties:
        weights = _ridge_fit(features["train"], train_labels, EXPECTED_TREATMENTS, penalty)
        metrics, _, _ = _score_matrix(
            _ridge_scores(features["iid_validation"], weights),
            metadata["iid_validation"].label)
        probe_candidates.append({"penalty": penalty, **metrics})
    selected = max(probe_candidates, key=lambda row: (row["top1"], -row["penalty"]))
    probe_weights = _ridge_fit(
        features["train"], train_labels, EXPECTED_TREATMENTS, selected["penalty"])
    probe_rows = {}
    for index, role in enumerate(("iid_validation", "target")):
        role_scores = _ridge_scores(features[role], probe_weights)
        metrics, prediction, ranks = _score_matrix(role_scores, metadata[role].label)
        metrics["per_experiment"] = _per_experiment_scores(role_scores, metadata[role])
        probe_rows[role] = metrics
        predictions[index]["ridge_prediction"] = prediction
        predictions[index]["ridge_true_rank"] = ranks

    source_experiments = sorted(map(int, split_spec["source_experiments"]))
    experiment_map = {value: index for index, value in enumerate(source_experiments)}
    batch_train_labels = metadata["train"].experiment.map(experiment_map).to_numpy(np.int64)
    batch_iid_labels = metadata["iid_validation"].experiment.map(experiment_map).to_numpy(np.int64)
    batch_weights = _ridge_fit(features["train"], batch_train_labels, len(source_experiments), 1.0)
    batch_prediction = _ridge_scores(features["iid_validation"], batch_weights).argmax(1)
    batch_accuracy = float((batch_prediction == batch_iid_labels).mean())

    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_parquet(output / "well_predictions.parquet", index=False)
    result = {
        "schema_version": 1, "state": "complete", "model": config["model"],
        "initialization": "random" if args.random_init else "mae",
        "pretraining_run": config, "pretraining_audit": pretrain_audit,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
        "model_audit": model_audit,
        "prototype_retrieval": prototype_rows,
        "prototype_train_wells_per_class_range": prototype_count_range,
        "ridge_linear_probe": {
            "selection_role": "source iid_validation", "target_used_for_selection": False,
            "candidates": probe_candidates, "selected_penalty": selected["penalty"],
            **probe_rows,
        },
        "source_batch_probe": {
            "iid_accuracy": batch_accuracy, "chance": 1.0 / len(source_experiments),
            "n_source_experiments": len(source_experiments),
        },
        "evaluation_counts": {
            role: {"wells": len(metadata[role]), "sites": len(frames[role])}
            for role in metadata},
        "elapsed_seconds": time.time() - started,
        "device": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
    }
    _atomic_json(output / "RESULT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--site-manifest", required=True)
    parser.add_argument("--raw-root", required=True)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint")
    checkpoint.add_argument("--random-init", action="store_true")
    parser.add_argument("--model", choices=("vit_tiny", "vit_micro"), default="vit_tiny")
    parser.add_argument("--split-id", default="primary_fold0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
