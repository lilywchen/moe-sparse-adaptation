#!/usr/bin/env python
"""Checkpoint-only RxRx1 batch difficulty and embedding geometry analysis.

The analysis uses a deterministic class-matched panel: within each cell line it
selects the same perturbations from every train, OOD-validation, and OOD-test
experiment.  This prevents label composition from masquerading as batch shift.

OOD severity is declared independently of the adapted models.  It is the distance
from each held-out experiment's class-residual centroid to its nearest training
experiment centroid in the *frozen pretrained Cell-DINO* representation, divided
by the median train-to-train distance within the same cell line.

For every supplied fine-tuned checkpoint the script reports:

* per-experiment accuracy, confidence, ECE, and error overlap;
* class and experiment variance fractions in the embedding space;
* cross-batch perturbation retrieval from train prototypes;
* representation drift and linear CKA relative to pretrained Cell-DINO;
* accuracy degradation versus independently measured OOD severity; and
* expert-use/routing-distribution shifts by experiment for routed models.

No optimization occurs.  OOD test is descriptive and must not be used to choose a
new architecture.  The JSON output is the source of truth; ``*.md`` is a compact
human-readable companion.
"""
import argparse
import copy
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.model import build_ccas
from moe_shift.data.rxrx1 import (
    _RawSiteView,
    _SiteView,
    _cell_type_column,
    _rxrx1_raw_transform,
    _rxrx1_transform,
)
from scripts.run_ccas import _sha256_file, git_info


def _parse_named(values):
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        if not label or label in parsed:
            raise ValueError(f"checkpoint label must be nonempty and unique: {label!r}")
        parsed[label] = Path(path).expanduser().resolve()
    return parsed


def _rank(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _corr(x, y, rank=False):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    if rank:
        x, y = _rank(x), _rank(y)
    return float(np.corrcoef(x, y)[0, 1])


def _entropy(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > 0]
    return 0.0 if len(p) <= 1 else float(-(p * np.log(p)).sum() / math.log(len(probabilities)))


def _js(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        keep = a > 0
        return float((a[keep] * np.log(a[keep] / b[keep])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _ece(confidence, correct, bins=10):
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, score = max(len(confidence), 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if mask.any():
            score += mask.sum() / total * abs(confidence[mask].mean() - correct[mask].mean())
    return float(score)


def _linear_cka(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = x - x.mean(0, keepdims=True), y - y.mean(0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return None if denom == 0 else float(cross / denom)


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _metadata_rows(subset, exp_col, cell_col):
    global_indices = np.asarray(subset.indices, dtype=np.int64)
    labels = np.asarray(subset.dataset.y_array[global_indices], dtype=np.int64)
    metadata = np.asarray(subset.metadata_array)
    cells = (np.full(len(labels), -1, dtype=np.int64) if cell_col is None
             else metadata[:, int(cell_col)].astype(np.int64))
    envs = metadata[:, int(exp_col)].astype(np.int64)
    return labels, envs, cells


def _class_matched_indices(subsets, exp_col, cell_col, max_labels):
    """Choose one image per (split, environment, shared cell/label key)."""
    tables, labels_by_cell_env = {}, defaultdict(dict)
    for split, subset in subsets.items():
        labels, envs, cells = _metadata_rows(subset, exp_col, cell_col)
        first = {}
        for local, (label, env, cell) in enumerate(zip(labels, envs, cells)):
            first.setdefault((int(env), int(cell), int(label)), local)
        tables[split] = {"labels": labels, "envs": envs, "cells": cells, "first": first}
        for env, cell, label in first:
            labels_by_cell_env[int(cell)].setdefault((split, int(env)), set()).add(int(label))

    chosen = {}
    for cell, groups in labels_by_cell_env.items():
        shared = set.intersection(*groups.values()) if groups else set()
        if not shared:
            raise ValueError(f"cell type {cell} has no perturbation shared by every environment")
        chosen[cell] = sorted(shared)[: int(max_labels)]

    selected = {}
    for split, table in tables.items():
        indices = []
        for (env, cell, label), local in sorted(table["first"].items()):
            if label in chosen[cell]:
                indices.append(local)
        selected[split] = indices
    return chosen, selected


def _prepare_panel(cfg, max_labels, batch_size, num_workers):
    from wilds import get_dataset

    ds = get_dataset(dataset="rxrx1", root_dir=cfg["data_root"], download=False)
    exp_col = ds.metadata_fields.index("experiment")
    cell_col = _cell_type_column(ds)
    raw_root = cfg.get("rxrx1_raw_root")
    style = str(cfg["train"].get("rxrx1_transform", "imagenet"))
    layout = cfg["train"].get("rxrx1_channel_layout")
    img_size = cfg.get("img_size") or cfg["model"].get("vit", {}).get("img", 256)
    subsets = {name: ds.get_subset(name) if raw_root else ds.get_subset(
        name, transform=_rxrx1_transform(img_size, False, False, style, layout))
               for name in ("train", "val", "test")}
    chosen, selected = _class_matched_indices(subsets, exp_col, cell_col, max_labels)
    train_envs = sorted(set(_metadata_rows(subsets["train"], exp_col, cell_col)[1].tolist()))
    remap = {env: index for index, env in enumerate(train_envs)}

    if raw_root:
        transform = _rxrx1_raw_transform(img_size, False, layout)
        make_view = lambda subset: _RawSiteView(
            subset, exp_col, remap, raw_root, transform, cell_col=cell_col)
    else:
        make_view = lambda subset: _SiteView(subset, exp_col, remap, cell_col=cell_col)
    loaders = {}
    for split, subset in subsets.items():
        panel = Subset(make_view(subset), selected[split])
        loaders[split] = DataLoader(
            panel, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
            pin_memory=True, persistent_workers=(int(num_workers) > 0))
    return loaders, chosen


@torch.no_grad()
def _collect(model, loaders, device, capture_routes=True):
    model.eval()
    output, route_counts = {}, defaultdict(lambda: defaultdict(lambda: None))
    for split, loader in loaders.items():
        feats, logits, labels, envs, cells = [], [], [], [], []
        for batch in loader:
            x, y, _site, env, cell = batch[:5]
            x = x.to(device, non_blocking=True)
            feature = model.forward_features(x)
            logit = model.fc(feature)
            feats.append(feature.float().cpu())
            logits.append(logit.float().cpu())
            labels.append(torch.as_tensor(y).long())
            envs.append(torch.as_tensor(env).long())
            cells.append(torch.as_tensor(cell).long())

            if capture_routes:
                batch_env = torch.as_tensor(env).long().numpy()
                for block_index, block in zip(model.capacity.block_indices, model.moe_blocks):
                    last = getattr(block, "last", None) or {}
                    assignment = last.get("assign")
                    if assignment is None:
                        continue
                    assignment = assignment.detach().cpu().numpy().reshape(-1)
                    if len(assignment) % len(batch_env):
                        raise ValueError("routing assignments do not align with image batch")
                    per_image = assignment.reshape(len(batch_env), -1)
                    for row, raw_env in zip(per_image, batch_env):
                        key = (split, int(raw_env))
                        counts = np.bincount(row, minlength=int(block.n_experts)).astype(np.int64)
                        existing = route_counts[int(block_index)][key]
                        route_counts[int(block_index)][key] = counts if existing is None else existing + counts
        output[split] = {
            "features": torch.cat(feats).numpy(), "logits": torch.cat(logits).numpy(),
            "labels": torch.cat(labels).numpy(), "envs": torch.cat(envs).numpy(),
            "cells": torch.cat(cells).numpy(),
        }
    routes = {
        str(block): {f"{split}:{env}": counts.tolist() for (split, env), counts in rows.items()}
        for block, rows in route_counts.items()
    }
    return output, routes


def _train_prototypes(train):
    prototypes = {}
    for cell in np.unique(train["cells"]):
        for label in np.unique(train["labels"][train["cells"] == cell]):
            mask = (train["cells"] == cell) & (train["labels"] == label)
            prototypes[(int(cell), int(label))] = train["features"][mask].mean(0)
    return prototypes


def _residual_geometry(data):
    train, prototypes = data["train"], _train_prototypes(data["train"])
    combined = {}
    for split, row in data.items():
        residual = np.stack([
            feature - prototypes[(int(cell), int(label))]
            for feature, cell, label in zip(row["features"], row["cells"], row["labels"])
        ])
        combined[split] = {**row, "residual": residual}

    train_residual = combined["train"]["residual"]
    train_global = train_residual.mean(0)
    total_ss = float(((train_residual - train_global) ** 2).sum())
    between_ss = 0.0
    env_centroids, env_cells = {}, {}
    for split, row in combined.items():
        for env in np.unique(row["envs"]):
            mask = row["envs"] == env
            key = (split, int(env))
            env_centroids[key] = row["residual"][mask].mean(0)
            env_cells[key] = int(np.unique(row["cells"][mask])[0])
            if split == "train":
                between_ss += int(mask.sum()) * float(
                    ((env_centroids[key] - train_global) ** 2).sum())

    train_distances = defaultdict(list)
    train_keys = [key for key in env_centroids if key[0] == "train"]
    for i, left in enumerate(train_keys):
        for right in train_keys[i + 1:]:
            if env_cells[left] == env_cells[right]:
                train_distances[env_cells[left]].append(float(np.linalg.norm(
                    env_centroids[left] - env_centroids[right])))
    severity = {}
    for key, centroid in env_centroids.items():
        if key[0] == "train":
            continue
        cell = env_cells[key]
        candidates = [other for other in train_keys if env_cells[other] == cell]
        nearest = min(float(np.linalg.norm(centroid - env_centroids[other])) for other in candidates)
        scale = float(np.median(train_distances[cell])) if train_distances[cell] else 1.0
        severity[f"{key[0]}:{key[1]}"] = nearest / max(scale, 1e-12)

    class_means = np.stack(list(prototypes.values()))
    all_train = train["features"]
    global_feature = all_train.mean(0)
    class_between = sum(
        int(((train["cells"] == cell) & (train["labels"] == label)).sum())
        * float(((prototype - global_feature) ** 2).sum())
        for (cell, label), prototype in prototypes.items())
    class_total = float(((all_train - global_feature) ** 2).sum())
    return combined, prototypes, {
        "batch_variance_fraction_train": between_ss / max(total_ss, 1e-12),
        "class_variance_fraction_train": class_between / max(class_total, 1e-12),
        "severity": severity,
        "n_class_prototypes": len(class_means),
    }


def _retrieval(row, prototypes):
    correct, total = 0, 0
    for cell in np.unique(row["cells"]):
        keys = sorted(key for key in prototypes if key[0] == int(cell))
        proto = _unit(np.stack([prototypes[key] for key in keys]))
        mask = row["cells"] == cell
        query = _unit(row["features"][mask])
        predicted = np.asarray([keys[index][1] for index in (query @ proto.T).argmax(1)])
        correct += int((predicted == row["labels"][mask]).sum())
        total += int(mask.sum())
    return correct / max(total, 1)


def _prediction_summary(data):
    summary, predictions = {}, {}
    for split, row in data.items():
        logits = row["logits"]
        shifted = logits - logits.max(1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(1, keepdims=True)
        pred, conf = probs.argmax(1), probs.max(1)
        correct = pred == row["labels"]
        env_rows = {}
        for env in np.unique(row["envs"]):
            mask = row["envs"] == env
            env_rows[str(int(env))] = {
                "n": int(mask.sum()), "accuracy": float(correct[mask].mean()),
                "mean_confidence": float(conf[mask].mean()),
                "ece": _ece(conf[mask], correct[mask]),
            }
        summary[split] = {
            "n": len(pred), "accuracy": float(correct.mean()),
            "mean_confidence": float(conf.mean()), "ece": _ece(conf, correct),
            "worst_environment_accuracy": min(item["accuracy"] for item in env_rows.values()),
            "per_environment": env_rows,
        }
        predictions[split] = {
            "prediction": pred, "correct": correct, "confidence": conf,
            "labels": row["labels"], "envs": row["envs"], "cells": row["cells"],
        }
    return summary, predictions


def _route_summary(routes, severity):
    report = {}
    for block, rows in routes.items():
        parsed = {}
        train_total = None
        for key, counts in rows.items():
            split, env = key.split(":", 1)
            counts = np.asarray(counts, dtype=np.float64)
            parsed[(split, env)] = counts
            if split == "train":
                train_total = counts if train_total is None else train_total + counts
        if train_total is None:
            continue
        train_p = train_total / train_total.sum()
        env_rows = {}
        sev, shift = [], []
        for (split, env), counts in sorted(parsed.items()):
            p = counts / counts.sum()
            key = f"{split}:{env}"
            js = _js(p, train_p)
            env_rows[key] = {"distribution": p.tolist(), "entropy": _entropy(p),
                             "js_from_train_global": js}
            if key in severity:
                sev.append(severity[key]); shift.append(js)
        report[block] = {
            "train_global_distribution": train_p.tolist(),
            "experts_used": int((train_total > 0).sum()),
            "train_entropy": _entropy(train_p),
            "severity_routing_shift_spearman": _corr(sev, shift, rank=True),
            "per_environment": env_rows,
        }
    return report


def _model_report(data, routes, frozen, severity):
    geometry_data, prototypes, geometry = _residual_geometry(data)
    prediction, predictions = _prediction_summary(data)
    geometry["retrieval"] = {
        split: _retrieval(geometry_data[split], prototypes) for split in ("val", "test")
    }
    aligned = []
    drift_env = {}
    for split in ("train", "val", "test"):
        x, y = data[split]["features"], frozen[split]["features"]
        aligned.append((x, y))
        cosine_drift = 1.0 - (_unit(x) * _unit(y)).sum(1)
        for env in np.unique(data[split]["envs"]):
            mask = data[split]["envs"] == env
            drift_env[f"{split}:{int(env)}"] = float(cosine_drift[mask].mean())
    cka_x = np.concatenate([pair[0] for pair in aligned])
    cka_y = np.concatenate([pair[1] for pair in aligned])
    geometry["linear_cka_to_pretrained"] = _linear_cka(cka_x, cka_y)
    geometry["mean_cosine_drift_by_environment"] = drift_env

    sev, acc, drift = [], [], []
    for key, value in severity.items():
        split, env = key.split(":", 1)
        sev.append(value)
        acc.append(prediction[split]["per_environment"][env]["accuracy"])
        drift.append(drift_env[key])
    geometry["severity_accuracy_pearson"] = _corr(sev, acc)
    geometry["severity_accuracy_spearman"] = _corr(sev, acc, rank=True)
    geometry["severity_drift_spearman"] = _corr(sev, drift, rank=True)
    return {
        "prediction": prediction, "embedding_geometry": geometry,
        "routing": _route_summary(routes, severity),
    }, predictions


def _error_overlap(predictions):
    labels = sorted(predictions)
    report = {}
    for split in ("val", "test"):
        rows = {}
        for i, left in enumerate(labels):
            for right in labels[i + 1:]:
                a = ~predictions[left][split]["correct"]
                b = ~predictions[right][split]["correct"]
                union = int((a | b).sum())
                rows[f"{left}__vs__{right}"] = {
                    "error_jaccard": float((a & b).sum() / max(union, 1)),
                    "prediction_disagreement": float((
                        predictions[left][split]["prediction"]
                        != predictions[right][split]["prediction"]).mean()),
                    f"{left}_rescues_{right}": float((~a & b).mean()),
                    f"{right}_rescues_{left}": float((a & ~b).mean()),
                }
        report[split] = rows
    return report


def _markdown(payload):
    lines = ["# RxRx1 batch and embedding diagnostics", "",
             f"Class-matched panel: {payload['panel']['n_examples']} images; "
             f"{payload['panel']['labels_per_cell']} perturbations per cell line.", "",
             "OOD severity is frozen-pretrained, class-residual, within-cell-line distance.", "",
             "| Model | Val acc | Test acc | Worst test | Val conf | Batch R² | Class R² | "
             "Val retrieval | CKA to pretrained | Severity→acc ρ |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, row in payload["models"].items():
        pred, geom = row["prediction"], row["embedding_geometry"]
        fmt = lambda x: "—" if x is None else f"{float(x):.4f}"
        lines.append(
            f"| {label} | {fmt(pred['val']['accuracy'])} | {fmt(pred['test']['accuracy'])} | "
            f"{fmt(pred['test']['worst_environment_accuracy'])} | "
            f"{fmt(pred['val']['mean_confidence'])} | "
            f"{fmt(geom['batch_variance_fraction_train'])} | "
            f"{fmt(geom['class_variance_fraction_train'])} | "
            f"{fmt(geom['retrieval']['val'])} | {fmt(geom['linear_cka_to_pretrained'])} | "
            f"{fmt(geom['severity_accuracy_spearman'])} |")
    lines += ["", "## Held-out experiment difficulty", "",
              "| Split:experiment | Frozen severity | " + " | ".join(payload["models"]) + " |",
              "|---|---:|" + "---:|" * len(payload["models"])]
    for key, severity in sorted(payload["frozen_severity"].items()):
        split, env = key.split(":", 1)
        values = [payload["models"][label]["prediction"][split]["per_environment"][env]["accuracy"]
                  for label in payload["models"]]
        lines.append(f"| {key} | {severity:.3f} | "
                     + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += ["", "Panel accuracies are diagnostic estimates; terminal result JSONs remain the "
              "authoritative full-split accuracy table. OOD test is descriptive only."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-labels-per-cell", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    checkpoints = _parse_named(args.checkpoint)
    payloads = {}
    for label, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads[label] = torch.load(path, map_location="cpu", weights_only=False)
    first = next(iter(payloads.values()))
    cfg = first["config"]
    loaders, chosen = _prepare_panel(
        cfg, args.max_labels_per_cell, args.batch_size, args.num_workers)

    frozen_cfg = copy.deepcopy(cfg)
    frozen_cfg["model"]["variant"] = "original"
    torch.manual_seed(0)
    frozen_model = build_ccas(frozen_cfg).to(args.device)
    frozen, _ = _collect(frozen_model, loaders, args.device, capture_routes=False)
    del frozen_model
    torch.cuda.empty_cache()
    _frozen_rows, _frozen_prototypes, frozen_geometry = _residual_geometry(frozen)
    severity = frozen_geometry["severity"]

    reports, predictions, provenance = {}, {}, {}
    for label, checkpoint in checkpoints.items():
        stored = payloads[label]
        model = build_ccas(stored["config"]).to(args.device)
        model.load_state_dict(stored["model"], strict=True)
        data, routes = _collect(model, loaders, args.device)
        reports[label], predictions[label] = _model_report(data, routes, frozen, severity)
        provenance[label] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256_file(checkpoint),
            "run_id": stored.get("run_id"), "epoch": stored.get("epoch"),
            "variant": stored.get("config", {}).get("model", {}).get("variant"),
        }
        del model, data
        torch.cuda.empty_cache()

    git_sha, git_dirty = git_info()
    result = {
        "schema_version": 1,
        "selection_rule": "diagnostic_only; architecture decisions use OOD validation",
        "severity_definition": (
            "nearest train-experiment class-residual centroid distance in frozen pretrained "
            "Cell-DINO, normalized by within-cell median train-experiment distance"
        ),
        "panel": {
            "labels_per_cell": {str(cell): len(labels) for cell, labels in chosen.items()},
            "n_examples": sum(len(loader.dataset) for loader in loaders.values()),
            "split_sizes": {split: len(loader.dataset) for split, loader in loaders.items()},
        },
        "frozen_severity": severity,
        "models": reports,
        "error_overlap": _error_overlap(predictions),
        "checkpoints": provenance,
        "analysis_git_sha": git_sha,
        "analysis_git_dirty": git_dirty,
        "ood_test_role": "descriptive_fixed_checkpoint_analysis_only",
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, output)
    output.with_suffix(".md").write_text(_markdown(result))
    print(_markdown(result), flush=True)
    print(f"\nwrote {output} and {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
