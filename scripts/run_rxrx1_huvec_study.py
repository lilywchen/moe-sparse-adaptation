#!/usr/bin/env python
"""Run one audited raw-image arm from the frozen RxRx1 HUVEC study registry."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.data.rxrx1_huvec import (
    EXPECTED_TREATMENTS,
    Native6SiteDataset,
    deterministic_split,
)
from moe_shift.models.huvec import (
    MaskedAutoencoder,
    build_study_model,
    clone_encoder_state,
)


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _atomic_torch(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _git_info():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip())
    return sha, dirty


def _split_hash(frame):
    columns = frame[["global_index", "well_id", "role"]].sort_values("global_index")
    return hashlib.sha256(columns.to_csv(index=False).encode()).hexdigest()


def _load_spec(result_root, run_id):
    root = Path(result_root)
    manifest = json.loads((root / "wave_manifest.json").read_text())
    matches = [row for row in manifest["runs"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"run_id {run_id!r} occurs {len(matches)} times in wave manifest")
    registry = json.loads((root / "study_registry.json").read_text())
    split_specs = {row["split_id"]: row for row in registry["main_training_splits"]}
    spec = matches[0]
    if spec["split_id"] not in split_specs:
        raise KeyError(f"unknown frozen split {spec['split_id']!r}")
    return manifest, registry, spec, split_specs[spec["split_id"]]


def _make_loaders(root, registry, split_spec, batch_size, workers, image_size,
                  canary=False):
    sites = pd.read_parquet(registry["site_manifest"])
    assignment = deterministic_split(
        sites, split_spec["source_experiments"], split_spec["target_experiments"],
        split_spec["split_id"])
    roles = {role: assignment[assignment.role == role].copy()
             for role in ("train", "iid_validation", "target")}
    train_wells = set(roles["train"].well_id)
    if train_wells & set(roles["iid_validation"].well_id) or train_wells & set(roles["target"].well_id):
        raise ValueError("well leakage across training/evaluation roles")
    if set(map(int, roles["target"].experiment)) & set(map(int, roles["train"].experiment)):
        raise ValueError("target experiment leakage into source training")
    if canary:
        # Cycle across labels so the canary exercises the full label mapping, not one contiguous
        # metadata region. Both sites of every selected well remain together.
        wells = roles["train"].drop_duplicates("well_id").sort_values(["label", "well_id"])
        wells = wells.groupby("label", sort=True).head(1).head(256)
        roles["train"] = roles["train"][roles["train"].well_id.isin(wells.well_id)].copy()
    normalization = split_spec["normalization"]
    raw_root = registry["raw_root"]
    datasets = {
        "train": Native6SiteDataset(
            roles["train"], raw_root, image_size, normalization["mean"], normalization["std"],
            train=not canary),
        "train_eval": Native6SiteDataset(
            roles["train"], raw_root, image_size, normalization["mean"], normalization["std"],
            train=False),
        "iid_validation": Native6SiteDataset(
            roles["iid_validation"], raw_root, image_size, normalization["mean"],
            normalization["std"], train=False),
        "target": Native6SiteDataset(
            roles["target"], raw_root, image_size, normalization["mean"], normalization["std"],
            train=False),
    }
    generator = torch.Generator().manual_seed(20260814)
    loaders = {
        name: DataLoader(
            dataset, batch_size=int(batch_size), shuffle=(name == "train"),
            num_workers=int(workers), pin_memory=True, drop_last=False,
            persistent_workers=(int(workers) > 0),
            generator=(generator if name == "train" else None),
        ) for name, dataset in datasets.items()
    }
    return assignment, loaders


def _well_metrics(logits, labels, experiments, well_ids):
    grouped = defaultdict(list)
    for index, well_id in enumerate(well_ids):
        grouped[str(well_id)].append(index)
    rows, averaged, truth = [], [], []
    for well_id, indices in grouped.items():
        unique_labels = set(map(int, labels[indices].tolist()))
        unique_experiments = set(map(int, experiments[indices].tolist()))
        if len(unique_labels) != 1 or len(unique_experiments) != 1:
            raise ValueError(f"inconsistent site metadata within well {well_id}")
        mean_logits = logits[indices].mean(0)
        label = unique_labels.pop(); experiment = unique_experiments.pop()
        order = torch.argsort(mean_logits, descending=True)
        rank = int((order == label).nonzero(as_tuple=False)[0, 0]) + 1
        rows.append({
            "well_id": well_id, "experiment": experiment, "label": label,
            "prediction": int(order[0]), "true_class_rank": rank,
            "true_log_probability": float(F.log_softmax(mean_logits, 0)[label]),
            "correct_top1": bool(rank == 1), "correct_top5": bool(rank <= 5),
            "n_sites": len(indices),
        })
        averaged.append(mean_logits); truth.append(label)
    prediction_frame = pd.DataFrame(rows)
    averaged = torch.stack(averaged); truth = torch.tensor(truth, dtype=torch.long)
    per_experiment = {}
    for experiment, group in prediction_frame.groupby("experiment"):
        per_experiment[str(int(experiment))] = {
            "n_wells": len(group), "top1": float(group.correct_top1.mean()),
            "top5": float(group.correct_top5.mean()),
            "mean_rank": float(group.true_class_rank.mean()),
        }
    metrics = {
        "n_wells": len(prediction_frame), "n_sites": len(labels),
        "top1": float(prediction_frame.correct_top1.mean()),
        "top5": float(prediction_frame.correct_top5.mean()),
        "loss": float(F.cross_entropy(averaged, truth)),
        "mean_rank": float(prediction_frame.true_class_rank.mean()),
        "per_experiment": per_experiment,
    }
    return metrics, prediction_frame


@torch.no_grad()
def evaluate(model, loader, device, capture_routing=False):
    model.eval()
    all_logits, all_labels, all_experiments, all_wells = [], [], [], []
    routing = defaultdict(lambda: defaultdict(lambda: None))
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images).float().cpu()
        all_logits.append(logits)
        all_labels.append(torch.as_tensor(batch["label"]).long())
        all_experiments.append(torch.as_tensor(batch["experiment"]).long())
        all_wells.extend(list(batch["well_id"]))
        if capture_routing and getattr(model, "moe_blocks", ()):
            batch_experiments = torch.as_tensor(batch["experiment"]).long().numpy()
            for block_index, block in zip(model.moe_block_indices, model.moe_blocks):
                assignment = block.last["idx"][:, 0].detach().cpu().numpy()
                if len(assignment) % len(batch_experiments):
                    raise ValueError("routing assignments do not align to site images")
                assignment = assignment.reshape(len(batch_experiments), -1)
                for values, experiment in zip(assignment, batch_experiments):
                    count = np.bincount(values, minlength=block.N).astype(np.int64)
                    existing = routing[int(block_index)][int(experiment)]
                    routing[int(block_index)][int(experiment)] = (
                        count if existing is None else existing + count)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    experiments = torch.cat(all_experiments)
    metrics, predictions = _well_metrics(logits, labels, experiments, all_wells)
    routing_summary = {}
    for block, rows in routing.items():
        routing_summary[str(block)] = {}
        for experiment, counts in rows.items():
            probabilities = counts / max(int(counts.sum()), 1)
            nonzero = probabilities[probabilities > 0]
            entropy = 0.0 if len(nonzero) <= 1 else float(
                -(nonzero * np.log(nonzero)).sum() / math.log(len(probabilities)))
            routing_summary[str(block)][str(experiment)] = {
                "counts": counts.tolist(), "distribution": probabilities.tolist(),
                "entropy": entropy,
            }
    return metrics, predictions, routing_summary


def _schedule(optimizer, epoch, total_epochs, warmup=3):
    if epoch < warmup:
        factor = float(epoch + 1) / max(int(warmup), 1)
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup - 1, 1)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor


def _optimizer(model, kind):
    lr = 1e-3 if kind == "resnet18" else 7.5e-4
    wd = 1e-4 if kind == "resnet18" else 0.05
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for group in optimizer.param_groups:
        group["initial_lr"] = lr
    return optimizer, {"name": "adamw", "lr": lr, "weight_decay": wd}


def _train_mae(model, loader, device, epochs, curve_path):
    mae = MaskedAutoencoder(model, mask_ratio=0.75).to(device)
    optimizer = torch.optim.AdamW(mae.parameters(), lr=1e-3, weight_decay=0.05)
    for group in optimizer.param_groups:
        group["initial_lr"] = 1e-3
    curves = []
    for epoch in range(int(epochs)):
        _schedule(optimizer, epoch, int(epochs), warmup=2)
        mae.train(); total = count = 0
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            reconstruction, auxiliary = mae(images)
            loss = reconstruction + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite MAE loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if batch_index == 0:
                gradients = [parameter.grad for parameter in mae.parameters()
                             if parameter.requires_grad and parameter.grad is not None]
                if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                    raise FloatingPointError("missing or non-finite MAE gradients")
            torch.nn.utils.clip_grad_norm_(mae.parameters(), 5.0); optimizer.step()
            total += float(reconstruction) * len(images); count += len(images)
        row = {"phase": "mae", "epoch": epoch + 1, "reconstruction_loss": total / count,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        curves.append(row); print(f"[mae] {row}", flush=True)
    curve_path.write_text("".join(json.dumps(row) + "\n" for row in curves))
    return clone_encoder_state(mae), curves


def train_run(result_root, run_id):
    root = Path(result_root)
    _manifest, registry, run_spec, split_spec = _load_spec(root, run_id)
    output_path = root / "runs" / f"{run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        print(f"[skip] complete {run_id}", flush=True); return json.loads(output_path.read_text())
    torch.manual_seed(int(run_spec.get("seed", 0)))
    np.random.seed(int(run_spec.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kind = run_spec["model"]
    canary = bool(run_spec.get("canary", False))
    batch_size = int(run_spec.get("batch_size", 128))
    workers = int(run_spec.get("num_workers", 8))
    image_size = int(run_spec.get("image_size", 224))
    assignment, loaders = _make_loaders(
        root, registry, split_spec, batch_size, workers, image_size, canary=canary)
    model_kind = kind.replace("mae_", "") if kind.startswith("mae_") else kind
    if model_kind == "vit_tiny_moe":
        build_kind = "vit_tiny_moe"
    elif model_kind == "vit_tiny":
        build_kind = "vit_tiny"
    else:
        build_kind = model_kind
    model, model_audit = build_study_model(build_kind, EXPECTED_TREATMENTS, image_size)
    model = model.to(device)
    first = next(iter(loaders["train"]))
    if tuple(first["image"].shape[1:]) != (6, image_size, image_size):
        raise ValueError(f"raw-image loader contract failed: {tuple(first['image'].shape)}")
    with torch.no_grad():
        if model(first["image"].to(device)).shape[-1] != EXPECTED_TREATMENTS:
            raise ValueError("classifier output dimension mismatch")

    curve_path = root / "runs" / f"{run_id}.curves.jsonl"
    pretraining_curves = []
    if kind.startswith("mae_"):
        state, pretraining_curves = _train_mae(
            model, loaders["train"], device, int(run_spec.get("pretrain_epochs", 20)),
            root / "runs" / f"{run_id}.mae_curves.jsonl")
        model.load_state_dict(state, strict=True)

    if canary:
        # No augmentation, dropout, or weight decay; cycle the fixed data until memorized.
        model.train(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        iterator = iter(loaders["train_eval"]); steps = int(run_spec.get("canary_steps", 3000))
        curve = []
        for step in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loaders["train_eval"]); batch = next(iterator)
            images = batch["image"].to(device); labels = batch["label"].to(device)
            logits = model(images); loss = F.cross_entropy(logits, labels)
            if getattr(model, "moe_blocks", ()):
                loss = loss + model.routing_aux_loss()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if (step + 1) % 100 == 0 or step == 0:
                metrics, _, _ = evaluate(model, loaders["train_eval"], device)
                row = {"phase": "canary", "step": step + 1, "loss": float(loss),
                       "train_top1": metrics["top1"]}
                curve.append(row); print(f"[canary] {row}", flush=True)
                if metrics["top1"] >= 0.99:
                    break
        curve_path.write_text("".join(json.dumps(row) + "\n" for row in curve))
        passed = bool(curve[-1]["train_top1"] >= 0.99)
        sha, dirty = _git_info()
        result = {
            "schema_version": 1, "run_id": run_id, "stage": run_spec["stage"],
            "model": kind, "canary": True, "canary_passed": passed,
            "train_top1": curve[-1]["train_top1"], "steps": curve[-1]["step"],
            "model_audit": model_audit, "split_hash": _split_hash(assignment),
            "git_commit": sha, "git_dirty": dirty, "hostname": socket.gethostname(),
        }
        _atomic_json(output_path, result)
        if not passed:
            raise RuntimeError(f"{kind} failed the >=99% memorization gate")
        return result

    epochs = int(run_spec.get("epochs", 30))
    optimizer, optimizer_audit = _optimizer(model, build_kind)
    best_iid, best_state, best_epoch = -1.0, None, None
    curves = list(pretraining_curves)
    started = time.time()
    initial_loss = None
    initial_router_task_gradient = None
    for epoch in range(epochs):
        _schedule(optimizer, epoch, epochs, warmup=3)
        model.train(); total_loss = total_correct = total_count = 0
        for batch_index, batch in enumerate(loaders["train"]):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images)
            classification = F.cross_entropy(logits, labels)
            if epoch == 0 and batch_index == 0 and getattr(model, "moe_blocks", ()):
                router_parameters = []
                for block in model.moe_blocks:
                    router_parameters.extend(list(block.proj.parameters()))
                    router_parameters.extend([block.codebook, block.log_temp])
                router_gradients = torch.autograd.grad(
                    classification, router_parameters, retain_graph=True, allow_unused=True)
                initial_router_task_gradient = float(torch.stack([
                    gradient.float().square().sum() for gradient in router_gradients
                    if gradient is not None]).sum().sqrt())
                if not math.isfinite(initial_router_task_gradient) or initial_router_task_gradient <= 0:
                    raise RuntimeError("top-1 MoE router has no classification-loss gradient")
            auxiliary = model.routing_aux_loss() if getattr(model, "moe_blocks", ()) else classification.new_zeros(())
            loss = classification + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite supervised loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if epoch == 0 and batch_index == 0:
                gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("missing or non-finite supervised gradients")
                initial_loss = float(classification)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total_loss += float(classification) * len(images)
            total_correct += int((logits.argmax(1) == labels).sum()); total_count += len(images)
        should_evaluate = epoch == epochs - 1 or (epoch + 1) % 5 == 0
        row = {
            "phase": "supervised", "epoch": epoch + 1,
            "train_augmented_loss": total_loss / total_count,
            "train_augmented_top1": total_correct / total_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if should_evaluate:
            iid_metrics, _, _ = evaluate(model, loaders["iid_validation"], device)
            row["iid_top1"] = iid_metrics["top1"]
            if iid_metrics["top1"] > best_iid:
                best_iid, best_epoch = iid_metrics["top1"], epoch + 1
                best_state = copy.deepcopy({key: value.detach().cpu()
                                            for key, value in model.state_dict().items()})
        curves.append(row); print(f"[train] {run_id} {row}", flush=True)
        curve_path.write_text("".join(json.dumps(value) + "\n" for value in curves))
    if best_state is None:
        raise RuntimeError("no source-IID checkpoint was selected")
    model.load_state_dict(best_state); model.to(device)
    train_metrics, train_predictions, _ = evaluate(model, loaders["train_eval"], device)
    iid_metrics, iid_predictions, iid_routing = evaluate(
        model, loaders["iid_validation"], device, capture_routing=True)
    target_metrics, target_predictions, target_routing = evaluate(
        model, loaders["target"], device, capture_routing=True)
    predictions = pd.concat([
        train_predictions.assign(role="train"), iid_predictions.assign(role="iid_validation"),
        target_predictions.assign(role="target")], ignore_index=True)
    predictions.insert(0, "run_id", run_id)
    predictions.to_parquet(root / "runs" / f"{run_id}.predictions.parquet", index=False)
    _atomic_torch(root / "runs" / f"{run_id}.checkpoint.pt", {
        "run_id": run_id, "model": best_state, "model_audit": model_audit,
        "split_id": split_spec["split_id"], "best_epoch": best_epoch,
    })
    sha, dirty = _git_info()
    chance = 1.0 / EXPECTED_TREATMENTS
    training_certified = bool(
        train_metrics["top1"] >= 0.80 and best_iid >= 2.0 * chance
        and curves[-1]["train_augmented_loss"] <= 0.8 * max(initial_loss, 1e-12))
    result = {
        "schema_version": 1, "run_id": run_id, "stage": run_spec["stage"],
        "model": kind, "canary": False, "seed": int(run_spec.get("seed", 0)),
        "split_id": split_spec["split_id"], "split_kind": split_spec["kind"],
        "difficulty_tier": split_spec["difficulty_tier"],
        "source_experiments": split_spec["source_experiments"],
        "target_experiments": split_spec["target_experiments"],
        "target_difficulty": split_spec["target_difficulty"],
        "raw_qc_target_difficulty": split_spec["raw_qc_target_difficulty"],
        "target_label_coverage": split_spec["target_label_coverage"],
        "normalization": split_spec["normalization"],
        "split_hash": _split_hash(assignment), "best_epoch": best_epoch,
        "training_certified": training_certified, "initial_batch_loss": initial_loss,
        "initial_router_task_gradient": initial_router_task_gradient,
        "train": train_metrics, "iid_validation": iid_metrics, "target": target_metrics,
        "iid_to_target_gap": iid_metrics["top1"] - target_metrics["top1"],
        "routing": {"iid_validation": iid_routing, "target": target_routing},
        "model_audit": model_audit, "optimizer": optimizer_audit,
        "epochs": epochs, "pretrain_epochs": int(run_spec.get("pretrain_epochs", 0)),
        "elapsed_seconds": time.time() - started, "git_commit": sha, "git_dirty": dirty,
        "hostname": socket.gethostname(), "torch_version": torch.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    _atomic_json(output_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = train_run(args.result_root, args.run_id)
    print(json.dumps({"run_id": result["run_id"], "complete": True,
                      "training_certified": result.get("training_certified"),
                      "target_top1": (result.get("target") or {}).get("top1")},
                     indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
