#!/usr/bin/env python
"""Train one frozen HUVEC batch-degradation or capacity-mechanism arm."""
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
from moe_shift.models.huvec import build_study_model
from scripts.evaluate_rxrx1_huvec_sites import _site_metrics
from scripts.run_rxrx1_huvec_study import _split_hash, _well_metrics


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def atomic_torch(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def atomic_parquet(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False); os.replace(temporary, path)


def git_identity():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
        text=True).strip())
    return commit, dirty


def fingerprint(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def append_jsonl(path, row):
    with open(path, "a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def load_spec(result_root, run_id):
    root = Path(result_root).expanduser().resolve()
    manifest = json.loads((root / "wave_manifest.json").read_text())
    registry = json.loads((root / "study_registry.json").read_text())
    run_matches = [row for row in manifest["runs"] if row["run_id"] == run_id]
    if len(run_matches) != 1:
        raise ValueError(f"run {run_id!r} occurs {len(run_matches)} times")
    split_matches = [row for row in registry["splits"]
                     if row["split_id"] == run_matches[0]["split_id"]]
    if len(split_matches) != 1:
        raise ValueError(f"split for {run_id!r} occurs {len(split_matches)} times")
    return root, manifest, registry, run_matches[0], split_matches[0]


def make_source_loaders(registry, split, batch_size, workers, image_size):
    sites = pd.read_parquet(registry["site_manifest"])
    assignment = deterministic_split(
        sites, split["source_experiments"], split["target_experiments"],
        split["split_id"])
    if _split_hash(assignment) != split["split_hash"]:
        raise RuntimeError("frozen diagnostic split hash changed")
    roles = {role: assignment[assignment.role == role].copy()
             for role in ("train", "iid_validation", "target")}
    target_experiments = set(map(int, split["target_experiments"]))
    if target_experiments & set(map(int, roles["train"].experiment)):
        raise RuntimeError("target experiment leaked into training")
    normalization = split["normalization"]
    common = dict(
        raw_root=registry["raw_root"], img_size=image_size,
        mean=normalization["mean"], std=normalization["std"])
    datasets = {
        "train": Native6SiteDataset(roles["train"], train=True, **common),
        "train_eval": Native6SiteDataset(roles["train"], train=False, **common),
        "iid_validation": Native6SiteDataset(
            roles["iid_validation"], train=False, **common),
    }
    generator = torch.Generator().manual_seed(20260815)
    loaders = {
        name: DataLoader(
            dataset, batch_size=int(batch_size), shuffle=(name == "train"),
            num_workers=int(workers), pin_memory=True, drop_last=False,
            persistent_workers=(int(workers) > 0),
            generator=(generator if name == "train" else None),
        ) for name, dataset in datasets.items()
    }
    return sites, assignment, roles, loaders, generator


def make_target_loader(registry, split, target_frame, batch_size, workers, image_size):
    normalization = split["normalization"]
    dataset = Native6SiteDataset(
        target_frame, registry["raw_root"], image_size,
        normalization["mean"], normalization["std"], train=False)
    return DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False,
        num_workers=int(workers), pin_memory=True, drop_last=False,
        persistent_workers=(int(workers) > 0))


def schedule(optimizer, epoch, recipe):
    warmup = int(recipe["warmup_epochs"])
    total = int(recipe["schedule_epochs"])
    minimum = float(recipe["min_lr_ratio"])
    if epoch < warmup:
        factor = float(epoch + 1) / max(warmup, 1)
    else:
        progress = (epoch - warmup) / max(total - warmup - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        factor = minimum + (1.0 - minimum) * cosine
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor


def build_optimizer(model, recipe):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(recipe["lr"]),
        weight_decay=float(recipe["weight_decay"]))
    for group in optimizer.param_groups:
        group["initial_lr"] = float(recipe["lr"])
    return optimizer


def routing_summary(model, experiments):
    rows = {}
    if not getattr(model, "moe_blocks", ()):
        return rows
    experiments = np.asarray(experiments, dtype=np.int64)
    for block_index, block in zip(model.moe_block_indices, model.moe_blocks):
        assignment = block.last["idx"][:, 0].detach().cpu().numpy()
        if len(assignment) % len(experiments):
            raise ValueError("routing assignments do not align with site images")
        assignment = assignment.reshape(len(experiments), -1)
        block_rows = rows.setdefault(str(block_index), {})
        for values, experiment in zip(assignment, experiments):
            counts = np.bincount(values, minlength=block.N).astype(np.int64)
            key = str(int(experiment))
            block_rows[key] = counts if key not in block_rows else block_rows[key] + counts
    return rows


def merge_routing(total, batch):
    for block, experiments in batch.items():
        destination = total.setdefault(block, {})
        for experiment, counts in experiments.items():
            destination[experiment] = (
                counts.copy() if experiment not in destination
                else destination[experiment] + counts)


@torch.inference_mode()
def evaluate_detailed(model, loader, device, role):
    model.eval()
    logits, features, labels, experiments = [], [], [], []
    sites, global_indices, well_ids = [], [], []
    routing = {}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        feature = model.forward_features(images)
        output = model.fc(feature)
        logits.append(output.float().cpu()); features.append(feature.float().cpu())
        batch_labels = torch.as_tensor(batch["label"]).long()
        batch_experiments = torch.as_tensor(batch["experiment"]).long()
        labels.append(batch_labels); experiments.append(batch_experiments)
        sites.append(torch.as_tensor(batch["site"]).long())
        global_indices.append(torch.as_tensor(batch["global_index"]).long())
        well_ids.extend(list(batch["well_id"]))
        merge_routing(routing, routing_summary(model, batch_experiments.numpy()))
    logits = torch.cat(logits); features = torch.cat(features)
    labels = torch.cat(labels); experiments = torch.cat(experiments)
    sites = torch.cat(sites); global_indices = torch.cat(global_indices)
    well_metrics, well_predictions = _well_metrics(
        logits, labels, experiments, well_ids)
    site_metrics, site_predictions = _site_metrics(
        logits, labels, experiments, well_ids, sites, global_indices)
    well_predictions.insert(0, "role", role)
    site_predictions.insert(0, "role", role)

    feature_rows = []
    feature_array = features.numpy().astype(np.float16)
    grouped_indices = defaultdict(list)
    for index, well_id in enumerate(well_ids):
        grouped_indices[str(well_id)].append(index)
    for well_id, indices in grouped_indices.items():
        index = np.asarray(indices, dtype=np.int64)
        feature_rows.append({
            "role": role, "well_id": str(well_id),
            "experiment": int(experiments[index[0]]), "label": int(labels[index[0]]),
            "n_sites": len(index), "embedding": feature_array[index].mean(0).tolist(),
        })
    route_payload = {}
    for block, rows in routing.items():
        route_payload[block] = {}
        for experiment, counts in rows.items():
            probabilities = counts / max(int(counts.sum()), 1)
            nonzero = probabilities[probabilities > 0]
            entropy = 0.0 if len(nonzero) <= 1 else float(
                -(nonzero * np.log(nonzero)).sum() / math.log(len(probabilities)))
            route_payload[block][experiment] = {
                "counts": counts.tolist(), "distribution": probabilities.tolist(),
                "normalized_entropy": entropy,
            }
    metrics = {"site": site_metrics, "well": well_metrics}
    return metrics, site_predictions, well_predictions, pd.DataFrame(feature_rows), route_payload


def simple_iid_metrics(model, loader, device):
    model.eval(); correct = count = 0
    well_logits = defaultdict(list); well_labels = {}
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device, non_blocking=True)).float().cpu()
            labels = torch.as_tensor(batch["label"]).long()
            correct += int((logits.argmax(1) == labels).sum()); count += len(labels)
            for index, well_id in enumerate(batch["well_id"]):
                well_logits[str(well_id)].append(logits[index])
                well_labels[str(well_id)] = int(labels[index])
    well_correct = sum(int(torch.stack(values).mean(0).argmax() == well_labels[well])
                       for well, values in well_logits.items())
    return {"site_top1": correct / max(count, 1),
            "well_top1": well_correct / max(len(well_logits), 1)}


def _resume(path, expected_fingerprint, model, optimizer, generator, device):
    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["fingerprint"] != expected_fingerprint:
        raise RuntimeError("resume checkpoint does not match the frozen run configuration")
    model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda" and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    np.random.set_state(state["numpy_rng"])
    generator.set_state(state["loader_rng"])
    return state


def _save_resume(path, run_fingerprint, model, optimizer, generator, device, **state):
    atomic_torch(path, {
        "schema_version": 1, "fingerprint": run_fingerprint,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "numpy_rng": np.random.get_state(), "loader_rng": generator.get_state(), **state,
    })


def train(result_root, run_id):
    root, manifest, registry, run_spec, split = load_spec(result_root, run_id)
    run_dir = root / "runs" / run_id; run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "RESULT.json"; status_path = run_dir / "STATUS.json"
    if result_path.is_file():
        print(f"[skip] complete {run_id}", flush=True)
        return json.loads(result_path.read_text())
    commit, dirty = git_identity()
    if dirty:
        raise RuntimeError("refuse to train from a dirty tracked checkout")
    if commit != manifest["source_git_commit"]:
        raise RuntimeError("runtime commit differs from the frozen wave manifest")
    if set(map(int, split["target_experiments"])) & set(
            map(int, registry["sealed_primary_targets"])):
        raise RuntimeError("a sealed primary target entered the diagnostic wave")

    torch.manual_seed(int(run_spec["seed"])); np.random.seed(int(run_spec["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("the 12-hour HUVEC study requires CUDA")
    device = torch.device("cuda")
    sites, assignment, roles, loaders, generator = make_source_loaders(
        registry, split, run_spec["batch_size"], run_spec["num_workers"],
        run_spec["image_size"])
    del sites
    # No target Dataset or DataLoader exists before checkpoint selection.
    model, model_audit = build_study_model(
        run_spec["model"], EXPECTED_TREATMENTS, run_spec["image_size"])
    model = model.to(device); optimizer = build_optimizer(model, run_spec["recipe"])
    run_fingerprint = fingerprint({
        "run_spec": run_spec, "split": split, "manifest_commit": commit,
    })
    frozen = {
        "schema_version": 1, "run_id": run_id, "run_fingerprint": run_fingerprint,
        "run_spec": run_spec, "split": split, "model_audit": model_audit,
        "split_hash": _split_hash(assignment), "git_commit": commit,
        "target_policy": registry["target_policy"],
    }
    frozen_path = run_dir / "FROZEN_RUN.json"
    if frozen_path.is_file():
        if json.loads(frozen_path.read_text())["run_fingerprint"] != run_fingerprint:
            raise RuntimeError("frozen run metadata changed")
    else:
        atomic_json(frozen_path, frozen)

    resume_path = run_dir / "resume.pt"; best_path = run_dir / "best_checkpoint.pt"
    curve_path = run_dir / "curves.jsonl"
    state = _resume(resume_path, run_fingerprint, model, optimizer, generator, device)
    start_epoch = int(state["epoch"]) if state else 0
    best_iid = float(state["best_iid"]) if state else -1.0
    best_epoch = int(state["best_epoch"]) if state else None
    plateau_best = float(state["plateau_best"]) if state else -1.0
    stale = int(state["stale_evaluations"]) if state else 0
    elapsed_before = float(state["elapsed_seconds"]) if state else 0.0
    initial_loss = state.get("initial_loss") if state else None
    initial_router_gradient = state.get("initial_router_gradient") if state else None
    if curve_path.is_file():
        rows = [json.loads(line) for line in curve_path.read_text().splitlines() if line.strip()]
        rows = [row for row in rows if int(row["epoch"]) <= start_epoch]
        curve_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    selection = run_spec["selection"]
    started = time.time(); terminal_epoch = start_epoch; stop_reason = None
    atomic_json(status_path, {
        **frozen, "state": "training", "epoch": start_epoch,
        "best_source_iid_site_top1": best_iid, "best_epoch": best_epoch,
        "stale_evaluations": stale,
    })
    for epoch in range(start_epoch, int(selection["maximum_epochs"])):
        schedule(optimizer, epoch, run_spec["recipe"])
        model.train(); total_loss = total_correct = total_count = 0
        for batch_index, batch in enumerate(loaders["train"]):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(images); classification = F.cross_entropy(logits, labels)
            if epoch == 0 and batch_index == 0 and getattr(model, "moe_blocks", ()):
                router_parameters = []
                for block in model.moe_blocks:
                    router_parameters += list(block.proj.parameters())
                    router_parameters += [block.codebook, block.log_temp]
                gradients = torch.autograd.grad(
                    classification, router_parameters, retain_graph=True, allow_unused=True)
                initial_router_gradient = float(torch.stack([
                    value.float().square().sum() for value in gradients if value is not None
                ]).sum().sqrt())
                if not math.isfinite(initial_router_gradient) or initial_router_gradient <= 0:
                    raise RuntimeError("MoE router receives no classification-loss gradient")
            auxiliary = (model.routing_aux_loss() if getattr(model, "moe_blocks", ())
                         else classification.new_zeros(()))
            loss = classification + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite supervised loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if initial_loss is None:
                initial_loss = float(classification)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total_loss += float(classification) * len(images)
            total_correct += int((logits.argmax(1) == labels).sum()); total_count += len(images)
        terminal_epoch = epoch + 1
        should_evaluate = (
            terminal_epoch % int(selection["eval_every_epochs"]) == 0
            or terminal_epoch == int(selection["maximum_epochs"]))
        row = {
            "epoch": terminal_epoch, "train_augmented_loss": total_loss / total_count,
            "train_augmented_site_top1": total_correct / total_count,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "evaluated": should_evaluate,
        }
        latest_iid = None
        if should_evaluate:
            latest_iid = simple_iid_metrics(model, loaders["iid_validation"], device)
            iid_score = float(latest_iid["site_top1"])
            row.update({"iid_site_top1": iid_score,
                        "iid_well_top1": float(latest_iid["well_top1"])})
            if iid_score > best_iid:
                best_iid, best_epoch = iid_score, terminal_epoch
                atomic_torch(best_path, {
                    "schema_version": 1, "run_id": run_id, "run_fingerprint": run_fingerprint,
                    "model": {key: value.detach().cpu()
                              for key, value in model.state_dict().items()},
                    "model_audit": model_audit, "best_epoch": best_epoch,
                    "best_source_iid_site_top1": best_iid,
                })
            meaningful = bool(
                plateau_best < 0 or iid_score >= plateau_best + float(
                    selection["minimum_delta"]))
            if meaningful:
                plateau_best, stale = iid_score, 0
            else:
                stale += 1
            row.update({"meaningful_iid_improvement": meaningful,
                        "stale_evaluations": stale})
            _save_resume(
                resume_path, run_fingerprint, model, optimizer, generator, device,
                epoch=terminal_epoch, best_iid=best_iid, best_epoch=best_epoch,
                plateau_best=plateau_best, stale_evaluations=stale,
                elapsed_seconds=elapsed_before + time.time() - started,
                initial_loss=initial_loss,
                initial_router_gradient=initial_router_gradient)
            if (terminal_epoch >= int(selection["minimum_epochs"])
                    and stale >= int(selection["patience_evaluations"])):
                stop_reason = "source_iid_plateau"
        append_jsonl(curve_path, row)
        atomic_json(status_path, {
            **frozen, "state": "training", "epoch": terminal_epoch,
            "latest": row, "latest_source_iid": latest_iid,
            "best_source_iid_site_top1": best_iid, "best_epoch": best_epoch,
            "stale_evaluations": stale,
            "elapsed_seconds": elapsed_before + time.time() - started,
            "target_state": "pixels_not_loaded",
        })
        print(f"[batch-study] {run_id} {json.dumps(row, sort_keys=True)}", flush=True)
        if stop_reason:
            break
    if best_epoch is None or not best_path.is_file():
        raise RuntimeError("no source-IID checkpoint was selected")
    if stop_reason is None:
        stop_reason = "maximum_epoch_ceiling"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"], strict=True); model.to(device)

    # Target pixels first become accessible only after the selected checkpoint is restored.
    target_loader = make_target_loader(
        registry, split, roles["target"], run_spec["batch_size"],
        run_spec["num_workers"], run_spec["image_size"])
    role_loaders = {
        "train": loaders["train_eval"], "iid_validation": loaders["iid_validation"],
        "target": target_loader,
    }
    metrics, site_rows, well_rows, embedding_rows, routing = {}, [], [], [], {}
    for role, loader in role_loaders.items():
        role_metrics, site_frame, well_frame, feature_frame, route = evaluate_detailed(
            model, loader, device, role)
        metrics[role] = role_metrics; site_rows.append(site_frame); well_rows.append(well_frame)
        if role != "train":
            embedding_rows.append(feature_frame)
        routing[role] = route
    site_predictions = pd.concat(site_rows, ignore_index=True)
    well_predictions = pd.concat(well_rows, ignore_index=True)
    embeddings = pd.concat(embedding_rows, ignore_index=True)
    for frame in (site_predictions, well_predictions, embeddings):
        frame.insert(0, "run_id", run_id)
    atomic_parquet(run_dir / "site_predictions.parquet", site_predictions)
    atomic_parquet(run_dir / "well_predictions.parquet", well_predictions)
    atomic_parquet(run_dir / "well_embeddings.parquet", embeddings)

    result = {
        **frozen, "state": "complete", "stop_reason": stop_reason,
        "terminal_epoch": terminal_epoch, "best_epoch": best_epoch,
        "best_source_iid_site_top1": best_iid,
        "training_certified": bool(metrics["train"]["site"]["top1"] >= 0.80),
        "metrics": metrics,
        "site_iid_to_target_gap": (
            metrics["iid_validation"]["site"]["top1"] - metrics["target"]["site"]["top1"]),
        "well_iid_to_target_gap": (
            metrics["iid_validation"]["well"]["top1"] - metrics["target"]["well"]["top1"]),
        "routing": routing, "initial_batch_loss": initial_loss,
        "initial_router_task_gradient": initial_router_gradient,
        "elapsed_seconds": elapsed_before + time.time() - started,
        "hostname": socket.gethostname(), "torch_version": torch.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device),
        "artifacts": {
            "checkpoint": str(best_path),
            "site_predictions": str(run_dir / "site_predictions.parquet"),
            "well_predictions": str(run_dir / "well_predictions.parquet"),
            "well_embeddings": str(run_dir / "well_embeddings.parquet"),
        },
    }
    atomic_json(result_path, result); atomic_json(status_path, result)
    print(json.dumps({
        "run_id": run_id, "state": "complete", "best_epoch": best_epoch,
        "train_site_top1": metrics["train"]["site"]["top1"],
        "iid_site_top1": metrics["iid_validation"]["site"]["top1"],
        "target_site_top1": metrics["target"]["site"]["top1"],
    }, indent=2, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        train(args.result_root, args.run_id)
    except KeyboardInterrupt as error:
        root = Path(args.result_root).expanduser().resolve()
        atomic_json(root / "runs" / args.run_id / "STATUS.json", {
            "state": "interrupted", "run_id": args.run_id, "error": repr(error),
            "updated_at": time.time(),
        })
        raise
    except Exception as error:
        root = Path(args.result_root).expanduser().resolve()
        atomic_json(root / "runs" / args.run_id / "STATUS.json", {
            "state": "failed", "run_id": args.run_id, "error": repr(error),
            "updated_at": time.time(),
        })
        raise


if __name__ == "__main__":
    main()
