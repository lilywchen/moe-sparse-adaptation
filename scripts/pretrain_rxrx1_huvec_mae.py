#!/usr/bin/env python
"""Standalone, split-sealed MAE pretraining for RxRx1 HUVEC.

This entry point intentionally does not perform perturbation-supervised training.  It consumes
one frozen HUVEC registry, rebuilds its audited experiment/well split, and then uses only the
registry's ``train`` role.  A deterministic well-level subset of that role is reserved for MAE
reconstruction validation.  Source-IID and target sites are never placed in a Dataset.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.data.rxrx1 import _native_channel_paths
from moe_shift.data.rxrx1_huvec import (
    EXPECTED_TREATMENTS,
    Native6SiteDataset,
    deterministic_split,
)
from moe_shift.models.huvec import MaskedAutoencoder, build_study_model, parameter_count


VALID_MODELS = ("vit_micro", "vit_tiny")
VALIDATION_MASK_SEED = 2026081401
_STOP_REQUESTED = False


def _request_stop(signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"[signal] received {signum}; checkpointing after the current batch", flush=True)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_hash(frame, columns):
    ordered = frame[list(columns)].sort_values(list(columns)).reset_index(drop=True)
    return hashlib.sha256(ordered.to_csv(index=False).encode()).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _torch_load(path):
    """Load our own full-state checkpoints across the PyTorch 2.6 weights-only change."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only keyword.
        return torch.load(path, map_location="cpu")


def _append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git_info():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True,
    ).strip())
    return sha, dirty


def _stable_token(*values):
    return hashlib.sha256("|".join(map(str, values)).encode()).hexdigest()


def partition_mae_wells(train_sites, validation_fraction=0.10, seed=0):
    """Deterministically reserve wells within every source experiment for MAE validation."""
    fraction = float(validation_fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("MAE validation fraction must be in (0, 0.5)")
    wells = train_sites.drop_duplicates("well_id").copy()
    validation_wells = set()
    for experiment, rows in wells.groupby("experiment", sort=True):
        ranked = sorted(
            (_stable_token("mae-validation", seed, experiment, row.well_id), row.well_id)
            for row in rows.itertuples(index=False)
        )
        count = min(max(1, int(round(len(ranked) * fraction))), len(ranked) - 1)
        if count < 1:
            raise ValueError(f"source experiment {experiment} has too few training wells")
        validation_wells.update(well_id for _, well_id in ranked[:count])
    partitioned = train_sites.copy()
    partitioned["mae_role"] = "mae_train"
    partitioned.loc[partitioned.well_id.isin(validation_wells), "mae_role"] = "mae_validation"
    roles_per_well = partitioned.groupby("well_id").mae_role.nunique()
    if int(roles_per_well.max()) != 1:
        raise ValueError("a well crosses MAE train/validation roles")
    for role in ("mae_train", "mae_validation"):
        if partitioned[partitioned.mae_role == role].empty:
            raise ValueError(f"empty {role} partition")
    return partitioned.sort_values(["mae_role", "experiment", "well_id", "site"])


def load_sealed_partition(registry_path, site_manifest_path, split_id,
                          validation_fraction=0.10, seed=0):
    registry_path = Path(registry_path).expanduser().resolve()
    site_manifest_path = Path(site_manifest_path).expanduser().resolve()
    registry = json.loads(registry_path.read_text())
    candidates = registry.get("main_training_splits") or registry.get("primary_splits") or []
    matches = [row for row in candidates if row["split_id"] == split_id]
    if len(matches) != 1:
        raise ValueError(f"split {split_id!r} occurs {len(matches)} times in frozen registry")
    split_spec = matches[0]
    sites = pd.read_parquet(site_manifest_path)
    assignment = deterministic_split(
        sites, split_spec["source_experiments"], split_spec["target_experiments"],
        split_spec["split_id"],
    )
    source_train = assignment[assignment.role == "train"].copy()
    source_iid = assignment[assignment.role == "iid_validation"]
    target = assignment[assignment.role == "target"]
    partitioned = partition_mae_wells(
        source_train, validation_fraction=validation_fraction, seed=seed)

    loaded_indices = set(map(int, partitioned.global_index))
    if loaded_indices & set(map(int, source_iid.global_index)):
        raise ValueError("source-IID site leaked into MAE partition")
    if loaded_indices & set(map(int, target.global_index)):
        raise ValueError("target site leaked into MAE partition")
    observed_experiments = set(map(int, partitioned.experiment.unique()))
    expected_experiments = set(map(int, split_spec["source_experiments"]))
    if observed_experiments != expected_experiments:
        raise ValueError("MAE partition does not cover exactly the frozen source experiments")
    target_experiments = set(map(int, split_spec["target_experiments"]))
    if observed_experiments & target_experiments:
        raise ValueError("target experiment leaked into MAE partition")

    role_counts = {}
    for role, rows in partitioned.groupby("mae_role", sort=True):
        role_counts[str(role)] = {
            "sites": len(rows), "wells": int(rows.well_id.nunique()),
            "experiments": sorted(map(int, rows.experiment.unique())),
        }
    audit = {
        "registry": str(registry_path), "registry_sha256": _sha256(registry_path),
        "site_manifest": str(site_manifest_path),
        "site_manifest_sha256": _sha256(site_manifest_path),
        "split_id": split_id,
        "source_experiments": sorted(expected_experiments),
        "target_experiments": sorted(target_experiments),
        "source_iid_sites_excluded": len(source_iid),
        "target_sites_excluded": len(target),
        "validation_fraction": float(validation_fraction),
        "partition_seed": int(seed), "role_counts": role_counts,
        "assignment_sha256": _frame_hash(
            assignment, ("global_index", "well_id", "experiment", "role")),
        "mae_partition_sha256": _frame_hash(
            partitioned, ("global_index", "well_id", "experiment", "mae_role")),
    }
    return registry, split_spec, partitioned, audit


def audit_raw_paths(partitioned, raw_root):
    """Verify every loaded channel and physically exclude all target experiment folders."""
    raw_root = Path(raw_root).expanduser().resolve()
    expected_source_folders = set()
    missing = []
    for row in partitioned.itertuples(index=False):
        relative = Path(str(row.relative_path))
        if len(relative.parts) < 2:
            raise ValueError(f"unexpected RxRx1 relative path: {relative}")
        expected_source_folders.add(relative.parts[1])
        for path in _native_channel_paths(raw_root, relative):
            if not path.is_file():
                missing.append(str(path))
                if len(missing) >= 10:
                    break
        if len(missing) >= 10:
            break
    if missing:
        raise FileNotFoundError(f"missing sealed MAE channels; first paths: {missing}")
    image_root = raw_root / "images"
    available = {path.name for path in image_root.glob("HUVEC-*") if path.is_dir()}
    unexpected = available - expected_source_folders
    missing_folders = expected_source_folders - available
    if unexpected:
        raise ValueError(f"raw root contains non-source HUVEC folders: {sorted(unexpected)}")
    if missing_folders:
        raise FileNotFoundError(f"raw root lacks source HUVEC folders: {sorted(missing_folders)}")
    return {
        "raw_root": str(raw_root), "source_folders": sorted(expected_source_folders),
        "available_huvec_folders": sorted(available),
        "audited_sites": len(partitioned), "audited_channel_files": len(partitioned) * 6,
        "target_folders_physically_present": [],
    }


def _make_loaders(partitioned, raw_root, normalization, image_size, batch_size,
                  workers, seed):
    frames = {
        role: partitioned[partitioned.mae_role == role].copy()
        for role in ("mae_train", "mae_validation")
    }
    datasets = {
        "train": Native6SiteDataset(
            frames["mae_train"], raw_root, image_size,
            normalization["mean"], normalization["std"], train=True),
        "validation": Native6SiteDataset(
            frames["mae_validation"], raw_root, image_size,
            normalization["mean"], normalization["std"], train=False),
    }
    generator = torch.Generator().manual_seed(int(seed))
    common = {
        "batch_size": int(batch_size), "num_workers": int(workers),
        "pin_memory": True, "drop_last": False,
        "persistent_workers": int(workers) > 0,
    }
    if int(workers) > 0:
        common["prefetch_factor"] = 2
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
        "validation": DataLoader(datasets["validation"], shuffle=False, **common),
    }
    return loaders, generator


def _learning_rate(epoch, max_epochs, base_lr, warmup_epochs):
    if epoch < int(warmup_epochs):
        return float(base_lr) * float(epoch + 1) / max(int(warmup_epochs), 1)
    progress = (epoch - int(warmup_epochs)) / max(
        int(max_epochs) - int(warmup_epochs) - 1, 1)
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _set_lr(optimizer, value):
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def _optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _rng_state(generator):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": generator.get_state(),
    }


def _restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader_generator"])


def _checkpoint_payload(mae, optimizer, generator, config, audit, history,
                        epoch_completed, best_val, patience_best, stale_epochs,
                        elapsed_seconds, preempted_mid_epoch=False):
    return {
        "schema_version": 1, "model": mae.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": _rng_state(generator), "config": config, "audit": audit,
        "history": history, "epoch_completed": int(epoch_completed),
        "best_validation_loss": float(best_val), "patience_best": float(patience_best),
        "stale_epochs": int(stale_epochs), "elapsed_seconds": float(elapsed_seconds),
        "preempted_mid_epoch": bool(preempted_mid_epoch),
    }


def _save_encoder(path, mae, config, audit, epoch, validation_loss):
    _atomic_torch(path, {
        "schema_version": 1,
        "encoder": copy.deepcopy({key: value.detach().cpu()
                                  for key, value in mae.encoder.state_dict().items()}),
        "config": config, "audit": audit, "epoch": int(epoch),
        "validation_reconstruction_loss": float(validation_loss),
    })


@torch.no_grad()
def _validate(mae, loader, device, amp_dtype):
    mae.eval()
    total = count = 0
    devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(VALIDATION_MASK_SEED)
        torch.cuda.manual_seed_all(VALIDATION_MASK_SEED)
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                reconstruction, auxiliary = mae(images)
                loss = reconstruction + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite MAE validation loss")
            total += float(reconstruction) * len(images)
            count += len(images)
    return total / max(count, 1)


def pretrain(args):
    if args.model not in VALID_MODELS:
        raise ValueError(f"model must be one of {VALID_MODELS}")
    if not torch.cuda.is_available():
        raise RuntimeError("standalone MAE pretraining requires a CUDA GPU")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    git_commit, git_dirty = _git_info()
    if git_dirty and not args.allow_dirty:
        raise RuntimeError("refusing MAE pretraining from a dirty tracked checkout")

    registry, split_spec, partitioned, split_audit = load_sealed_partition(
        args.registry, args.site_manifest, args.split_id,
        validation_fraction=args.validation_fraction, seed=args.seed)
    path_audit = audit_raw_paths(partitioned, args.raw_root)
    audit = {**split_audit, "raw_paths": path_audit}
    normalization = split_spec["normalization"]

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    loaders, generator = _make_loaders(
        partitioned, args.raw_root, normalization, args.image_size,
        args.batch_size, args.workers, args.seed)
    encoder, model_audit = build_study_model(
        args.model, num_classes=EXPECTED_TREATMENTS, image_size=args.image_size)
    for parameter in encoder.fc.parameters():
        parameter.requires_grad_(False)
    mae = MaskedAutoencoder(
        encoder, mask_ratio=args.mask_ratio, decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth, decoder_heads=args.decoder_heads).to(device)
    trainable = [parameter for parameter in mae.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.base_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    config = {
        "model": args.model, "image_size": args.image_size,
        "mask_ratio": args.mask_ratio, "decoder_dim": args.decoder_dim,
        "decoder_depth": args.decoder_depth, "decoder_heads": args.decoder_heads,
        "batch_size": args.batch_size, "workers": args.workers,
        "base_lr": args.base_lr, "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs, "min_epochs": args.min_epochs,
        "warmup_epochs": args.warmup_epochs, "patience": args.patience,
        "min_delta": args.min_delta, "seed": args.seed,
        "max_train_steps": args.max_train_steps,
        "split_id": args.split_id, "validation_fraction": args.validation_fraction,
        "git_commit": git_commit, "git_dirty": git_dirty,
    }
    audit.update({
        "model": model_audit, "mae_total_params": parameter_count(mae),
        "mae_trainable_params": sum(value.numel() for value in trainable),
        "classifier_head_frozen": True, "validation_mask_seed": VALIDATION_MASK_SEED,
        "normalization": normalization,
    })
    _atomic_json(output / "FROZEN_RUN.json", {"config": config, "audit": audit})

    # Inspect a deterministic validation item directly.  Avoid starting persistent DataLoader
    # workers before a resume checkpoint restores its generator state.
    first_image = loaders["validation"].dataset[0]["image"].unsqueeze(0)
    if tuple(first_image.shape[1:]) != (6, args.image_size, args.image_size):
        raise ValueError(f"six-channel loader contract failed: {tuple(first_image.shape)}")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        initial_reconstruction, _ = mae(first_image.to(device))
    if not torch.isfinite(initial_reconstruction):
        raise FloatingPointError("non-finite initial reconstruction loss")

    last_path = output / "last.pt"
    history = []
    start_epoch = 0
    best_val = float("inf")
    patience_best = float("inf")
    stale_epochs = 0
    elapsed_before = 0.0
    if args.resume and last_path.is_file():
        state = _torch_load(last_path)
        if state["config"] != config:
            raise ValueError("resume checkpoint configuration differs from frozen run")
        if state["audit"]["mae_partition_sha256"] != audit["mae_partition_sha256"]:
            raise ValueError("resume checkpoint MAE partition differs from current manifest")
        mae.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"]); _optimizer_to(optimizer, device)
        _restore_rng(state["rng"], generator)
        history = list(state["history"])
        start_epoch = int(state["epoch_completed"]) + 1
        best_val = float(state["best_validation_loss"])
        patience_best = float(state["patience_best"])
        stale_epochs = int(state["stale_epochs"])
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        print(f"[resume] completed_epochs={start_epoch} best_val={best_val:.6f}", flush=True)

    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    started = time.time()
    stop_reason = "max_epochs"
    total_steps = 0
    for epoch in range(start_epoch, int(args.max_epochs)):
        lr = _learning_rate(epoch, args.max_epochs, args.base_lr, args.warmup_epochs)
        _set_lr(optimizer, lr)
        mae.train(); total = count = 0
        for batch_index, batch in enumerate(loaders["train"]):
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                reconstruction, auxiliary = mae(images)
                loss = reconstruction + auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite MAE training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == start_epoch and batch_index == 0:
                gradients = [p.grad for p in trainable if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("missing or non-finite MAE gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad_norm)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite MAE gradient norm")
            optimizer.step()
            total += float(reconstruction) * len(images); count += len(images)
            total_steps += 1
            if total_steps == 1 or total_steps % int(args.log_every_steps) == 0:
                print(
                    f"[train] model={args.model} epoch={epoch + 1}/{args.max_epochs} "
                    f"batch={batch_index + 1}/{len(loaders['train'])} "
                    f"loss={total / count:.6f} lr={lr:.3e}", flush=True)
            if _STOP_REQUESTED:
                payload = _checkpoint_payload(
                    mae, optimizer, generator, config, audit, history, epoch - 1,
                    best_val, patience_best, stale_epochs,
                    elapsed_before + time.time() - started, preempted_mid_epoch=True)
                _atomic_torch(last_path, payload)
                _atomic_json(output / "STATUS.json", {
                    "state": "requeue_requested", "model": args.model,
                    "epoch": epoch + 1, "batch": batch_index + 1,
                    "checkpoint": str(last_path),
                })
                return 99
            if args.max_train_steps and total_steps >= int(args.max_train_steps):
                break

        train_loss = total / max(count, 1)
        validation_loss = _validate(mae, loaders["validation"], device, amp_dtype)
        row = {
            "epoch": epoch + 1, "train_reconstruction_loss": train_loss,
            "validation_reconstruction_loss": validation_loss,
            "learning_rate": lr, "train_steps": total_steps,
            "elapsed_seconds": elapsed_before + time.time() - started,
        }
        history.append(row); _append_jsonl(output / "curves.jsonl", row)
        improved = validation_loss < best_val
        if improved:
            best_val = validation_loss
            _save_encoder(output / "best_encoder.pt", mae, config, audit,
                          epoch + 1, validation_loss)
        if validation_loss < patience_best - float(args.min_delta):
            patience_best = validation_loss; stale_epochs = 0
        else:
            stale_epochs += 1
        if (epoch + 1) % int(args.encoder_checkpoint_every) == 0:
            _save_encoder(
                output / f"encoder_epoch{epoch + 1:04d}.pt", mae, config, audit,
                epoch + 1, validation_loss)
        payload = _checkpoint_payload(
            mae, optimizer, generator, config, audit, history, epoch,
            best_val, patience_best, stale_epochs,
            elapsed_before + time.time() - started)
        _atomic_torch(last_path, payload)
        _atomic_json(output / "STATUS.json", {
            "state": "training", "model": args.model, "epoch": epoch + 1,
            "train_reconstruction_loss": train_loss,
            "validation_reconstruction_loss": validation_loss,
            "best_validation_reconstruction_loss": best_val,
            "stale_epochs": stale_epochs, "patience": args.patience,
            "checkpoint": str(last_path),
        })
        print(f"[epoch] {json.dumps(row, sort_keys=True)}", flush=True)

        if args.max_train_steps and total_steps >= int(args.max_train_steps):
            stop_reason = "max_train_steps"; break
        if epoch + 1 >= int(args.min_epochs) and stale_epochs >= int(args.patience):
            stop_reason = "validation_plateau"; break

    best_payload = _torch_load(output / "best_encoder.pt")
    best_epoch = int(best_payload["epoch"])
    result = {
        "schema_version": 1, "state": "complete", "model": args.model,
        "stop_reason": stop_reason, "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_reconstruction_loss": float(best_val),
        "initial_reconstruction_loss": float(initial_reconstruction),
        "elapsed_seconds": elapsed_before + time.time() - started,
        "config": config, "audit": audit, "hostname": socket.gethostname(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "best_encoder": str(output / "best_encoder.pt"),
        "last_checkpoint": str(last_path),
    }
    _atomic_json(output / "RESULT.json", result)
    _atomic_json(output / "STATUS.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--site-manifest", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-id", default="primary_fold0")
    parser.add_argument("--model", choices=VALID_MODELS, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--decoder-depth", type=int, default=2)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-checkpoint-every", type=int, default=10)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    raise SystemExit(pretrain(args))


if __name__ == "__main__":
    main()
