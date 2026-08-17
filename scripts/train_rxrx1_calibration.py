#!/usr/bin/env python
"""DDP-capable published-strength RxRx1 calibration and cross-fit runner."""
from __future__ import annotations

import argparse
import contextlib
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
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.data.rxrx1_huvec import Native6SiteDataset
from moe_shift.models.rxrx1_calibration import build_rxrx1_calibration_model


STOP_REQUESTED = False


def _request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def atomic_torch(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_identity() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).strip())
    return commit, dirty


class EvenTrainSampler(Sampler[int]):
    """Equal-length distributed sharding with no padded/duplicated samples."""

    def __init__(self, size: int, rank: int, world_size: int, seed: int):
        self.size, self.rank, self.world_size = size, rank, world_size
        self.seed, self.epoch = seed, 0
        self.total = (size // world_size) * world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.total // self.world_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator)[:self.total]
        return iter(indices[self.rank:self.total:self.world_size].tolist())


class ExactEvalSampler(Sampler[int]):
    """Non-padding evaluation shards; every sample is seen exactly once."""

    def __init__(self, size: int, rank: int, world_size: int):
        self.indices = list(range(rank, size, world_size))

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


def setup_distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device("cuda", local_rank)


def reduce_triplet(loss_sum: float, correct: int, count: int, device):
    values = torch.tensor([loss_sum, correct, count], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(values)
    return tuple(values.cpu().tolist())


def barrier():
    if dist.is_initialized():
        dist.barrier()


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def frame_roles(manifest: Path, split: str):
    frame = pd.read_parquet(manifest)
    if split == "official":
        if "dataset" not in frame:
            raise ValueError("official mode requires the all-sites dataset column")
        roles = {
            "train": frame[frame.dataset.eq("train")].copy(),
            "target": frame[frame.dataset.eq("test")].copy(),
        }
    else:
        required = {"train", "selection_validation", "target"}
        observed = set(map(str, frame.role.unique()))
        if not required <= observed:
            raise ValueError(f"custom assignment roles are {sorted(observed)}")
        roles = {role: frame[frame.role.eq(role)].copy() for role in required}
    for role, rows in roles.items():
        if rows.empty:
            raise ValueError(f"{role} split is empty")
        if rows.groupby("well_id").site.nunique().max() > 2:
            raise ValueError(f"{role} contains a well with more than two sites")
    train_wells = set(roles["train"].well_id.astype(str))
    for role in ("selection_validation", "target"):
        if role in roles and train_wells & set(roles[role].well_id.astype(str)):
            raise ValueError(f"well leakage from train into {role}")
    return roles


def make_dataset(frame, raw_root, image_size, train, normalization):
    return Native6SiteDataset(
        frame, raw_root, image_size, [0.0] * 6, [1.0] * 6,
        train=train, normalization_mode=normalization,
        vertical_flip=train)


def make_loaders(roles, args, rank, world):
    loaders, samplers = {}, {}
    for role, frame in roles.items():
        training = role == "train"
        dataset = make_dataset(
            frame, args.raw_root, args.image_size, training, args.normalization)
        sampler = (EvenTrainSampler(len(dataset), rank, world, args.seed)
                   if training else ExactEvalSampler(len(dataset), rank, world))
        loaders[role] = DataLoader(
            dataset, batch_size=args.per_gpu_batch, sampler=sampler,
            num_workers=args.workers, pin_memory=True, drop_last=training,
            persistent_workers=args.workers > 0)
        samplers[role] = sampler
    return loaders, samplers


def cutmix(images, labels, alpha: float):
    if alpha <= 0 or len(images) < 2:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(images), device=images.device)
    height, width = images.shape[-2:]
    ratio = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(width * ratio), int(height * ratio)
    center_x = int(torch.randint(width, (), device=images.device))
    center_y = int(torch.randint(height, (), device=images.device))
    x1, x2 = max(center_x - cut_w // 2, 0), min(center_x + cut_w // 2, width)
    y1, y2 = max(center_y - cut_h // 2, 0), min(center_y + cut_h // 2, height)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, labels, labels[permutation], lam


def learning_rate(epoch: int, args, effective_batch: int) -> float:
    base = args.learning_rate * effective_batch / 512.0
    if epoch < args.warmup_epochs:
        return base * (epoch + 1) / max(args.warmup_epochs, 1)
    progress = (epoch - args.warmup_epochs) / max(
        args.epochs - args.warmup_epochs - 1, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


@torch.inference_mode()
def evaluate_scalar(model, loader, device, amp_dtype):
    model.eval()
    loss_sum = 0.0; correct = 0; count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
        loss_sum += float(loss); correct += int((logits.argmax(1) == labels).sum())
        count += len(images)
    loss_sum, correct, count = reduce_triplet(loss_sum, correct, count, device)
    return {"loss": loss_sum / count, "site_top1": correct / count,
            "n_sites": int(count)}


def _part_path(run_dir: Path, role: str, rank: int) -> Path:
    return run_dir / "prediction_parts" / f"{role}.rank{rank}.npz"


@torch.inference_mode()
def evaluate_detailed(model, loader, device, amp_dtype, run_dir, role, rank, world):
    model.eval()
    logits_rows, labels_rows, index_rows, experiment_rows = [], [], [], []
    well_rows, cell_rows = [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(images)
        logits_rows.append(logits.float().cpu().to(torch.float16).numpy())
        labels_rows.append(torch.as_tensor(batch["label"]).numpy())
        index_rows.append(torch.as_tensor(batch["global_index"]).numpy())
        experiment_rows.append(torch.as_tensor(batch["experiment"]).numpy())
        well_rows.extend(map(str, batch["well_id"]))
        cell_rows.extend(map(str, batch.get("cell_type", [""] * len(images))))
    part = _part_path(run_dir, role, rank)
    part.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        part, logits=np.concatenate(logits_rows), labels=np.concatenate(labels_rows),
        global_index=np.concatenate(index_rows), experiment=np.concatenate(experiment_rows),
        well_id=np.asarray(well_rows), cell_type=np.asarray(cell_rows))
    barrier()
    if rank != 0:
        return None
    pieces = [np.load(_part_path(run_dir, role, other), allow_pickle=False)
              for other in range(world)]
    logits = np.concatenate([piece["logits"] for piece in pieces]).astype(np.float32)
    labels = np.concatenate([piece["labels"] for piece in pieces]).astype(np.int64)
    indices = np.concatenate([piece["global_index"] for piece in pieces]).astype(np.int64)
    experiments = np.concatenate([piece["experiment"] for piece in pieces]).astype(np.int64)
    wells = np.concatenate([piece["well_id"] for piece in pieces]).astype(str)
    cells = np.concatenate([piece["cell_type"] for piece in pieces]).astype(str)
    order = np.argsort(indices)
    logits, labels, indices = logits[order], labels[order], indices[order]
    experiments, wells, cells = experiments[order], wells[order], cells[order]
    tensor_logits = torch.from_numpy(logits)
    prediction = logits.argmax(1)
    nll = F.cross_entropy(tensor_logits, torch.from_numpy(labels), reduction="none").numpy()
    site_frame = pd.DataFrame({
        "role": role, "global_index": indices, "well_id": wells,
        "experiment": experiments, "cell_type": cells, "label": labels,
        "prediction": prediction, "nll": nll,
        "correct": prediction == labels,
    })
    site_frame.to_parquet(run_dir / f"{role}_site_predictions.parquet", index=False)
    well_records = []
    for well_id, positions in site_frame.groupby("well_id", sort=False).indices.items():
        positions = np.asarray(positions)
        if len(np.unique(labels[positions])) != 1:
            raise RuntimeError(f"sites in well {well_id} disagree on label")
        mean_logits = logits[positions].mean(0)
        pred = int(mean_logits.argmax())
        well_records.append({
            "role": role, "well_id": well_id,
            "experiment": int(experiments[positions[0]]),
            "cell_type": str(cells[positions[0]]),
            "label": int(labels[positions[0]]), "prediction": pred,
            "correct": pred == int(labels[positions[0]]), "n_sites": len(positions),
        })
    well_frame = pd.DataFrame(well_records)
    well_frame.to_parquet(run_dir / f"{role}_well_predictions.parquet", index=False)
    per_experiment = []
    for experiment, rows in site_frame.groupby("experiment", sort=True):
        well_subset = well_frame[well_frame.experiment.eq(experiment)]
        per_experiment.append({
            "experiment": int(experiment), "n_sites": int(len(rows)),
            "site_top1": float(rows.correct.mean()), "n_wells": int(len(well_subset)),
            "well_top1": float(well_subset.correct.mean()),
        })
    metrics = {
        "n_sites": int(len(site_frame)), "site_top1": float(site_frame.correct.mean()),
        "n_wells": int(len(well_frame)), "well_top1": float(well_frame.correct.mean()),
        "mean_nll": float(site_frame.nll.mean()), "per_experiment": per_experiment,
    }
    for piece in pieces:
        piece.close()
    for other in range(world):
        _part_path(run_dir, role, other).unlink(missing_ok=True)
    return metrics


@torch.inference_mode()
def adapt_batch_norm(model, loader, device, amp_dtype):
    """One unlabeled target pass; only BatchNorm running statistics change."""
    module = unwrap(model)
    module.train()
    for child in module.modules():
        if not isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            child.training = False
    local_batches = torch.tensor(len(loader), dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(local_batches, op=dist.ReduceOp.MIN)
    for batch_index, batch in enumerate(loader):
        if batch_index >= int(local_batches):
            break
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            model(images)


def rng_state():
    return {
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="official")
    parser.add_argument("--model", choices=("densenet161", "resnet50", "vit_small"),
                        default="densenet161")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=1139)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1024)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--per-gpu-batch", type=int, default=16)
    parser.add_argument("--effective-batch", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--normalization", choices=("per_image", "frozen_global"),
                        default="per_image")
    parser.add_argument("--selection-every", type=int, default=5)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--memory-efficient", action="store_true")
    parser.add_argument("--adabn", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGUSR1, _request_stop)
    rank, world, local_rank, device = setup_distributed()
    is_main = rank == 0
    random.seed(args.seed + rank); np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank); torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    run_dir = Path(args.result_root).expanduser().resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path, status_path = run_dir / "RESULT.json", run_dir / "STATUS.json"
    if result_path.is_file():
        if is_main:
            print(f"[skip] complete {result_path}", flush=True)
        if dist.is_initialized(): dist.destroy_process_group()
        return
    commit, dirty = git_identity()
    if dirty:
        raise RuntimeError("tracked checkout is dirty")
    manifest = Path(args.manifest).expanduser().resolve()
    roles = frame_roles(manifest, args.split)
    loaders, samplers = make_loaders(roles, args, rank, world)
    micro_global = args.per_gpu_batch * world
    if args.effective_batch % micro_global:
        raise ValueError(
            f"effective batch {args.effective_batch} is not divisible by "
            f"per_gpu_batch*world_size={micro_global}")
    accumulation = args.effective_batch // micro_global
    effective_batch = micro_global * accumulation
    config = {
        **vars(args), "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "raw_root": str(Path(args.raw_root).expanduser().resolve()),
        "git_commit": commit, "world_size": world, "per_gpu_batch": args.per_gpu_batch,
        "gradient_accumulation": accumulation, "effective_global_batch": effective_batch,
        "amp_dtype": str(amp_dtype), "hostname": socket.gethostname(),
        "role_inventory": {role: {"n_sites": int(len(rows)),
                                   "n_wells": int(rows.well_id.nunique()),
                                   "n_experiments": int(rows.experiment_name.nunique())}
                           for role, rows in roles.items()},
    }
    fingerprint = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    frozen = {"schema_version": 1, "fingerprint": fingerprint, "config": config}
    if is_main:
        frozen_path = run_dir / "FROZEN_RUN.json"
        if frozen_path.is_file():
            if json.loads(frozen_path.read_text())["fingerprint"] != fingerprint:
                raise RuntimeError("frozen run configuration changed")
        else:
            atomic_json(frozen_path, frozen)

    model = build_rxrx1_calibration_model(
        args.model, args.num_classes, args.image_size, args.pretrained,
        args.memory_efficient).to(device)
    if world > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum,
        weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    resume_path = run_dir / "resume.pt"
    start_epoch = 0; best_score = -1.0; best_epoch = None; elapsed_before = 0.0
    if resume_path.is_file():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise RuntimeError("resume checkpoint belongs to a different frozen run")
        unwrap(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]); best_score = float(state["best_score"])
        best_epoch = state["best_epoch"]; elapsed_before = float(state["elapsed_seconds"])
        restore_rng(state["rng_by_rank"][rank])
    barrier()
    curve_path = run_dir / "curves.jsonl"
    if is_main and curve_path.is_file():
        rows = [json.loads(line) for line in curve_path.read_text().splitlines() if line]
        curve_path.write_text("".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows if int(row["epoch"]) <= start_epoch))
    started = time.time(); terminal_epoch = start_epoch; stop_reason = None
    for epoch in range(start_epoch, args.epochs):
        samplers["train"].set_epoch(epoch)
        lr = learning_rate(epoch, args, effective_batch)
        for group in optimizer.param_groups: group["lr"] = lr
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0; correct = 0; count = 0; optimizer_steps = 0
        batches = len(loaders["train"])
        usable_batches = (batches // accumulation) * accumulation
        for batch_index, batch in enumerate(loaders["train"]):
            if batch_index >= usable_batches: break
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            images, labels_a, labels_b, lam = cutmix(images, labels, args.cutmix_alpha)
            synchronization = ((batch_index + 1) % accumulation == 0)
            sync_context = (contextlib.nullcontext() if synchronization or world == 1
                            else model.no_sync())
            with sync_context:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(images)
                    loss = (lam * F.cross_entropy(logits, labels_a)
                            + (1.0 - lam) * F.cross_entropy(logits, labels_b))
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
            if synchronization:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
                if args.smoke_steps and optimizer_steps >= args.smoke_steps:
                    break
            loss_sum += float(loss) * len(images)
            correct += int((logits.argmax(1) == labels).sum()); count += len(images)
        loss_sum, correct, count = reduce_triplet(loss_sum, correct, count, device)
        terminal_epoch = epoch + 1
        row = {"epoch": terminal_epoch, "learning_rate": lr,
               "train_augmented_loss": loss_sum / count,
               "train_augmented_site_top1": correct / count,
               "optimizer_steps": optimizer_steps}
        selection = None
        if "selection_validation" in loaders and (
                terminal_epoch % args.selection_every == 0 or terminal_epoch == args.epochs):
            selection = evaluate_scalar(model, loaders["selection_validation"], device, amp_dtype)
            row.update({f"selection_{key}": value for key, value in selection.items()})
            if selection["site_top1"] > best_score:
                best_score, best_epoch = selection["site_top1"], terminal_epoch
                if is_main:
                    atomic_torch(run_dir / "best.pt", {
                        "fingerprint": fingerprint, "epoch": best_epoch,
                        "score": best_score, "model": unwrap(model).state_dict()})
        elapsed = elapsed_before + time.time() - started
        local_rng = rng_state()
        if dist.is_initialized():
            gathered_rng = [None] * world if is_main else None
            dist.gather_object(local_rng, gathered_rng, dst=0)
        else:
            gathered_rng = [local_rng]
        if is_main:
            with open(curve_path, "a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n"); handle.flush()
            checkpoint = {
                "fingerprint": fingerprint, "epoch": terminal_epoch,
                "model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "best_score": best_score,
                "best_epoch": best_epoch, "elapsed_seconds": elapsed,
                "rng_by_rank": gathered_rng,
            }
            atomic_torch(resume_path, checkpoint)
            atomic_json(status_path, {**frozen, "state": "training", "latest": row,
                                      "best_score": best_score, "best_epoch": best_epoch,
                                      "elapsed_seconds": elapsed})
            print(f"[rxrx1-calibration] {json.dumps(row, sort_keys=True)}", flush=True)
        barrier()
        if args.smoke_steps:
            stop_reason = "smoke_complete"; break
        if STOP_REQUESTED:
            stop_reason = "scheduler_signal"; break
    if stop_reason == "scheduler_signal":
        if is_main:
            atomic_json(status_path, {**frozen, "state": "interrupted_resumable",
                                      "epoch": terminal_epoch})
        barrier()
        if dist.is_initialized(): dist.destroy_process_group()
        return
    if args.smoke_steps:
        if is_main:
            atomic_json(result_path, {**frozen, "state": "smoke_complete",
                                      "epoch": terminal_epoch})
        barrier()
        if dist.is_initialized(): dist.destroy_process_group()
        return

    if best_epoch is not None:
        barrier()
        best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(best["model"])
    metrics = {}
    for role, loader in loaders.items():
        metrics[role] = evaluate_detailed(
            model, loader, device, amp_dtype, run_dir, role, rank, world)
        barrier()
    if args.adabn:
        adapt_batch_norm(model, loaders["target"], device, amp_dtype)
        metrics["target_adabn"] = evaluate_detailed(
            model, loaders["target"], device, amp_dtype, run_dir,
            "target_adabn", rank, world)
        barrier()
    if is_main:
        elapsed = elapsed_before + time.time() - started
        result = {**frozen, "state": "complete", "terminal_epoch": terminal_epoch,
                  "selected_epoch": best_epoch or terminal_epoch,
                  "selection_site_top1": best_score if best_epoch is not None else None,
                  "metrics": metrics, "elapsed_seconds": elapsed,
                  "target_used_for_selection": False,
                  "adabn_uses_unlabeled_target_pixels": bool(args.adabn)}
        atomic_json(result_path, result); atomic_json(status_path, result)
        print(json.dumps({"state": "complete", "run": args.run_name,
                          "metrics": metrics}, indent=2, sort_keys=True), flush=True)
    barrier()
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    main()
