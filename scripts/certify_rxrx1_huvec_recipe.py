#!/usr/bin/env python
"""Certify a full-data RxRx1 HUVEC training recipe before model comparisons.

This is deliberately different from the small memorization canary.  It uses the complete
``primary_fold0`` source training set, the production image augmentation, and the frozen
source-IID split.  A checkpoint is eligible only after its full, unaugmented source-training
accuracy clears the requested threshold.  Among eligible checkpoints, source-IID accuracy
selects the checkpoint.  Target-batch images are never loaded or evaluated here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.data.rxrx1_huvec import EXPECTED_TREATMENTS
from moe_shift.models.huvec import build_study_model
from scripts.run_rxrx1_huvec_study import (
    _atomic_json,
    _atomic_torch,
    _git_info,
    _make_loaders,
    _split_hash,
    evaluate,
)


def default_recipes(model):
    """Return a bounded, ordered ladder of production-compatible recipes."""
    if model == "resnet18":
        return [
            {
                "name": "adamw_standard_extended", "optimizer": "adamw",
                "lr": 1e-3, "weight_decay": 1e-4, "max_epochs": 120,
                "warmup_epochs": 5, "min_lr_ratio": 0.02, "augmentation": True,
            },
            {
                "name": "adamw_low_regularization", "optimizer": "adamw",
                "lr": 1e-3, "weight_decay": 1e-5, "max_epochs": 160,
                "warmup_epochs": 5, "min_lr_ratio": 0.02, "augmentation": True,
            },
            {
                "name": "sgd_low_regularization", "optimizer": "sgd",
                "lr": 5e-2, "weight_decay": 1e-4, "momentum": 0.9,
                "max_epochs": 180, "warmup_epochs": 5, "min_lr_ratio": 0.01,
                "augmentation": True,
            },
        ]
    if model == "vit_tiny":
        return [
            {
                "name": "adamw_standard_extended", "optimizer": "adamw",
                "lr": 7.5e-4, "weight_decay": 0.05, "max_epochs": 160,
                "warmup_epochs": 5, "min_lr_ratio": 0.02, "augmentation": True,
            },
            {
                "name": "adamw_low_regularization", "optimizer": "adamw",
                "lr": 1e-3, "weight_decay": 0.01, "max_epochs": 200,
                "warmup_epochs": 8, "min_lr_ratio": 0.02, "augmentation": True,
            },
            {
                "name": "adamw_weak_regularization", "optimizer": "adamw",
                "lr": 5e-4, "weight_decay": 1e-3, "max_epochs": 240,
                "warmup_epochs": 8, "min_lr_ratio": 0.02, "augmentation": True,
            },
        ]
    raise ValueError(f"recipe certification supports dense models only, got {model!r}")


def _validate_recipes(recipes):
    required = {
        "name", "optimizer", "lr", "weight_decay", "max_epochs", "warmup_epochs",
        "min_lr_ratio", "augmentation",
    }
    if not isinstance(recipes, list) or not recipes:
        raise ValueError("the recipe ladder must be a non-empty JSON list")
    names = []
    validated = []
    for raw in recipes:
        missing = required - set(raw)
        if missing:
            raise ValueError(f"recipe is missing fields: {sorted(missing)}")
        recipe = dict(raw)
        recipe["name"] = str(recipe["name"])
        recipe["optimizer"] = str(recipe["optimizer"]).lower()
        recipe["lr"] = float(recipe["lr"])
        recipe["weight_decay"] = float(recipe["weight_decay"])
        recipe["max_epochs"] = int(recipe["max_epochs"])
        recipe["warmup_epochs"] = int(recipe["warmup_epochs"])
        recipe["min_lr_ratio"] = float(recipe["min_lr_ratio"])
        recipe["augmentation"] = bool(recipe["augmentation"])
        if recipe["optimizer"] not in {"adamw", "sgd"}:
            raise ValueError(f"unsupported optimizer {recipe['optimizer']!r}")
        if recipe["lr"] <= 0 or recipe["weight_decay"] < 0 or recipe["max_epochs"] <= 0:
            raise ValueError(f"invalid optimization values in recipe {recipe['name']!r}")
        if not 0 <= recipe["warmup_epochs"] < recipe["max_epochs"]:
            raise ValueError(f"invalid warmup in recipe {recipe['name']!r}")
        if not 0 <= recipe["min_lr_ratio"] <= 1:
            raise ValueError(f"invalid min_lr_ratio in recipe {recipe['name']!r}")
        if not recipe["augmentation"]:
            raise ValueError(
                f"recipe {recipe['name']!r} disables production augmentation; "
                "that belongs in a diagnostic, not full-data certification")
        if recipe["optimizer"] == "sgd":
            recipe["momentum"] = float(recipe.get("momentum", 0.9))
            if not 0 <= recipe["momentum"] < 1:
                raise ValueError(f"invalid SGD momentum in recipe {recipe['name']!r}")
        names.append(recipe["name"])
        validated.append(recipe)
    if len(names) != len(set(names)):
        raise ValueError("recipe names must be unique")
    return validated


def _load_recipes(path, model, max_epochs=None):
    recipes = default_recipes(model) if path is None else json.loads(Path(path).read_text())
    recipes = _validate_recipes(recipes)
    if max_epochs is not None:
        for recipe in recipes:
            recipe["max_epochs"] = min(int(recipe["max_epochs"]), int(max_epochs))
            recipe["warmup_epochs"] = min(
                int(recipe["warmup_epochs"]), max(int(recipe["max_epochs"]) - 1, 0))
    return recipes


def _fingerprint(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _metric(metrics, unit):
    return float(metrics["site_top1"] if unit == "site" else metrics["top1"])


def checkpoint_is_eligible(train_metrics, iid_metrics, train_threshold, min_iid,
                           train_unit="site", iid_unit="site"):
    return bool(
        _metric(train_metrics, train_unit) >= float(train_threshold)
        and _metric(iid_metrics, iid_unit) >= float(min_iid)
    )


def choose_eligible_checkpoint(current, candidate):
    """Select by source-IID accuracy, with earlier epoch as a deterministic tiebreaker."""
    if current is None:
        return candidate
    candidate_key = (float(candidate["selection_iid_top1"]), -int(candidate["epoch"]))
    current_key = (float(current["selection_iid_top1"]), -int(current["epoch"]))
    return candidate if candidate_key > current_key else current


def _build_optimizer(model, recipe):
    if recipe["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"],
            momentum=recipe["momentum"], nesterov=True)
    for group in optimizer.param_groups:
        group["initial_lr"] = float(recipe["lr"])
    return optimizer


def _schedule(optimizer, epoch, recipe):
    warmup = int(recipe["warmup_epochs"])
    total = int(recipe["max_epochs"])
    minimum = float(recipe["min_lr_ratio"])
    if epoch < warmup:
        factor = float(epoch + 1) / max(warmup, 1)
    else:
        progress = (epoch - warmup) / max(total - warmup - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        factor = minimum + (1.0 - minimum) * cosine
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor


def _append_jsonl(path, row):
    with open(path, "a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _status_path(result_root, run_name, model):
    return Path(result_root) / "recipe_certification" / run_name / model / "status.json"


def format_status(payload):
    if payload is None:
        return "Recipe certification has not started."
    lines = [(
        f"state={payload.get('state', 'unknown')}  model={payload.get('model', '?')}  "
        f"split={payload.get('split_id', '?')}"
    )]
    if payload.get("attempt_name"):
        lines.append(
            f"attempt={payload.get('attempt_index', '?')}/{payload.get('n_attempts', '?')} "
            f"{payload['attempt_name']}  epoch={payload.get('epoch', 0)}/"
            f"{payload.get('max_epochs', '?')}")
    augmented = payload.get("latest_augmented") or {}
    if augmented:
        lines.append(
            f"latest augmented train: site_top1={augmented.get('site_top1', float('nan')):.4f} "
            f"loss={augmented.get('loss', float('nan')):.4f} "
            f"lr={augmented.get('learning_rate', float('nan')):.3g}")
    measured = payload.get("latest_evaluation") or {}
    if measured:
        lines.append(
            f"full unaugmented train: site={measured['train_site_top1']:.4f} "
            f"well={measured['train_well_top1']:.4f}  "
            f"threshold({payload.get('train_unit', 'site')})="
            f"{payload.get('train_threshold', float('nan')):.4f}")
        lines.append(
            f"source IID only: site={measured['iid_site_top1']:.4f} "
            f"well={measured['iid_well_top1']:.4f}  "
            f"eligible={measured['eligible']}")
    selected = payload.get("best_eligible")
    if selected:
        lines.append(
            f"best eligible: epoch={selected['epoch']} "
            f"train={selected['selection_train_top1']:.4f} "
            f"IID={selected['selection_iid_top1']:.4f}")
    lines.append("target batches: excluded from loading, evaluation, stopping, and selection")
    if payload.get("message"):
        lines.append(str(payload["message"]))
    return "\n".join(lines)


def show_status(result_root, run_name, model):
    path = _status_path(result_root, run_name, model)
    payload = json.loads(path.read_text()) if path.is_file() else None
    print(format_status(payload), flush=True)
    return payload


def watch_status(result_root, run_name, model, interval):
    terminal = bool(sys.stdout.isatty())
    while True:
        path = _status_path(result_root, run_name, model)
        payload = json.loads(path.read_text()) if path.is_file() else None
        if terminal:
            print("\033[2J\033[H", end="")
        print(time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        print(format_status(payload), flush=True)
        if payload and payload.get("state") in {
            "certified", "exhausted", "failed", "interrupted",
        }:
            return payload
        time.sleep(max(float(interval), 1.0))


def _record_terminal_error(args, state, error):
    path = _status_path(args.result_root, args.run_name, args.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(path.read_text()) if path.is_file() else {
        "schema_version": 1, "model": args.model, "split_id": args.split_id,
        "run_name": args.run_name,
    }
    payload.update({
        "state": state, "message": f"{type(error).__name__}: {error}",
        "updated_at": time.time(),
    })
    _atomic_json(path, payload)


def _write_status(path, common, **updates):
    payload = {**common, **updates, "updated_at": time.time()}
    _atomic_json(path, payload)
    return payload


def _source_split(registry, split_id):
    matches = [row for row in registry["main_training_splits"] if row["split_id"] == split_id]
    if len(matches) != 1:
        raise ValueError(f"split_id {split_id!r} occurs {len(matches)} times")
    return matches[0]


def _resume_state(path, fingerprint, model, optimizer, device):
    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("recipe_fingerprint") != fingerprint:
        raise RuntimeError(f"resume checkpoint recipe mismatch: {path}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    model.to(device)
    if state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda" and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"], device)
    if state.get("numpy_rng_state") is not None:
        np.random.set_state(state["numpy_rng_state"])
    return state


def _save_resume(path, fingerprint, model, optimizer, epoch, initial_loss,
                 best_eligible, best_eligible_state, consecutive_passes, device,
                 train_generator):
    _atomic_torch(path, {
        "schema_version": 1, "recipe_fingerprint": fingerprint, "epoch": int(epoch),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "initial_loss": initial_loss,
        "best_eligible": best_eligible, "best_eligible_state": best_eligible_state,
        "consecutive_passes": int(consecutive_passes),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "numpy_rng_state": np.random.get_state(),
        "train_generator_state": (
            train_generator.get_state() if train_generator is not None else None),
    })


def certify(args):
    result_root = Path(args.result_root).expanduser().resolve()
    registry_path = result_root / "study_registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"missing frozen study registry: {registry_path}")
    registry = json.loads(registry_path.read_text())
    split_spec = _source_split(registry, args.split_id)
    recipes = _load_recipes(args.recipes_json, args.model, args.max_epochs)
    sha, dirty = _git_info()
    if dirty and not args.allow_dirty:
        raise RuntimeError("refuse to certify a recipe from a dirty tracked checkout")

    output_dir = result_root / "recipe_certification" / args.run_name / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    certificate_path = output_dir / "CERTIFIED_RECIPE.json"
    exhausted_path = output_dir / "EXHAUSTED.json"
    minimum_iid = (2.0 / EXPECTED_TREATMENTS if args.min_iid is None else float(args.min_iid))
    config = {
        "schema_version": 1, "model": args.model, "split_id": args.split_id,
        "train_threshold": float(args.train_threshold), "train_unit": args.train_unit,
        "iid_unit": args.iid_unit, "minimum_iid": minimum_iid,
        "eval_every": int(args.eval_every),
        "confirmation_evaluations": int(args.confirmation_evaluations),
        "batch_size": int(args.batch_size), "num_workers": int(args.num_workers),
        "image_size": int(args.image_size), "seed": int(args.seed), "recipes": recipes,
        "target_policy": "excluded from loading, evaluation, stopping, and selection",
    }
    config["fingerprint"] = _fingerprint(config)
    config_path = output_dir / "certification_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text())
        if existing.get("fingerprint") != config["fingerprint"]:
            raise RuntimeError(
                f"certification configuration changed in {output_dir}; choose a new --run-name")
    else:
        _atomic_json(config_path, config)
    if certificate_path.is_file():
        payload = json.loads(certificate_path.read_text())
        print(f"[certified] existing certificate: {certificate_path}", flush=True)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return payload
    if exhausted_path.is_file():
        payload = json.loads(exhausted_path.read_text())
        print(f"[exhausted] existing bounded ladder: {exhausted_path}", flush=True)
        return payload

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("recipe certification requires CUDA; pass --allow-cpu only for tests")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = {
        "schema_version": 1, "model": args.model, "split_id": args.split_id,
        "run_name": args.run_name, "n_attempts": len(recipes),
        "train_threshold": float(args.train_threshold), "train_unit": args.train_unit,
        "iid_unit": args.iid_unit, "minimum_iid": minimum_iid,
        "git_commit": sha, "git_dirty": dirty,
    }
    _write_status(
        status_path, common, state="initializing", epoch=0,
        message="Building full source-only loaders; target loader is disabled.")

    assignment, validation_loaders = _make_loaders(
        result_root, registry, split_spec, args.batch_size, args.num_workers,
        args.image_size, canary=False, include_target=False, train_augmentation=True)
    if "target" in validation_loaders:
        raise RuntimeError("target loader must not exist during recipe certification")
    del validation_loaders
    split_digest = _split_hash(assignment)
    attempt_summaries = []
    for attempt_index, recipe in enumerate(recipes, start=1):
        attempt_dir = output_dir / f"attempt_{attempt_index:02d}_{recipe['name']}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_complete = attempt_dir / "COMPLETE.json"
        if attempt_complete.is_file():
            summary = json.loads(attempt_complete.read_text())
            attempt_summaries.append(summary)
            continue
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
        _attempt_assignment, loaders = _make_loaders(
            result_root, registry, split_spec, args.batch_size, args.num_workers,
            args.image_size, canary=False, include_target=False,
            train_augmentation=recipe["augmentation"])
        model, model_audit = build_study_model(
            args.model, EXPECTED_TREATMENTS, args.image_size)
        model = model.to(device)
        optimizer = _build_optimizer(model, recipe)
        recipe_fingerprint = _fingerprint(recipe)
        resume_path = attempt_dir / "resume.pt"
        resume = _resume_state(
            resume_path, recipe_fingerprint, model, optimizer, device)
        if resume and resume.get("train_generator_state") is not None:
            loaders["train"].generator.set_state(resume["train_generator_state"])
        start_epoch = int(resume["epoch"]) if resume else 0
        initial_loss = resume.get("initial_loss") if resume else None
        best_eligible = resume.get("best_eligible") if resume else None
        best_eligible_state = resume.get("best_eligible_state") if resume else None
        consecutive_passes = int(resume.get("consecutive_passes", 0)) if resume else 0
        latest_evaluation = None
        curve_path = attempt_dir / "curves.jsonl"
        started = time.time()
        if resume:
            print(f"[resume] {recipe['name']} after epoch {start_epoch}", flush=True)
        for epoch in range(start_epoch, int(recipe["max_epochs"])):
            _schedule(optimizer, epoch, recipe)
            model.train()
            total_loss = total_correct = total_count = 0
            for batch_index, batch in enumerate(loaders["train"]):
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite supervised loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if epoch == 0 and batch_index == 0:
                    gradients = [
                        parameter.grad for parameter in model.parameters()
                        if parameter.requires_grad and parameter.grad is not None
                    ]
                    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
                        raise FloatingPointError("missing or non-finite supervised gradients")
                    initial_loss = float(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss) * len(images)
                total_correct += int((logits.argmax(1) == labels).sum())
                total_count += len(images)
            augmented = {
                "loss": total_loss / total_count, "site_top1": total_correct / total_count,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            should_evaluate = (
                (epoch + 1) % int(args.eval_every) == 0
                or epoch + 1 == int(recipe["max_epochs"])
            )
            row = {
                "phase": "supervised", "attempt": attempt_index, "recipe": recipe["name"],
                "epoch": epoch + 1, "train_augmented_loss": augmented["loss"],
                "train_augmented_site_top1": augmented["site_top1"],
                "learning_rate": augmented["learning_rate"], "evaluated": should_evaluate,
            }
            if should_evaluate:
                train_metrics, _, _ = evaluate(model, loaders["train_eval"], device)
                iid_metrics, _, _ = evaluate(model, loaders["iid_validation"], device)
                eligible = checkpoint_is_eligible(
                    train_metrics, iid_metrics, args.train_threshold, minimum_iid,
                    args.train_unit, args.iid_unit)
                consecutive_passes = consecutive_passes + 1 if eligible else 0
                candidate = {
                    "epoch": epoch + 1,
                    "selection_train_top1": _metric(train_metrics, args.train_unit),
                    "selection_iid_top1": _metric(iid_metrics, args.iid_unit),
                    "train_site_top1": float(train_metrics["site_top1"]),
                    "train_well_top1": float(train_metrics["top1"]),
                    "train_site_loss": float(train_metrics["site_loss"]),
                    "train_well_loss": float(train_metrics["loss"]),
                    "iid_site_top1": float(iid_metrics["site_top1"]),
                    "iid_well_top1": float(iid_metrics["top1"]),
                    "iid_site_loss": float(iid_metrics["site_loss"]),
                    "iid_well_loss": float(iid_metrics["loss"]),
                }
                if eligible:
                    selected = choose_eligible_checkpoint(best_eligible, candidate)
                    if selected is candidate:
                        best_eligible = candidate
                        best_eligible_state = copy.deepcopy({
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        })
                latest_evaluation = {**candidate, "eligible": eligible}
                row.update(latest_evaluation)
                _save_resume(
                    resume_path, recipe_fingerprint, model, optimizer, epoch + 1,
                    initial_loss, best_eligible, best_eligible_state,
                    consecutive_passes, device, loaders["train"].generator)
            _append_jsonl(curve_path, row)
            payload = _write_status(
                status_path, common, state="training", attempt_index=attempt_index,
                attempt_name=recipe["name"], max_epochs=recipe["max_epochs"],
                epoch=epoch + 1, latest_augmented=augmented,
                latest_evaluation=latest_evaluation, best_eligible=best_eligible,
                consecutive_passes=consecutive_passes,
                message=(
                    "Threshold-clearing checkpoints are selected by source-IID accuracy."
                    if best_eligible else "No checkpoint has cleared the full-data gate yet."
                ),
            )
            print("[recipe] " + format_status(payload).replace("\n", " | "), flush=True)
            if best_eligible and consecutive_passes >= int(args.confirmation_evaluations):
                checkpoint_path = output_dir / "certified_checkpoint.pt"
                _atomic_torch(checkpoint_path, {
                    "schema_version": 1, "model": best_eligible_state,
                    "model_kind": args.model, "model_audit": model_audit,
                    "split_id": args.split_id, "split_hash": split_digest,
                    "recipe": recipe, "selected_epoch": best_eligible["epoch"],
                    "certified_at_epoch": epoch + 1,
                })
                certificate = {
                    **common, "state": "certified", "certified": True,
                    "attempt_index": attempt_index, "attempt_name": recipe["name"],
                    "recipe": recipe, "selected_checkpoint": best_eligible,
                    "selected_epoch": best_eligible["epoch"],
                    "certified_at_epoch": epoch + 1,
                    "schedule_total_epochs": recipe["max_epochs"],
                    "checkpoint": str(checkpoint_path), "split_hash": split_digest,
                    "model_audit": model_audit, "initial_batch_loss": initial_loss,
                    "selection_rule": (
                        f"highest source-IID {args.iid_unit} top1 among checkpoints with "
                        f"unaugmented train {args.train_unit} top1 >= {args.train_threshold}"
                    ),
                    "target_policy": config["target_policy"],
                    "attempts_before_certificate": attempt_summaries,
                    "elapsed_seconds_this_attempt": time.time() - started,
                }
                _atomic_json(certificate_path, certificate)
                _write_status(
                    status_path, common, **certificate,
                    latest_augmented=augmented, latest_evaluation=latest_evaluation,
                    best_eligible=best_eligible,
                    message=f"Certified recipe written to {certificate_path}")
                print(f"[certified] {certificate_path}", flush=True)
                return certificate
        summary = {
            "attempt_index": attempt_index, "attempt_name": recipe["name"],
            "certified": False, "recipe": recipe, "best_eligible": best_eligible,
            "latest_evaluation": latest_evaluation,
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(attempt_complete, summary)
        attempt_summaries.append(summary)

    exhausted = {
        **common, "state": "exhausted", "certified": False,
        "attempts": attempt_summaries, "split_hash": split_digest,
        "target_policy": config["target_policy"],
        "message": (
            "The bounded full-data recipe ladder ended without clearing the threshold. "
            "Do not launch the dense-versus-MoE comparison."
        ),
    }
    _atomic_json(exhausted_path, exhausted)
    _write_status(status_path, common, **exhausted)
    print(f"[exhausted] {exhausted_path}", flush=True)
    return exhausted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--run-name", default="production_v1")
    parser.add_argument("--model", choices=("resnet18", "vit_tiny"), default="resnet18")
    parser.add_argument("--split-id", default="primary_fold0")
    parser.add_argument("--train-threshold", type=float, default=0.80)
    parser.add_argument("--train-unit", choices=("site", "well"), default="site")
    parser.add_argument("--iid-unit", choices=("site", "well"), default="site")
    parser.add_argument("--min-iid", type=float)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--confirmation-evaluations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recipes-json")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=float, default=30.0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.eval_every <= 0 or args.confirmation_evaluations <= 0:
        parser.error("--eval-every and --confirmation-evaluations must be positive")
    if not 0 < args.train_threshold <= 1:
        parser.error("--train-threshold must lie in (0, 1]")
    if args.status and args.watch:
        parser.error("choose either --status or --watch")
    if args.status:
        show_status(args.result_root, args.run_name, args.model)
        return
    if args.watch:
        watch_status(args.result_root, args.run_name, args.model, args.watch_interval)
        return
    try:
        result = certify(args)
    except KeyboardInterrupt as error:
        _record_terminal_error(args, "interrupted", error)
        raise
    except Exception as error:
        _record_terminal_error(args, "failed", error)
        raise
    if not result.get("certified"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
