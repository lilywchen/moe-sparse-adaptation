#!/usr/bin/env python
"""Measure full-data RxRx1 HUVEC recipes until source-IID performance plateaus.

This is deliberately different from the small memorization canary.  It uses the complete
``primary_fold0`` source training set, production image augmentation, and the frozen source-IID
split. Training continues past the diagnostic 80% marker and stops only after a predefined
source-IID plateau rule. Target-batch images are never loaded or evaluated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
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
    if model in ("vit_tiny", "vit_micro"):
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


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pretrained_encoder(model, checkpoint_path, expected_model):
    """Load an MAE encoder while deliberately retaining a fresh classifier head."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or {}
    if config.get("model") != expected_model:
        raise ValueError(
            f"pretrained checkpoint model {config.get('model')!r} != {expected_model!r}")
    state = dict(checkpoint["encoder"])
    removed = sorted(key for key in state if key in {"fc.weight", "fc.bias"})
    for key in removed:
        state.pop(key)
    incompatible = model.load_state_dict(state, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if missing != ["fc.bias", "fc.weight"] or unexpected:
        raise RuntimeError(
            f"MAE encoder mismatch: missing={missing}, unexpected={unexpected}")
    return {
        "kind": "mae", "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "pretraining_config": config,
        "pretraining_epoch": checkpoint.get("epoch"),
        "classifier_head_loaded": False,
        "loaded_backbone_keys": len(state),
    }


def _metric(metrics, unit):
    return float(metrics["site_top1"] if unit == "site" else metrics["top1"])


def choose_source_iid_checkpoint(current, candidate):
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


def _curve_history(path):
    """Recover live metrics from an existing curve when extending an older run."""
    latest_augmented = None
    latest_evaluation = None
    best_source_iid = None
    threshold_ever_reached = False
    if not path.is_file():
        return latest_augmented, latest_evaluation, best_source_iid, threshold_ever_reached
    evaluation_keys = (
        "epoch", "selection_train_top1", "selection_iid_top1",
        "train_site_top1", "train_well_top1", "train_site_loss", "train_well_loss",
        "iid_site_top1", "iid_well_top1", "iid_site_loss", "iid_well_loss",
    )
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") != "supervised":
            continue
        latest_augmented = {
            "loss": float(row["train_augmented_loss"]),
            "site_top1": float(row["train_augmented_site_top1"]),
            "learning_rate": float(row["learning_rate"]),
        }
        if row.get("evaluated") and all(key in row for key in evaluation_keys):
            threshold_reached = bool(
                row.get("threshold_reached", row.get("eligible", False)))
            candidate = {key: row[key] for key in evaluation_keys}
            latest_evaluation = {**candidate, "threshold_reached": threshold_reached}
            best_source_iid = choose_source_iid_checkpoint(best_source_iid, candidate)
            threshold_ever_reached = threshold_ever_reached or threshold_reached
    return latest_augmented, latest_evaluation, best_source_iid, threshold_ever_reached


def _plateau_history(path, min_delta):
    """Return the meaningful best IID score and evaluations since it improved."""
    best_score = None
    best_epoch = None
    stale_evaluations = 0
    if not path.is_file():
        return best_score, best_epoch, stale_evaluations
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") != "supervised" or not row.get("evaluated"):
            continue
        if "selection_iid_top1" not in row:
            continue
        score = float(row["selection_iid_top1"])
        if best_score is None or score >= best_score + float(min_delta):
            best_score = score
            best_epoch = int(row["epoch"])
            stale_evaluations = 0
        else:
            stale_evaluations += 1
    return best_score, best_epoch, stale_evaluations


def _normalize_attempt_summary(summary, attempt_dir, recipe):
    """Read both new full-horizon and legacy exhausted-at-horizon summaries."""
    if "terminal_evaluation" in summary:
        return summary
    augmented, evaluation, best, threshold_reached = _curve_history(
        attempt_dir / "curves.jsonl")
    evaluation = summary.get("latest_evaluation") or evaluation
    best = summary.get("best_source_iid", summary.get("best_eligible")) or best
    if evaluation is None:
        raise RuntimeError(f"completed attempt has no terminal evaluation: {attempt_dir}")
    normalized = {
        **summary,
        "state": "complete",
        "terminal_epoch": int(recipe["max_epochs"]),
        "terminal_evaluation": evaluation,
        "best_source_iid": best,
        "threshold_ever_reached": bool(
            summary.get("threshold_ever_reached", threshold_reached)),
        "latest_augmented": summary.get("latest_augmented") or augmented,
    }
    _atomic_json(attempt_dir / "COMPLETE.json", normalized)
    return normalized


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
        displayed_epoch = payload.get("epoch")
        if not displayed_epoch and payload.get("state") in {"certified", "complete"}:
            displayed_epoch = payload.get("certified_at_epoch") or payload.get("selected_epoch")
        displayed_max = payload.get("max_epochs") or (payload.get("recipe") or {}).get(
            "max_epochs", "?")
        lines.append(
            f"attempt={payload.get('attempt_index', '?')}/{payload.get('n_attempts', '?')} "
            f"{payload['attempt_name']}  epoch={displayed_epoch or 0}/{displayed_max}")
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
            f"well={measured['train_well_top1']:.4f}")
        lines.append(
            f"source IID only: site={measured['iid_site_top1']:.4f} "
            f"well={measured['iid_well_top1']:.4f}")
        lines.append(
            f"diagnostic train marker >= {payload.get('train_threshold', float('nan')):.2f}: "
            f"reached={measured.get('threshold_reached', measured.get('eligible', False))} "
            "(never stops training)")
    selected = payload.get("best_source_iid") or payload.get("best_eligible")
    if selected:
        lines.append(
            f"best source-IID so far: epoch={selected['epoch']} "
            f"train={selected['selection_train_top1']:.4f} "
            f"IID={selected['selection_iid_top1']:.4f}")
    elapsed = payload.get("elapsed_seconds")
    if elapsed is None and payload.get("selected_attempt"):
        elapsed = payload["selected_attempt"].get("elapsed_seconds")
    if elapsed is not None:
        lines.append(f"wall clock: {float(elapsed) / 60.0:.1f} minutes")
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
            "complete", "failed", "interrupted",
        }:
            return payload
        time.sleep(max(float(interval), 1.0))


def _record_terminal_error(args, state, error):
    path = _status_path(args.result_root, args.run_name, args.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(path.read_text()) if path.is_file() else {
        "schema_version": 1, "model": args.model, "split_id": args.split_id,
        "image_size": int(args.image_size),
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
                 best_source_iid, threshold_ever_reached, elapsed_seconds_extension,
                 device, train_generator):
    _atomic_torch(path, {
        "schema_version": 1, "recipe_fingerprint": fingerprint, "epoch": int(epoch),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(), "initial_loss": initial_loss,
        "best_source_iid": best_source_iid,
        # Backward-compatible names allow an old threshold-stopped run to resume.
        "best_eligible": best_source_iid,
        "threshold_ever_reached": bool(threshold_ever_reached),
        "elapsed_seconds_extension": float(elapsed_seconds_extension),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "numpy_rng_state": np.random.get_state(),
        "train_generator_state": (
            train_generator.get_state() if train_generator is not None else None),
    })


def certify(args):
    result_root = Path(args.result_root).expanduser().resolve()
    registry_path = (Path(args.registry).expanduser().resolve()
                     if args.registry else result_root / "study_registry.json")
    if not registry_path.is_file():
        raise FileNotFoundError(f"missing frozen study registry: {registry_path}")
    registry = json.loads(registry_path.read_text())
    if args.site_manifest:
        registry["site_manifest"] = str(Path(args.site_manifest).expanduser().resolve())
    if args.raw_root:
        registry["raw_root"] = str(Path(args.raw_root).expanduser().resolve())
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
    plateau_result_path = output_dir / "PLATEAU_RESULT.json"
    minimum_iid = (2.0 / EXPECTED_TREATMENTS if args.min_iid is None else float(args.min_iid))
    initialization = {
        "kind": "mae" if args.init_checkpoint else "random",
        "checkpoint": str(Path(args.init_checkpoint).expanduser().resolve())
        if args.init_checkpoint else None,
        "checkpoint_sha256": _sha256(Path(args.init_checkpoint).expanduser().resolve())
        if args.init_checkpoint else None,
    }
    config = {
        "schema_version": 1, "model": args.model, "split_id": args.split_id,
        "train_threshold": float(args.train_threshold), "train_unit": args.train_unit,
        "iid_unit": args.iid_unit, "minimum_iid": minimum_iid,
        "eval_every": int(args.eval_every),
        "confirmation_evaluations": int(args.confirmation_evaluations),
        "batch_size": int(args.batch_size), "num_workers": int(args.num_workers),
        "image_size": int(args.image_size), "seed": int(args.seed), "recipes": recipes,
        "initialization": initialization,
        "registry": str(registry_path),
        "site_manifest": registry["site_manifest"], "raw_root": registry["raw_root"],
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
    plateau_config = {
        "schema_version": 1,
        "metric": f"source_iid_{args.iid_unit}_top1",
        "eval_every_epochs": int(args.eval_every),
        "patience_evaluations": int(args.plateau_patience_evals),
        "minimum_delta": float(args.plateau_min_delta),
        "minimum_epochs": int(args.plateau_min_epochs),
        "maximum_epochs": int(args.plateau_max_epochs),
        "target_policy": config["target_policy"],
    }
    plateau_config["fingerprint"] = _fingerprint(plateau_config)
    plateau_config_path = output_dir / "plateau_config.json"
    if plateau_config_path.is_file():
        existing = json.loads(plateau_config_path.read_text())
        if existing.get("fingerprint") != plateau_config["fingerprint"]:
            raise RuntimeError(
                f"plateau configuration changed in {output_dir}; choose a new --run-name")
    else:
        _atomic_json(plateau_config_path, plateau_config)
    if plateau_result_path.is_file():
        payload = json.loads(plateau_result_path.read_text())
        print(f"[complete] existing plateau result: {plateau_result_path}", flush=True)
        return payload
    early_snapshot_path = output_dir / "EARLY_THRESHOLD_SNAPSHOT.json"
    if certificate_path.is_file() and not early_snapshot_path.is_file():
        _atomic_json(early_snapshot_path, json.loads(certificate_path.read_text()))
    early_snapshot = (
        json.loads(early_snapshot_path.read_text()) if early_snapshot_path.is_file() else {})
    if exhausted_path.is_file():
        old_exhausted_path = output_dir / "EARLY_GATE_EXHAUSTED.json"
        if not old_exhausted_path.is_file():
            _atomic_json(old_exhausted_path, json.loads(exhausted_path.read_text()))

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("recipe certification requires CUDA; pass --allow-cpu only for tests")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = {
        "schema_version": 1, "model": args.model, "split_id": args.split_id,
        "image_size": int(args.image_size),
        "run_name": args.run_name, "n_attempts": len(recipes),
        "train_threshold": float(args.train_threshold), "train_unit": args.train_unit,
        "iid_unit": args.iid_unit, "minimum_iid": minimum_iid,
        "git_commit": sha, "git_dirty": dirty,
        "stopping_rule": (
            f"source-IID plateau: {args.plateau_patience_evals} evaluations without "
            f">= {args.plateau_min_delta:.4f} improvement after epoch "
            f"{args.plateau_min_epochs}; hard safety ceiling {args.plateau_max_epochs}"
        ),
        "plateau_config": plateau_config,
        "initialization": initialization,
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
            summary = _normalize_attempt_summary(
                json.loads(attempt_complete.read_text()), attempt_dir, recipe)
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
        init_audit = ({"kind": "random", "seed": int(args.seed)}
                      if not args.init_checkpoint else load_pretrained_encoder(
                          model, args.init_checkpoint, args.model))
        model_audit = {**model_audit, "initialization": init_audit}
        model = model.to(device)
        optimizer = _build_optimizer(model, recipe)
        recipe_fingerprint = _fingerprint({"recipe": recipe, "initialization": initialization})
        resume_path = attempt_dir / "resume.pt"
        resume = _resume_state(
            resume_path, recipe_fingerprint, model, optimizer, device)
        was_resumed = resume is not None
        if resume and resume.get("train_generator_state") is not None:
            loaders["train"].generator.set_state(resume["train_generator_state"])
        start_epoch = int(resume["epoch"]) if resume else 0
        initial_loss = resume.get("initial_loss") if resume else None
        best_source_iid = (
            resume.get("best_source_iid", resume.get("best_eligible")) if resume else None)
        threshold_ever_reached = bool(
            resume.get("threshold_ever_reached", best_source_iid is not None)
            if resume else False)
        elapsed_seconds_extension = float(
            resume.get("elapsed_seconds_extension", 0.0) if resume else 0.0)
        curve_path = attempt_dir / "curves.jsonl"
        best_checkpoint_path = attempt_dir / "best_source_iid_checkpoint.pt"
        legacy_checkpoint_path = output_dir / "certified_checkpoint.pt"
        if (not best_checkpoint_path.is_file() and legacy_checkpoint_path.is_file()
                and early_snapshot.get("attempt_name") == recipe["name"]):
            shutil.copy2(legacy_checkpoint_path, best_checkpoint_path)
        historical_augmented, historical_evaluation, historical_best, historical_threshold = (
            _curve_history(curve_path))
        if best_source_iid is None and historical_best is not None:
            best_source_iid = historical_best
        threshold_ever_reached = threshold_ever_reached or historical_threshold
        plateau_best_score, plateau_best_epoch, stale_evaluations = _plateau_history(
            curve_path, args.plateau_min_delta)
        del resume
        latest_evaluation = historical_evaluation
        augmented = historical_augmented
        started = time.time()
        prior_elapsed_seconds = 0.0
        if early_snapshot.get("attempt_name") == recipe["name"]:
            prior_elapsed_seconds = float(
                early_snapshot.get("elapsed_seconds_this_attempt", 0.0))
        if was_resumed:
            print(f"[resume] {recipe['name']} after epoch {start_epoch}", flush=True)
        stop_reason = None
        if (start_epoch >= int(args.plateau_min_epochs)
                and stale_evaluations >= int(args.plateau_patience_evals)):
            stop_reason = (
                f"source-IID plateau already satisfied at resume: {stale_evaluations} "
                f"evaluations without >= {args.plateau_min_delta:.4f} improvement"
            )
        terminal_epoch = start_epoch
        safety_horizon = min(int(recipe["max_epochs"]), int(args.plateau_max_epochs))
        for epoch in range(start_epoch, safety_horizon):
            if stop_reason:
                break
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
                or epoch + 1 == safety_horizon
            )
            row = {
                "phase": "supervised", "attempt": attempt_index, "recipe": recipe["name"],
                "epoch": epoch + 1, "train_augmented_loss": augmented["loss"],
                "train_augmented_site_top1": augmented["site_top1"],
                "learning_rate": augmented["learning_rate"], "evaluated": should_evaluate,
            }
            should_stop_for_plateau = False
            if should_evaluate:
                train_metrics, _, _ = evaluate(model, loaders["train_eval"], device)
                iid_metrics, _, _ = evaluate(model, loaders["iid_validation"], device)
                threshold_reached = bool(
                    _metric(train_metrics, args.train_unit) >= float(args.train_threshold))
                threshold_ever_reached = threshold_ever_reached or threshold_reached
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
                selected = choose_source_iid_checkpoint(best_source_iid, candidate)
                if selected is candidate:
                    best_source_iid = candidate
                    _atomic_torch(best_checkpoint_path, {
                        "schema_version": 1,
                        "model": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "model_kind": args.model, "model_audit": model_audit,
                        "split_id": args.split_id, "split_hash": split_digest,
                        "recipe": recipe, "selected_checkpoint": candidate,
                        "selected_epoch": candidate["epoch"],
                        "initialization": init_audit,
                    })
                iid_score = float(candidate["selection_iid_top1"])
                meaningful_improvement = bool(
                    plateau_best_score is None
                    or iid_score >= plateau_best_score + float(args.plateau_min_delta)
                )
                if meaningful_improvement:
                    plateau_best_score = iid_score
                    plateau_best_epoch = epoch + 1
                    stale_evaluations = 0
                else:
                    stale_evaluations += 1
                should_stop_for_plateau = bool(
                    epoch + 1 >= int(args.plateau_min_epochs)
                    and stale_evaluations >= int(args.plateau_patience_evals)
                )
                latest_evaluation = {
                    **candidate, "threshold_reached": threshold_reached,
                    "meaningful_iid_improvement": meaningful_improvement,
                    "plateau_best_score": plateau_best_score,
                    "plateau_best_epoch": plateau_best_epoch,
                    "stale_evaluations": stale_evaluations,
                }
                row.update(latest_evaluation)
                _save_resume(
                    resume_path, recipe_fingerprint, model, optimizer, epoch + 1,
                    initial_loss, best_source_iid, threshold_ever_reached,
                    elapsed_seconds_extension + time.time() - started,
                    device, loaders["train"].generator)
            _append_jsonl(curve_path, row)
            terminal_epoch = epoch + 1
            if should_evaluate and should_stop_for_plateau:
                stop_reason = (
                    f"source-IID plateau after {stale_evaluations} evaluations without "
                    f">= {args.plateau_min_delta:.4f} improvement"
                )
            payload = _write_status(
                status_path, common, state="training", attempt_index=attempt_index,
                attempt_name=recipe["name"], max_epochs=recipe["max_epochs"],
                epoch=epoch + 1, latest_augmented=augmented,
                latest_evaluation=latest_evaluation, best_source_iid=best_source_iid,
                threshold_ever_reached=threshold_ever_reached,
                elapsed_seconds=(
                    prior_elapsed_seconds + elapsed_seconds_extension
                    + time.time() - started),
                stale_evaluations=stale_evaluations,
                message=(
                    stop_reason or
                    f"Plateau watch: {stale_evaluations}/{args.plateau_patience_evals} "
                    "stale source-IID evaluations; the 80% marker never stops training."
                ),
            )
            print("[recipe] " + format_status(payload).replace("\n", " | "), flush=True)
            if stop_reason:
                break
        if latest_evaluation is None:
            raise RuntimeError("training ended without a source-IID evaluation")
        if not best_checkpoint_path.is_file():
            raise RuntimeError(
                f"best source-IID checkpoint was not persisted: {best_checkpoint_path}")
        if stop_reason is None:
            stop_reason = "maximum safety horizon reached before the plateau rule fired"
        checkpoint_path = attempt_dir / "terminal_checkpoint.pt"
        _atomic_torch(checkpoint_path, {
            "schema_version": 1,
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_kind": args.model, "model_audit": model_audit,
            "split_id": args.split_id, "split_hash": split_digest,
            "recipe": recipe, "terminal_epoch": terminal_epoch,
            "initialization": init_audit,
        })
        resumed_elapsed = elapsed_seconds_extension + time.time() - started
        summary = {
            "attempt_index": attempt_index, "attempt_name": recipe["name"],
            "state": "complete", "recipe": recipe,
            "terminal_epoch": terminal_epoch,
            "maximum_safety_epochs": safety_horizon,
            "stop_reason": stop_reason,
            "terminal_evaluation": latest_evaluation,
            "best_source_iid": best_source_iid,
            "best_source_iid_checkpoint": str(best_checkpoint_path),
            "threshold_ever_reached": threshold_ever_reached,
            "checkpoint": str(best_checkpoint_path),
            "terminal_checkpoint": str(checkpoint_path),
            "latest_augmented": augmented,
            "elapsed_seconds_resumed": resumed_elapsed,
            "elapsed_seconds": prior_elapsed_seconds + resumed_elapsed,
        }
        _atomic_json(attempt_complete, summary)
        attempt_summaries.append(summary)

    selected_attempt = max(
        attempt_summaries,
        key=lambda row: float(row["best_source_iid"]["selection_iid_top1"]),
    )
    result = {
        **common, "state": "complete", "plateau_complete": True,
        "attempts": attempt_summaries, "selected_attempt": selected_attempt,
        "split_hash": split_digest,
        "target_policy": config["target_policy"],
        "selection_rule": f"highest peak source-IID {args.iid_unit} accuracy",
        "message": (
            "Training continued past the diagnostic marker and stopped only under the "
            "frozen source-IID plateau rule."
        ),
    }
    _atomic_json(plateau_result_path, result)
    _write_status(
        status_path, common, **result,
        attempt_index=selected_attempt["attempt_index"],
        attempt_name=selected_attempt["attempt_name"],
        recipe=selected_attempt["recipe"],
        epoch=selected_attempt["terminal_epoch"],
        max_epochs=selected_attempt["maximum_safety_epochs"],
        latest_augmented=selected_attempt["latest_augmented"],
        latest_evaluation=selected_attempt["terminal_evaluation"],
        best_source_iid=selected_attempt["best_source_iid"],
    )
    print(f"[complete] {plateau_result_path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--registry")
    parser.add_argument("--site-manifest")
    parser.add_argument("--raw-root")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--run-name", default="production_v1")
    parser.add_argument(
        "--model", choices=("resnet18", "vit_tiny", "vit_micro"),
        default="resnet18")
    parser.add_argument("--split-id", default="primary_fold0")
    parser.add_argument("--train-threshold", type=float, default=0.80)
    parser.add_argument("--train-unit", choices=("site", "well"), default="site")
    parser.add_argument("--iid-unit", choices=("site", "well"), default="site")
    parser.add_argument("--min-iid", type=float)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--confirmation-evaluations", type=int, default=1)
    parser.add_argument("--plateau-patience-evals", type=int, default=4)
    parser.add_argument("--plateau-min-delta", type=float, default=0.001)
    parser.add_argument("--plateau-min-epochs", type=int, default=30)
    parser.add_argument("--plateau-max-epochs", type=int, default=80)
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
    if args.plateau_patience_evals <= 0 or args.plateau_min_epochs <= 0:
        parser.error("plateau patience and minimum epochs must be positive")
    if args.plateau_max_epochs < args.plateau_min_epochs:
        parser.error("--plateau-max-epochs must be at least --plateau-min-epochs")
    if args.plateau_min_delta < 0:
        parser.error("--plateau-min-delta must be nonnegative")
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
    if not result.get("plateau_complete"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
