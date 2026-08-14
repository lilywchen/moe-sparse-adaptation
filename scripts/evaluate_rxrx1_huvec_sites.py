#!/usr/bin/env python
"""Backfill site-level evaluation from completed RxRx1 HUVEC checkpoints.

This is deliberately separate from training so a running frozen wave does not need to be
restarted or changed.  It writes additive ``*.site_metrics.json`` and
``*.site_predictions.parquet`` artifacts; existing run JSON, checkpoints, and well predictions
are never modified.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.data.rxrx1_huvec import EXPECTED_TREATMENTS
from moe_shift.models.huvec import build_study_model
from moe_shift.utils import gpulease
from scripts.run_rxrx1_huvec_study import (
    _atomic_json,
    _git_info,
    _load_spec,
    _make_loaders,
    _split_hash,
)


def _atomic_parquet(path, frame):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _site_metrics(logits, labels, experiments, well_ids, sites, global_indices):
    """Score independent site predictions without pooling within wells."""
    logits = torch.as_tensor(logits).float().cpu()
    labels = torch.as_tensor(labels).long().cpu()
    experiments = torch.as_tensor(experiments).long().cpu()
    sites = torch.as_tensor(sites).long().cpu()
    global_indices = torch.as_tensor(global_indices).long().cpu()
    if not (len(logits) == len(labels) == len(experiments) == len(well_ids)
            == len(sites) == len(global_indices)):
        raise ValueError("site prediction arrays have inconsistent lengths")
    if logits.ndim != 2 or logits.shape[1] != EXPECTED_TREATMENTS:
        raise ValueError(f"expected site logits [N,{EXPECTED_TREATMENTS}], got {tuple(logits.shape)}")
    prediction = logits.argmax(1)
    true_score = logits.gather(1, labels[:, None])
    rank = (logits > true_score).sum(1) + 1
    true_log_probability = F.log_softmax(logits, 1).gather(1, labels[:, None])[:, 0]
    frame = pd.DataFrame({
        "global_index": global_indices.numpy(),
        "well_id": list(map(str, well_ids)),
        "site": sites.numpy(),
        "experiment": experiments.numpy(),
        "label": labels.numpy(),
        "prediction": prediction.numpy(),
        "true_class_rank": rank.numpy(),
        "true_log_probability": true_log_probability.numpy(),
        "correct_top1": (rank == 1).numpy(),
        "correct_top5": (rank <= 5).numpy(),
    })
    per_experiment = {}
    for experiment, group in frame.groupby("experiment"):
        per_experiment[str(int(experiment))] = {
            "n_sites": len(group), "n_wells": int(group.well_id.nunique()),
            "top1": float(group.correct_top1.mean()),
            "top5": float(group.correct_top5.mean()),
            "mean_rank": float(group.true_class_rank.mean()),
        }
    metrics = {
        "n_sites": len(frame), "n_wells": int(frame.well_id.nunique()),
        "top1": float(frame.correct_top1.mean()),
        "top5": float(frame.correct_top5.mean()),
        "loss": float(F.cross_entropy(logits, labels)),
        "mean_rank": float(frame.true_class_rank.mean()),
        "per_experiment": per_experiment,
    }
    return metrics, frame


@torch.inference_mode()
def evaluate_sites(model, loader, device):
    model.eval()
    logits, labels, experiments, sites, global_indices, well_ids = [], [], [], [], [], []
    for batch in loader:
        logits.append(model(batch["image"].to(device, non_blocking=True)).float().cpu())
        labels.append(torch.as_tensor(batch["label"]).long())
        experiments.append(torch.as_tensor(batch["experiment"]).long())
        sites.append(torch.as_tensor(batch["site"]).long())
        global_indices.append(torch.as_tensor(batch["global_index"]).long())
        well_ids.extend(list(batch["well_id"]))
    return _site_metrics(
        torch.cat(logits), torch.cat(labels), torch.cat(experiments), well_ids,
        torch.cat(sites), torch.cat(global_indices),
    )


def _agreement_summary(site_predictions, well_predictions):
    """Describe two-site concordance and how mean-logit pooling changes decisions."""
    grouped = site_predictions.groupby("well_id").agg(
        experiment=("experiment", "first"),
        n_sites=("site", "size"),
        site_prediction_count=("prediction", "nunique"),
        site_correct_count=("correct_top1", "sum"),
    ).reset_index()
    wells = well_predictions[["well_id", "correct_top1"]].rename(
        columns={"correct_top1": "well_correct_top1"})
    joined = grouped.merge(wells, on="well_id", validate="one_to_one")

    def summarize(frame):
        two = frame[frame.n_sites == 2]
        payload = {
            "n_wells": len(frame),
            "n_one_site_wells": int((frame.n_sites == 1).sum()),
            "n_two_site_wells": len(two),
        }
        if two.empty:
            payload.update({
                "two_site_prediction_agreement": None,
                "both_sites_correct": None,
                "exactly_one_site_correct": None,
                "neither_site_correct": None,
                "well_top1_on_two_site_wells": None,
                "well_correct_when_neither_site_correct": None,
                "well_incorrect_when_at_least_one_site_correct": None,
            })
            return payload
        payload.update({
            "two_site_prediction_agreement": float((two.site_prediction_count == 1).mean()),
            "both_sites_correct": float((two.site_correct_count == 2).mean()),
            "exactly_one_site_correct": float((two.site_correct_count == 1).mean()),
            "neither_site_correct": float((two.site_correct_count == 0).mean()),
            "well_top1_on_two_site_wells": float(two.well_correct_top1.mean()),
            "well_correct_when_neither_site_correct": float(
                (two.well_correct_top1 & (two.site_correct_count == 0)).mean()),
            "well_incorrect_when_at_least_one_site_correct": float(
                (~two.well_correct_top1 & (two.site_correct_count >= 1)).mean()),
        })
        return payload

    output = summarize(joined)
    output["per_experiment"] = {
        str(int(experiment)): summarize(group)
        for experiment, group in joined.groupby("experiment")
    }
    return output


def evaluate_checkpoint(result_root, run_id, batch_size=256, num_workers=6,
                        device_name="cuda", include_train=False):
    root = Path(result_root).expanduser().resolve()
    metrics_path = root / "runs" / f"{run_id}.site_metrics.json"
    predictions_path = root / "runs" / f"{run_id}.site_predictions.parquet"
    if metrics_path.is_file() and predictions_path.is_file():
        payload = json.loads(metrics_path.read_text())
        if payload.get("run_id") != run_id:
            raise ValueError(f"site metrics run ID mismatch in {metrics_path}")
        print(f"[skip] complete site evaluation {run_id}", flush=True)
        return payload

    _manifest, registry, run_spec, split_spec = _load_spec(root, run_id)
    result_path = root / "runs" / f"{run_id}.json"
    checkpoint_path = root / "runs" / f"{run_id}.checkpoint.pt"
    well_predictions_path = root / "runs" / f"{run_id}.predictions.parquet"
    if not result_path.is_file() or not checkpoint_path.is_file() or not well_predictions_path.is_file():
        raise FileNotFoundError(f"completed run artifacts are missing for {run_id}")
    result = json.loads(result_path.read_text())
    if result.get("canary") or run_spec.get("canary"):
        raise ValueError("site backfill is only defined for full evaluation runs")

    image_size = int(run_spec.get("image_size", 224))
    assignment, loaders = _make_loaders(
        root, registry, split_spec, int(batch_size), int(num_workers), image_size, canary=False)
    if result.get("split_hash") != _split_hash(assignment):
        raise ValueError(f"split hash changed since training for {run_id}")

    model_kind = str(result["model"]).removeprefix("mae_")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model, model_audit = build_study_model(model_kind, EXPECTED_TREATMENTS, image_size)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("run_id") != run_id or checkpoint.get("split_id") != split_spec["split_id"]:
        raise ValueError(f"checkpoint identity mismatch for {run_id}")
    if checkpoint.get("model_audit", {}).get("total_params") != model_audit.get("total_params"):
        raise ValueError(f"checkpoint/model parameter audit mismatch for {run_id}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)

    well_predictions = pd.read_parquet(well_predictions_path)
    role_loaders = {
        "iid_validation": loaders["iid_validation"],
        "target": loaders["target"],
    }
    if include_train:
        role_loaders = {"train": loaders["train_eval"], **role_loaders}
    role_payload, prediction_rows = {}, []
    started = time.time()
    for role, loader in role_loaders.items():
        metrics, predictions = evaluate_sites(model, loader, device)
        predictions.insert(0, "role", role)
        prediction_rows.append(predictions)
        role_wells = well_predictions[well_predictions.role == role]
        if metrics["n_wells"] != len(role_wells):
            raise ValueError(f"site/well evaluation count mismatch for {run_id} {role}")
        role_payload[role] = {
            **metrics,
            "agreement": _agreement_summary(predictions, role_wells),
        }
        print(f"[site-eval] {run_id} {role} top1={metrics['top1']:.6f}", flush=True)

    site_predictions = pd.concat(prediction_rows, ignore_index=True)
    site_predictions.insert(0, "run_id", run_id)
    _atomic_parquet(predictions_path, site_predictions)
    sha, dirty = _git_info()
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "split_id": split_spec["split_id"],
        "split_hash": result["split_hash"],
        "model": result["model"],
        "best_epoch": result["best_epoch"],
        "checkpoint": str(checkpoint_path),
        "well_predictions": str(well_predictions_path),
        "site_predictions": str(predictions_path),
        "includes_train_sites": bool(include_train),
        "roles": role_payload,
        "training_git_commit": result.get("git_commit"),
        "evaluation_git_commit": sha,
        "evaluation_git_dirty": dirty,
        "elapsed_seconds": time.time() - started,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    _atomic_json(metrics_path, payload)
    return payload


def _eligible_specs(root):
    manifest_path = root / "wave_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"wave manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    return [spec for spec in manifest["runs"] if not spec.get("canary")]


def _batch_evaluate(args):
    root = Path(args.result_root).expanduser().resolve()
    all_specs = _eligible_specs(root)
    selected = [spec for index, spec in enumerate(all_specs)
                if index % int(args.num_shards) == int(args.shard_index)]
    ready = [spec for spec in selected
             if (root / "runs" / f"{spec['run_id']}.json").is_file()
             and (root / "runs" / f"{spec['run_id']}.checkpoint.pt").is_file()]
    pending = [spec for spec in ready
               if not (root / "runs" / f"{spec['run_id']}.site_metrics.json").is_file()
               or not (root / "runs" / f"{spec['run_id']}.site_predictions.parquet").is_file()]
    print(f"[site-eval] shard={args.shard_index}/{args.num_shards} "
          f"eligible={len(selected)} ready={len(ready)} pending={len(pending)}", flush=True)
    if args.status:
        return
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required for checkpoint evaluation")
    running = {}
    logs = root / "logs"; logs.mkdir(parents=True, exist_ok=True)
    failures = root / "site_failures"; failures.mkdir(parents=True, exist_ok=True)
    while pending or running:
        while pending and len(running) < min(int(args.max_concurrent), len(gpus)):
            gpu = gpulease.acquire_any(gpus)
            if gpu is None:
                break
            spec = pending.pop(0); run_id = spec["run_id"]
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--result-root", str(root), "--run-id", run_id,
                "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers),
            ]
            if args.include_train:
                command.append("--include-train")
            log_path = logs / f"site_eval_{run_id}.log"
            handle = open(log_path, "a")  # noqa: SIM115
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, run_id, handle, log_path, command)
            print(f"[start] gpu={gpu} pid={process.pid} site_eval_{run_id}", flush=True)
        for gpu in list(running):
            process, run_id, handle, log_path, command = running[gpu]
            if process.poll() is None:
                continue
            handle.close(); gpulease.release(gpu, pid=process.pid)
            print(f"[exit] gpu={gpu} rc={process.returncode} site_eval_{run_id}", flush=True)
            output = root / "runs" / f"{run_id}.site_metrics.json"
            if process.returncode or not output.is_file():
                _atomic_json(failures / f"{run_id}.json", {
                    "run_id": run_id, "returncode": process.returncode,
                    "log": str(log_path), "command": command, "time": time.time(),
                })
                raise RuntimeError(f"site evaluation failed for {run_id}; see {log_path}")
            del running[gpu]
        if pending or running:
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-train", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    if args.run_id:
        payload = evaluate_checkpoint(
            args.result_root, args.run_id, args.batch_size, args.num_workers, args.device,
            args.include_train)
        print(json.dumps({"run_id": payload["run_id"], "complete": True,
                          "target_site_top1": payload["roles"]["target"]["top1"]},
                         indent=2, sort_keys=True))
    else:
        _batch_evaluate(args)


if __name__ == "__main__":
    main()
