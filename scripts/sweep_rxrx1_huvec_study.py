#!/usr/bin/env python
"""Resumable three-container launcher for the RxRx1 HUVEC systematic fast study."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.utils import gpulease
from scripts.aggregate_rxrx1_huvec_study import (
    aggregate,
    should_run_parameter_match,
    status_table,
)
from scripts.prepare_rxrx1_huvec_study import (
    DEFAULT_CONFIG,
    DEFAULT_RESULT_ROOT,
    atomic_json,
)


def _source_identity():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip())
    return sha, dirty


def _run_id(stage, model, split_id, suffix=""):
    value = f"huvec_{stage}_{model}_{split_id}{suffix}"
    return value.replace("/", "-").replace(" ", "_")


def planned_runs(registry):
    main = registry["main_training_splits"]
    by_id = {row["split_id"]: row for row in main}
    primary = [row for row in main if row["kind"] == "primary"]
    controlled = [row for row in main if row["kind"] == "controlled"]
    if len(primary) != 3 or len(controlled) != 9:
        raise ValueError("frozen training registry must contain 3 primary and 9 controlled splits")
    dev = next(row for row in primary if int(row["fold"]) == 0)
    rows = []

    def add(stage, model, split, **extra):
        defaults = {
            "stage": stage, "model": model, "split_id": split["split_id"], "seed": 0,
            "image_size": 224, "batch_size": 128, "num_workers": 6,
            "epochs": 30 if model == "resnet18" else 40,
        }
        defaults.update(extra)
        defaults["run_id"] = _run_id(stage, model, split["split_id"],
                                      "_canary" if defaults.get("canary") else "")
        rows.append(defaults)

    add("canary", "resnet18", dev, canary=True, canary_steps=3000, epochs=0)
    add("canary", "vit_tiny", dev, canary=True, canary_steps=3000, epochs=0)
    for split in main:
        add("F_G", "resnet18", split, epochs=30)
    add("F_G", "vit_tiny", dev, epochs=40)

    # The dense development-fold result above is reused; H adds its paired MoE and both models on
    # every other primary/controlled split.
    add("H", "vit_tiny_moe", dev, epochs=40)
    for split in primary:
        if split["split_id"] == dev["split_id"]:
            continue
        add("H", "vit_tiny", split, epochs=40)
        add("H", "vit_tiny_moe", split, epochs=40)
    for split in controlled:
        add("H", "vit_tiny", split, epochs=40)
        add("H", "vit_tiny_moe", split, epochs=40)

    for split in primary:
        add("I", "vit_tiny_dense_matched", split, epochs=40)

    high = [row for row in controlled if row["difficulty_tier"] == "high"]
    hardest = max(high, key=lambda row: next(iter(row["target_difficulty"].values())))
    add("J", "mae_vit_tiny", hardest, epochs=30, pretrain_epochs=20, batch_size=128)
    add("J", "mae_vit_tiny_moe", hardest, epochs=30, pretrain_epochs=20, batch_size=128)
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("HUVEC study run IDs collide")
    if {row["split_id"] for row in rows} - set(by_id):
        raise ValueError("a planned run names an unfrozen split")
    return rows


def write_wave_manifest(result_root, registry):
    root = Path(result_root); root.mkdir(parents=True, exist_ok=True)
    rows = planned_runs(registry)
    sha, dirty = _source_identity()
    if dirty:
        raise RuntimeError("refuse to freeze HUVEC wave from a dirty tracked checkout")
    payload = {
        "schema_version": 1, "campaign": "rxrx1_huvec_systematic_fast_20260814",
        "expected_runs": len(rows), "source_git_commit": sha, "source_git_dirty": dirty,
        "result_root": str(root.resolve()), "cell_type": "HUVEC", "seed_policy": "one seed only",
        "input": "official native six-channel RxRx1", "training_unit": "site",
        "evaluation_unit": "well; mean of two site logits",
        "checkpoint_rule": "best source-IID well accuracy; target never selects checkpoints",
        "parameter_match_gate": "launch iff mean primary-fold MoE minus dense target accuracy > 0",
        "stages": {
            stage: sum(row["stage"] == stage for row in rows)
            for stage in ("canary", "F_G", "H", "I", "J")},
        "runs": rows,
    }
    atomic_json(root / "wave_manifest.json", payload)
    return payload


def _failure_path(root, run_id):
    return Path(root) / "failures" / f"{run_id}.json"


def _run_tasks(tasks, result_root, gpus, max_concurrent, command_builder):
    root = Path(result_root); (root / "logs").mkdir(parents=True, exist_ok=True)
    pending = [task for task in tasks if not command_builder(task)[1].is_file()]
    running = {}
    while pending or running:
        while pending and len(running) < int(max_concurrent):
            gpu = gpulease.acquire_any(gpus)
            if gpu is None:
                break
            task = pending.pop(0)
            command, output_path, task_id = command_builder(task)
            log_path = root / "logs" / f"{task_id}.log"
            # Kept open until the child exits; the scheduler closes it in the reap path.
            log_handle = open(log_path, "a")  # noqa: SIM115
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log_handle,
                                       stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, task_id, output_path, log_handle, command)
            print(f"[start] gpu={gpu} pid={process.pid} {task_id}", flush=True)
        for gpu in list(running):
            process, task_id, output_path, log_handle, command = running[gpu]
            if process.poll() is None:
                continue
            log_handle.close(); gpulease.release(gpu, pid=process.pid)
            print(f"[exit] gpu={gpu} rc={process.returncode} {task_id}", flush=True)
            if process.returncode or not output_path.is_file():
                atomic_json(_failure_path(root, task_id), {
                    "task_id": task_id, "returncode": process.returncode,
                    "expected_output": str(output_path), "command": command,
                    "log": str(root / "logs" / f"{task_id}.log"), "time": time.time(),
                })
                raise RuntimeError(f"task failed: {task_id}; see persistent log")
            del running[gpu]
        if pending or running:
            time.sleep(5)


def _wait_for_outputs(result_root, outputs, label, timeout_hours=30):
    root = Path(result_root); started = time.time(); last_report = 0
    outputs = list(map(Path, outputs))
    while True:
        failures = list((root / "failures").glob("*.json")) if (root / "failures").is_dir() else []
        if failures:
            raise RuntimeError(f"shared study failure detected: {failures[0]}")
        complete = sum(path.is_file() for path in outputs)
        if complete == len(outputs):
            print(f"[ready] {label}: {complete}/{len(outputs)}", flush=True); return
        if time.time() - started > float(timeout_hours) * 3600:
            raise TimeoutError(f"timed out waiting for {label}: {complete}/{len(outputs)}")
        if time.time() - last_report >= 60:
            print(f"[wait] {label}: {complete}/{len(outputs)}", flush=True); last_report = time.time()
        time.sleep(10)


def _stage_tasks(manifest, stage, shard_index, num_shards):
    rows = [row for row in manifest["runs"] if row["stage"] == stage]
    return rows, [row for index, row in enumerate(rows) if index % int(num_shards) == int(shard_index)]


def _run_stage(result_root, manifest, stage, shard_index, num_shards, gpus, max_concurrent):
    root = Path(result_root)
    all_rows, local_rows = _stage_tasks(manifest, stage, shard_index, num_shards)
    print(f"[stage] {stage}: local={len(local_rows)} global={len(all_rows)}", flush=True)

    def command(task):
        output = root / "runs" / f"{task['run_id']}.json"
        return ([sys.executable, "scripts/run_rxrx1_huvec_study.py",
                 "--result-root", str(root), "--run-id", task["run_id"]],
                output, task["run_id"])

    _run_tasks(local_rows, root, gpus, max_concurrent, command)
    _wait_for_outputs(root, [root / "runs" / f"{row['run_id']}.json" for row in all_rows], stage)


def _certification_gate(result_root, manifest):
    root = Path(result_root)
    canaries = [row for row in manifest["runs"] if row["stage"] == "canary"]
    canary_ok = all(json.loads((root / "runs" / f"{row['run_id']}.json").read_text())
                     .get("canary_passed") for row in canaries)
    dev = [row for row in manifest["runs"] if row["stage"] == "F_G"
           and row["model"] == "vit_tiny"]
    resnet = [row for row in manifest["runs"] if row["stage"] == "F_G"
              and row["model"] == "resnet18" and row["split_id"] == "primary_fold0"]
    model_ok = bool(dev and resnet and all(
        json.loads((root / "runs" / f"{row['run_id']}.json").read_text())
        .get("training_certified") for row in dev + resnet))
    return canary_ok and model_ok, {"canaries_pass": canary_ok,
                                    "dense_vit_and_resnet_certified": model_ok}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    root = Path(args.result_root).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.status:
        print(status_table(root)); return
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 2:
        parser.error("the three-container fast path expects exactly two GPUs per container")

    # A: one frozen metadata manifest. The other launchers wait instead of racing its path audit.
    site_manifest = root / "data" / "huvec_sites.parquet"
    if args.shard_index == 0 and not site_manifest.is_file():
        subprocess.run([
            sys.executable, "scripts/prepare_rxrx1_huvec_study.py", "--result-root", str(root),
            "--config", args.config, "--build-manifest"], cwd=ROOT, check=True)
    _wait_for_outputs(root, [site_manifest], "frozen HUVEC site manifest", timeout_hours=2)

    # B: six extraction shards, one per physical GPU across the three containers.
    extraction_count = int(args.num_shards) * len(gpus)
    extraction_tasks = [args.shard_index * len(gpus) + index for index in range(len(gpus))]
    def extraction_command(index):
        output = root / "cache" / f"cell_dino_qc_shard{index:02d}-of-{extraction_count:02d}.parquet"
        return ([sys.executable, "scripts/prepare_rxrx1_huvec_study.py",
                 "--result-root", str(root), "--config", args.config,
                 "--extract-shard", str(index), "--num-extraction-shards", str(extraction_count),
                 "--batch-size", "128", "--num-workers", "6"],
                output, f"prepare_embedding_shard{index:02d}")
    _run_tasks(extraction_tasks, root, gpus, 2, extraction_command)
    extraction_outputs = [
        root / "cache" / f"cell_dino_qc_shard{index:02d}-of-{extraction_count:02d}.parquet"
        for index in range(extraction_count)]
    _wait_for_outputs(root, extraction_outputs, "Cell-DINO/QC extraction")

    # C-E: a single deterministic finalizer freezes folds/resamples and runs cheap probes.
    prepared = root / "PREPARED.json"
    if args.shard_index == 0 and not prepared.is_file():
        subprocess.run([
            sys.executable, "scripts/prepare_rxrx1_huvec_study.py", "--result-root", str(root),
            "--num-extraction-shards", str(extraction_count), "--finalize"],
            cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[0]), check=True)
    _wait_for_outputs(root, [prepared, root / "study_registry.json"], "frozen study registry",
                      timeout_hours=6)
    registry = json.loads((root / "study_registry.json").read_text())
    if args.shard_index == 0 and not (root / "wave_manifest.json").is_file():
        write_wave_manifest(root, registry)
    _wait_for_outputs(root, [root / "wave_manifest.json"], "wave manifest")
    manifest = json.loads((root / "wave_manifest.json").read_text())
    if args.dry_run:
        for stage in ("canary", "F_G", "H", "I", "J"):
            all_rows, local = _stage_tasks(manifest, stage, args.shard_index, args.num_shards)
            print(f"{stage}: local {len(local)} / global {len(all_rows)}")
            for row in local:
                print(f"  {row['run_id']}")
        return
    if args.prepare_only:
        print(status_table(root)); return

    _run_stage(root, manifest, "canary", args.shard_index, args.num_shards,
               gpus, args.max_concurrent)
    if not all(json.loads((root / "runs" / f"{row['run_id']}.json").read_text())
               .get("canary_passed") for row in manifest["runs"] if row["stage"] == "canary"):
        if args.shard_index == 0:
            atomic_json(root / "STOPPED.json", {"stage": "canary", "reason": "memorization gate"})
            aggregate(root)
        return

    _run_stage(root, manifest, "F_G", args.shard_index, args.num_shards,
               gpus, args.max_concurrent)
    certified, detail = _certification_gate(root, manifest)
    if not certified:
        if args.shard_index == 0:
            atomic_json(root / "STOPPED.json", {"stage": "F_G", "reason": "training certification",
                                                "detail": detail})
            aggregate(root)
        return

    _run_stage(root, manifest, "H", args.shard_index, args.num_shards,
               gpus, args.max_concurrent)
    run_match, match_detail = should_run_parameter_match(root)
    if args.shard_index == 0:
        atomic_json(root / "PARAMETER_MATCH_GATE.json", {"run": run_match, **match_detail})
    if run_match:
        _run_stage(root, manifest, "I", args.shard_index, args.num_shards,
                   gpus, args.max_concurrent)
    elif args.shard_index == 0:
        atomic_json(root / "SKIPPED_I.json", {"reason": "MoE mean primary-fold gain was not positive",
                                              **match_detail})

    _run_stage(root, manifest, "J", args.shard_index, args.num_shards,
               gpus, args.max_concurrent)
    if args.shard_index == 0:
        aggregate(root, require_complete=True)
    _wait_for_outputs(root, [root / "AGGREGATED.json"], "final aggregation")
    print(status_table(root), flush=True)


if __name__ == "__main__":
    main()
