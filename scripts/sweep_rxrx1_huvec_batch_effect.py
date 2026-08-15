#!/usr/bin/env python
"""Two-container, four-GPU launcher for the frozen HUVEC batch-effect wave."""
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
from scripts.aggregate_rxrx1_huvec_batch_effect import aggregate, print_status, result_table
from scripts.prepare_rxrx1_huvec_batch_effect import DEFAULT_BASE_ROOT, DEFAULT_RESULT_ROOT, prepare


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def wait_for(path, label, timeout_hours=2):
    path = Path(path); started = time.time(); last = 0
    while not path.is_file():
        if time.time() - started > timeout_hours * 3600:
            raise TimeoutError(f"timed out waiting for {label}: {path}")
        if time.time() - last >= 60:
            print(f"[wait] {label}", flush=True); last = time.time()
        time.sleep(5)
    print(f"[ready] {label}: {path}", flush=True)


def run_tasks(tasks, result_root, gpus, max_concurrent):
    root = Path(result_root); (root / "logs").mkdir(parents=True, exist_ok=True)
    pending = [row for row in tasks if not (root / "runs" / row["run_id"] / "RESULT.json").is_file()]
    running = {}
    print(f"[plan] local pending={len(pending)} complete={len(tasks)-len(pending)}", flush=True)
    while pending or running:
        while pending and len(running) < int(max_concurrent):
            gpu = gpulease.acquire_any(gpus)
            if gpu is None:
                break
            task = pending.pop(0); run_id = task["run_id"]
            run_dir = root / "runs" / run_id; run_dir.mkdir(parents=True, exist_ok=True)
            log_path = root / "logs" / f"{run_id}.log"
            handle = open(log_path, "a")  # noqa: SIM115 - closed after the subprocess exits
            command = [sys.executable, "scripts/run_rxrx1_huvec_batch_effect.py",
                       "--result-root", str(root), "--run-id", run_id]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT)
            if not gpulease.adopt(gpu, process.pid):
                process.terminate(); handle.close()
                raise RuntimeError(f"could not transfer GPU {gpu} lease to {run_id}")
            running[gpu] = (process, task, handle, log_path)
            print(f"[start] gpu={gpu} pid={process.pid} {run_id} "
                  f"model={task['model']} target={task['split_id']}", flush=True)
        for gpu in list(running):
            process, task, handle, log_path = running[gpu]
            if process.poll() is None:
                continue
            handle.close(); gpulease.release(gpu, pid=process.pid)
            run_id = task["run_id"]
            output = root / "runs" / run_id / "RESULT.json"
            print(f"[exit] gpu={gpu} rc={process.returncode} {run_id}", flush=True)
            del running[gpu]
            if process.returncode or not output.is_file():
                atomic_json(root / "failures" / f"{run_id}.json", {
                    "run_id": run_id, "returncode": process.returncode,
                    "expected_output": str(output), "log": str(log_path),
                    "time": time.time(),
                })
                raise RuntimeError(f"{run_id} failed; inspect {log_path}")
        if pending or running:
            time.sleep(5)


def wait_for_wave(result_root, manifest, timeout_hours=18):
    root = Path(result_root); started = time.time(); last = 0
    expected = [root / "runs" / row["run_id"] / "RESULT.json" for row in manifest["runs"]]
    while True:
        failures = list((root / "failures").glob("*.json")) if (root / "failures").is_dir() else []
        if failures:
            raise RuntimeError(f"another shard reported failure: {failures[0]}")
        complete = sum(path.is_file() for path in expected)
        if complete == len(expected):
            print(f"[ready] global wave complete: {complete}/{len(expected)}", flush=True); return
        if time.time() - started > timeout_hours * 3600:
            raise TimeoutError(f"global wave timed out: {complete}/{len(expected)} complete")
        if time.time() - last >= 60:
            print(f"[wait] global wave: {complete}/{len(expected)} complete", flush=True)
            last = time.time()
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-result-root", default=str(DEFAULT_BASE_ROOT))
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != 2 or int(args.max_concurrent) != 2:
        parser.error("the frozen fast path uses exactly two GPUs and two workers per container")
    root = Path(args.result_root).expanduser().resolve()
    if args.status:
        print_status(*result_table(root)[:2]); return

    manifest_path = root / "wave_manifest.json"
    if args.shard_index == 0 and not manifest_path.is_file():
        prepare(args.base_result_root, root)
    wait_for(manifest_path, "frozen 36-run manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest["expected_runs"] != 36 or len(manifest["runs"]) != 36:
        raise RuntimeError("the batch wave must contain exactly 36 frozen runs")
    local = [row for index, row in enumerate(manifest["runs"])
             if index % args.num_shards == args.shard_index]
    if len(local) != 18:
        raise RuntimeError(f"expected 18 runs on this container, found {len(local)}")
    print(f"[plan] shard={args.shard_index}/{args.num_shards} local=18 global=36 "
          f"gpus={','.join(gpus)}", flush=True)
    if args.dry_run:
        for row in local:
            print(f"  {row['run_id']} {row['model']} {row['split_id']}")
        return
    run_tasks(local, root, gpus, args.max_concurrent)
    wait_for_wave(root, manifest)
    report = root / "analysis" / "REPORT.html"
    if args.shard_index == 0 and not report.is_file():
        aggregate(root, require_complete=True)
    wait_for(report, "final statistics and HTML report", timeout_hours=1)
    print_status(*result_table(root)[:2])


if __name__ == "__main__":
    main()
