#!/usr/bin/env python
"""One-command, two-GPU launcher for parallel full-horizon HUVEC recipe studies."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.certify_rxrx1_huvec_recipe import default_recipes, format_status
from scripts.run_rxrx1_huvec_study import _atomic_json

DEFAULT_RESULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/huvec_systematic_fast_20260814"
)


def pair_plan(pair_index):
    pair_index = int(pair_index)
    if pair_index not in range(3):
        raise ValueError("pair-index must be 0, 1, or 2")
    run_name = f"parallel_recipe_{pair_index + 1}"
    return [
        {
            "model": "resnet18", "gpu": "0", "run_name": run_name,
            "recipe": default_recipes("resnet18")[pair_index],
        },
        {
            "model": "vit_tiny", "gpu": "1", "run_name": run_name,
            "recipe": default_recipes("vit_tiny")[pair_index],
        },
    ]


def _status_entries(result_root):
    root = Path(result_root)
    entries = []
    for pair_index in range(3):
        for item in pair_plan(pair_index):
            output_dir = (root / "recipe_certification" / item["run_name"] /
                          item["model"])
            path = output_dir / "status.json"
            payload = json.loads(path.read_text()) if path.is_file() else None
            certificate = output_dir / "CERTIFIED_RECIPE.json"
            if (payload and payload.get("elapsed_seconds") is None
                    and certificate.is_file()):
                legacy = json.loads(certificate.read_text())
                payload = {
                    **payload,
                    "elapsed_seconds": legacy.get("elapsed_seconds_this_attempt"),
                }
            entries.append((item["run_name"], item["model"], payload))
    return entries


def _status_signature(payload):
    if payload is None:
        return None
    return (
        payload.get("state"), payload.get("attempt_name"), payload.get("epoch"),
        payload.get("certified_at_epoch"), payload.get("updated_at"),
    )


def _print_entry(run_name, model, payload):
    print(f"===== {run_name} / {model} =====", flush=True)
    print(format_status(payload), flush=True)


def _all_status(result_root):
    for run_name, model, payload in _status_entries(result_root):
        _print_entry(run_name, model, payload)


def _watch(result_root, interval):
    previous = {}
    terminal_states = {"complete", "failed", "interrupted"}
    while True:
        entries = _status_entries(result_root)
        changed = []
        for run_name, model, payload in entries:
            key = (run_name, model)
            signature = _status_signature(payload)
            if key not in previous or previous[key] != signature:
                changed.append((run_name, model, payload))
            previous[key] = signature
        if changed:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
            for run_name, model, payload in changed:
                _print_entry(run_name, model, payload)
        states = [payload.get("state") for _, _, payload in entries if payload]
        if len(states) == 6 and all(state in terminal_states for state in states):
            print("[complete] all six recipe candidates reached a terminal state", flush=True)
            return
        time.sleep(max(float(interval), 1.0))


def launch_pair(result_root, pair_index):
    result_root = Path(result_root).expanduser().resolve()
    plan = pair_plan(pair_index)
    recipe_dir = result_root / "recipe_specs"
    log_dir = result_root / "logs"
    launcher_dir = result_root / "recipe_certification" / "launchers"
    for directory in (recipe_dir, log_dir, launcher_dir):
        directory.mkdir(parents=True, exist_ok=True)

    processes = []
    host = socket.gethostname()
    run_name = plan[0]["run_name"]
    launcher_path = launcher_dir / f"{run_name}.json"
    for item in plan:
        model = item["model"]
        recipe_path = recipe_dir / f"{run_name}_{model}.json"
        _atomic_json(recipe_path, [item["recipe"]])
        log_path = log_dir / f"{run_name}_{model}.log"
        log_handle = open(log_path, "a")  # noqa: SIM115
        command = [
            sys.executable, str(ROOT / "scripts" / "certify_rxrx1_huvec_recipe.py"),
            "--result-root", str(result_root), "--run-name", run_name,
            "--model", model, "--train-threshold", "0.80",
            "--recipes-json", str(recipe_path),
        ]
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES=item["gpu"], PYTHONUNBUFFERED="1")
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log_handle,
            stderr=subprocess.STDOUT)
        processes.append({
            **item, "process": process, "log_handle": log_handle,
            "log": str(log_path), "command": command,
        })
        print(
            f"[start] gpu={item['gpu']} pid={process.pid} model={model} "
            f"recipe={item['recipe']['name']} log={log_path}", flush=True)

    _atomic_json(launcher_path, {
        "state": "running", "hostname": host, "launcher_pid": os.getpid(),
        "run_name": run_name, "started_at": time.time(),
        "children": [
            {key: value for key, value in item.items()
             if key not in {"process", "log_handle", "command"}}
            | {"pid": item["process"].pid}
            for item in processes
        ],
    })

    stopping = False

    def terminate(_signum=None, _frame=None):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print("[stop] terminating both full-horizon training children", flush=True)
        for item in processes:
            if item["process"].poll() is None:
                item["process"].terminate()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    returncodes = {}
    try:
        while len(returncodes) < len(processes):
            for item in processes:
                process = item["process"]
                if item["model"] in returncodes or process.poll() is None:
                    continue
                returncodes[item["model"]] = int(process.returncode)
                item["log_handle"].close()
                print(
                    f"[exit] gpu={item['gpu']} model={item['model']} "
                    f"rc={process.returncode} log={item['log']}", flush=True)
            if len(returncodes) < len(processes):
                time.sleep(2)
    finally:
        terminate()
        for item in processes:
            process = item["process"]
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not item["log_handle"].closed:
                item["log_handle"].close()
    state = "complete" if all(code == 0 for code in returncodes.values()) else "ended"
    _atomic_json(launcher_path, {
        "state": state, "hostname": host, "launcher_pid": os.getpid(),
        "run_name": run_name, "ended_at": time.time(), "returncodes": returncodes,
    })
    if any(code != 0 for code in returncodes.values()):
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--pair-index", type=int, choices=(0, 1, 2))
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=float, default=30.0)
    args = parser.parse_args()
    if args.status or args.watch:
        if args.pair_index is not None:
            parser.error("--pair-index is not used with --status or --watch")
        if args.watch:
            _watch(args.result_root, args.watch_interval)
        else:
            _all_status(args.result_root)
        return
    if args.pair_index is None:
        parser.error("--pair-index is required for launch")
    launch_pair(args.result_root, args.pair_index)


if __name__ == "__main__":
    main()
