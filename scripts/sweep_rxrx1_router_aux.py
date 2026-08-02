#!/usr/bin/env python
"""Continuous-refill RxRx1 router-auxiliary screen on the early token/cosine MoE.

The factorial60 screen makes early routing pressure the leading provisional mechanism, while the
exact early-canonical dense trajectory is still pending. This bounded follow-up changes only the
two router auxiliary-loss weights. It crosses canonical/global and route/within-experiment
pressure with eight predeclared (load-balance, router-z-loss) settings. The frozen factorial60
default (1e-2, 1e-3) and its exact dense comparator are shared references and are not duplicated.

All 16 new cells are E8 top-1, seed 0, 60 epochs, and save 10/30/60 checkpoints. They therefore
share total parameters and active compute within each pressure pair; this is a mechanism screen,
not a fresh-seed efficacy confirmation.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config
from scripts.sweep_rxrx1_cell_dino import CONFIG, cells as factorial_cells


RESULT_ROOT = Path(os.environ.get(
    "MOE_RX_ROUTER_AUX_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/router_aux60_20260802",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-router-aux60-20260802"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/router_aux60_20260802"
)
WANDB_JOB_TYPE = "rxrx1_router_aux60"
WANDB_TAGS = "rxrx1,cell-dino,router-aux60,exploratory,ood-test-blind"

PRESSURES = ("route", "canonical")
AUX_SETTINGS = (
    ("bw0_z1em3", 0.0, 1.0e-3),
    ("bw1em4_z1em3", 1.0e-4, 1.0e-3),
    ("bw1em3_z1em3", 1.0e-3, 1.0e-3),
    ("bw1em2_z0", 1.0e-2, 0.0),
    ("bw1em2_z1em4", 1.0e-2, 1.0e-4),
    ("bw1em2_z1em2", 1.0e-2, 1.0e-2),
    ("bw0_z0", 0.0, 0.0),
    ("bw1em3_z0", 1.0e-3, 0.0),
)


def _busy_gpu_indices(gpu_rows, process_rows):
    """Map nvidia-smi compute-process UUIDs back to container-local GPU indices."""
    uuid_to_index = {}
    for row in gpu_rows.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) >= 2 and fields[0] and fields[1]:
            uuid_to_index[fields[1]] = fields[0]
    busy = set()
    for row in process_rows.splitlines():
        uuid = row.split(",", 1)[0].strip()
        if uuid in uuid_to_index:
            busy.add(uuid_to_index[uuid])
    return busy


def idle_nvidia_slots(slots):
    """Return physically idle slots, failing closed if GPU state cannot be inspected."""
    try:
        gpu_rows = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        process_rows = subprocess.run(
            [
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[occupancy-blocked] nvidia-smi inspection failed: {exc}", flush=True)
        return []
    busy = _busy_gpu_indices(gpu_rows, process_rows)
    return [slot for slot in slots if slot not in busy]


def active_marker_live(path):
    """Return whether a restart marker still names a live worker, removing stale markers."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
    except (ValueError, ProcessLookupError):
        path.unlink(missing_ok=True)
        return False
    except PermissionError:
        return True
    return True


def clear_active_marker(path, pid):
    """Clear only the marker owned by the exiting worker."""
    path = Path(path)
    try:
        marker_pid = int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return
    if marker_pid == pid:
        path.unlink(missing_ok=True)


def _base_overrides(config, pressure):
    tag = f"moe_early_token_cosine_{pressure}"
    for row_tag, overrides, _ in factorial_cells(config):
        if row_tag == tag:
            return list(overrides)
    raise RuntimeError(f"missing factorial reference cell {tag}")


def cells(config=CONFIG):
    rows = []
    for pressure in PRESSURES:
        base = _base_overrides(config, pressure)
        for label, balance_w, zloss_w in AUX_SETTINGS:
            run_tag = f"router_aux60_{pressure}_{label}_20260802"
            overrides = [
                *base,
                f"losses.balance_w={balance_w}",
                f"losses.zloss_w={zloss_w}",
                "analysis.run_mechanism=true",
                "train.save_checkpoint_epochs=[10,30,60]",
                f"run_tag={run_tag}",
            ]
            cfg = apply_overrides(load_config(config), overrides)
            rows.append((f"{pressure}_{label}", overrides, run_id_from(cfg)))
    return rows


def sharded_rows(rows, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = Path(args.result_root) if args.result_root else RESULT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    rows = sharded_rows(cells(args.config), args.shard_index, args.num_shards)
    pending = [
        row for row in rows
        if not (out / f"{row[2]}.json").exists()
        and not (out / f"{row[2]}.pruned").exists()
        and not active_marker_live(out / f"{row[2]}.active")
    ]
    if args.dry_run:
        print(f"router_aux60 shard {args.shard_index}/{args.num_shards}: "
              f"{len(rows)} planned, {len(pending)} pending -> {out}")
        print(f"W&B group: {WANDB_GROUP}; HF prefix: {HF_PREFIX}")
        for tag, overrides, run_id in rows:
            balance = next(
                (v for v in overrides if v.startswith("losses.balance_w=")),
                "losses.balance_w=n/a",
            )
            zloss = next(
                (v for v in overrides if v.startswith("losses.zloss_w=")),
                "losses.zloss_w=n/a",
            )
            print(f"  {tag}: {run_id} [{balance}, {zloss}]")
        return

    slots = [gpu.strip() for gpu in args.gpus.split(",")]
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            # gpulease is the coordination layer, while nvidia-smi is the physical source of
            # truth.  Both checks are required because a steward may preserve an adopted worker
            # after its original controller exits, leaving stale or missing lease metadata.
            physically_idle = idle_nvidia_slots(slots)
            gpu = gpulease.acquire_any(physically_idle) if physically_idle else None
            if gpu is None:
                break
            tag, overrides, run_id = pending.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            env["WANDB_GROUP"] = WANDB_GROUP
            env["WANDB_JOB_TYPE"] = WANDB_JOB_TYPE
            env["WANDB_TAGS"] = WANDB_TAGS
            env["CCAS_HF_PREFIX"] = HF_PREFIX
            env["HF_TOKEN"] = ""
            log_handle = open(out / f"{run_id}.log", "a")
            command = [
                sys.executable, "scripts/run_ccas.py", "--config", args.config,
                "--results-dir", str(out), "--override", *overrides,
            ]
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            gpulease.adopt(gpu, process.pid)
            (out / f"{run_id}.active").write_text(str(process.pid))
            running[gpu] = (process, run_id, tag, log_handle)
            print(f"[start] shard={args.shard_index} gpu={gpu} pid={process.pid} "
                  f"{tag} {run_id}", flush=True)

        for gpu in list(running):
            process, run_id, tag, log_handle = running[gpu]
            if process.poll() is not None:
                log_handle.close()
                gpulease.release(gpu, pid=process.pid)
                clear_active_marker(out / f"{run_id}.active", process.pid)
                print(f"[exit] gpu={gpu} rc={process.returncode} {tag} {run_id}", flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()
