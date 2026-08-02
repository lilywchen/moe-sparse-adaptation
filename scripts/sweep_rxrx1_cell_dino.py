#!/usr/bin/env python
"""Exploratory native-CP5 Cell-DINO factorial with continuous GPU refill.

The earlier canonical hypothesis90 comparison is a valid negative for that exact recipe. This
campaign asks a new question: whether placement, routing granularity, router geometry, or training
pressure exposes a materially stronger sparse advantage. It deliberately reuses the original
3x2x2x3 theory-driven factorial, but runs only RxRx1, only seed 0, and only the validated native-CP5
Cell-DINO data/model family.

There are 43 arms: 36 MoE, six placement/pressure-matched dense-wide controls, and one original
reference. All record milestones at 10/30/60 and save a final checkpoint. Dense controls and the
original also save 10/30 checkpoints so every sparse candidate has reloadable paired anchors.
"""
import argparse
import itertools
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


CONFIG = "configs/ccas_rxrx1_cell_dino_native.yaml"
RESULT_ROOT = Path(os.environ.get(
    "MOE_RX_FACTORIAL_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/factorial60_20260801",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-factorial60-20260801"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/factorial60_20260801"
)

PLACEMENTS = ("early", "middle", "late")
ROUTING_UNITS = ("image", "token")
GEOMETRIES = ("linear", "cosine")
PRESSURES = ("canonical", "route", "output")


def _common():
    return [
        "seed=0", "stage=1", "model.n_experts=8", "model.top_k=1",
        "model.freeze_backbone=false", "model.unfreeze_last_n_blocks=0",
        "model.sym_break_moe=0.0", "train.objective=erm", "train.epochs=60",
        "train.milestone_epochs=[10,30,60]", "train.warmup_epochs=5",
        "train.llrd=1.0", "train.batch_size=64", "train.optim.lr=1.0e-4",
        "train.optim.weight_decay=0.05", "model.drop_path=0.1",
        "analysis.run_mechanism=false", "run_tag=factorial60_20260801",
    ]


def cells(config=CONFIG):
    rows = []
    common = _common()
    for placement, unit, geometry, pressure in itertools.product(
            PLACEMENTS, ROUTING_UNITS, GEOMETRIES, PRESSURES):
        balance = "within_environment" if pressure == "route" else "global"
        overrides = [
            *common, "model.variant=moe", f"model.placement={placement}",
            f"model.routing_unit={unit}", f"model.geometry={geometry}",
            f"model.pressure={pressure}", f"model.balance={balance}",
            "train.save_checkpoint_epochs=[60]",
        ]
        cfg = apply_overrides(load_config(config), overrides)
        tag = f"moe_{placement}_{unit}_{geometry}_{pressure}"
        rows.append((tag, overrides, run_id_from(cfg)))

    for placement, pressure in itertools.product(PLACEMENTS, ("canonical", "output")):
        overrides = [
            *common, "model.variant=dense_wide", f"model.placement={placement}",
            f"model.pressure={pressure}", "model.balance=global",
            "train.save_checkpoint_epochs=[10,30,60]",
        ]
        cfg = apply_overrides(load_config(config), overrides)
        rows.append((f"dense_{placement}_{pressure}", overrides, run_id_from(cfg)))

    original_overrides = [
        *common, "model.variant=original", "model.pressure=canonical",
        "model.balance=global", "train.save_checkpoint_epochs=[10,30,60]",
    ]
    original_cfg = apply_overrides(load_config(config), original_overrides)
    rows.append(("original", original_overrides, run_id_from(original_cfg)))
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
    pending = [row for row in rows if not (out / f"{row[2]}.json").exists()]
    if args.dry_run:
        print(f"factorial60 shard {args.shard_index}/{args.num_shards}: "
              f"{len(rows)} planned, {len(pending)} pending -> {out}")
        print(f"W&B group: {WANDB_GROUP}; HF prefix: {HF_PREFIX}")
        for tag, _, run_id in rows:
            print(f"  {tag}: {run_id}")
        return

    slots = [gpu.strip() for gpu in args.gpus.split(",")]
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:
                break
            tag, overrides, run_id = pending.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            env["WANDB_GROUP"] = WANDB_GROUP
            env["WANDB_JOB_TYPE"] = "rxrx1_factorial60"
            env["WANDB_TAGS"] = "rxrx1,cell-dino,factorial60,exploratory,ood-test-blind"
            env["CCAS_HF_PREFIX"] = HF_PREFIX
            # Do not hold a GPU while multi-GB checkpoints upload. The steward validates and
            # publishes completed folders asynchronously after the training process releases it.
            env["HF_TOKEN"] = ""
            log_handle = open(out / f"{run_id}.log", "a")
            command = [
                sys.executable, "scripts/run_ccas.py", "--config", args.config,
                "--results-dir", str(out), "--override", *overrides,
            ]
            process = subprocess.Popen(
                command, env=env, stdout=log_handle, stderr=subprocess.STDOUT
            )
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, run_id, tag, log_handle)
            print(f"[start] shard={args.shard_index} gpu={gpu} pid={process.pid} "
                  f"{tag} {run_id}", flush=True)

        for gpu in list(running):
            process, run_id, tag, log_handle = running[gpu]
            if process.poll() is not None:
                log_handle.close()
                gpulease.release(gpu, pid=process.pid)
                print(f"[exit] gpu={gpu} rc={process.returncode} {tag} {run_id}", flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()
