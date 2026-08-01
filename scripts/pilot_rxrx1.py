#!/usr/bin/env python
"""Minimal RxRx1 competence and dense-vs-MoE kill test.

This deliberately replaces the old architecture grid.  It launches only experiments that answer
the next scientific decision:

  competence: frozen-backbone probe plus two full-fine-tuning learning rates;
  kill:       original reference, total-parameter-matched dense-wide, canonical learned MoE;
  replicate:  paired dense-wide/MoE seeds 1 and 2 after a seed-0 signal.

Every run is idempotent, selection-only (OOD val), and uses the shared global GPU lease.
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


CONFIG = "configs/ccas_rxrx1_cell_dino.yaml"
RESULT_ROOT = Path(os.environ.get(
    "MOE_RX_KILL_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/kill_rxrx1",
))


def _rows(phase, lr=1.0e-4, epochs=10, config=CONFIG):
    if phase == "instrument":
        # Minimal representation-vs-adaptation pair used to qualify a new channel interface.
        specs = [
            ("instrument_linear_probe", ["model.variant=original", "model.freeze_backbone=true",
                                          "train.epochs=5", "train.warmup_epochs=0",
                                          "train.optim.lr=1.0e-3", "train.llrd=1.0", "seed=0"]),
            ("instrument_full_ft_lr1e-4", ["model.variant=original",
                                            "model.freeze_backbone=false",
                                            "train.epochs=10", "train.optim.lr=1.0e-4",
                                            "train.llrd=1.0", "seed=0"]),
        ]
    elif phase == "competence":
        specs = [
            ("linear_probe", ["model.variant=original", "model.freeze_backbone=true",
                              "train.epochs=5", "train.warmup_epochs=0",
                              "train.optim.lr=1.0e-3", "train.llrd=1.0", "seed=0"]),
            ("full_ft_lr1e-4", ["model.variant=original", "model.freeze_backbone=false",
                                "train.optim.lr=1.0e-4", "train.llrd=1.0", "seed=0"]),
            ("full_ft_lr3e-4", ["model.variant=original", "model.freeze_backbone=false",
                                "train.optim.lr=3.0e-4", "train.llrd=1.0", "seed=0"]),
        ]
    elif phase == "kill":
        specs = [
            ("kill_original", ["model.variant=original", "seed=0",
                               f"train.optim.lr={lr}", f"train.epochs={epochs}"]),
            ("kill_dense_wide", ["model.variant=dense_wide", "seed=0",
                                 f"train.optim.lr={lr}", f"train.epochs={epochs}"]),
            ("kill_moe", ["model.variant=moe", "seed=0",
                          f"train.optim.lr={lr}", f"train.epochs={epochs}"]),
        ]
    elif phase == "replicate":
        specs = []
        for seed in (1, 2):
            specs += [
                (f"rep_dense_wide_s{seed}", ["model.variant=dense_wide", f"seed={seed}"]),
                (f"rep_moe_s{seed}", ["model.variant=moe", f"seed={seed}"]),
            ]
        specs = [(tag, [*ov, f"train.optim.lr={lr}", f"train.epochs={epochs}"])
                 for tag, ov in specs]
    else:
        raise ValueError(phase)

    rows = []
    for tag, overrides in specs:
        overrides = [*overrides, f"run_tag={tag}"]
        cfg = apply_overrides(load_config(config), overrides)
        rows.append((tag, overrides, run_id_from(cfg)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase", choices=("instrument", "competence", "kill", "replicate"), required=True)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--result-root", default=None)
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.0e-4,
                    help="frozen recipe LR for kill/replicate")
    ap.add_argument("--epochs", type=int, default=10,
                    help="frozen recipe epochs for kill/replicate")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result_root = Path(args.result_root) if args.result_root else RESULT_ROOT
    out = result_root / args.phase
    out.mkdir(parents=True, exist_ok=True)
    rows = _rows(args.phase, lr=args.lr, epochs=args.epochs, config=args.config)
    pending = [row for row in rows if not (out / f"{row[2]}.json").exists()]
    if args.dry_run:
        print(f"RxRx1 {args.phase}: {len(rows)} planned, {len(pending)} pending -> {out}")
        for tag, _, rid in rows:
            print(f"  {tag}: {rid}")
        return

    slots = [x.strip() for x in args.gpus.split(",")]
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:
                break
            tag, overrides, rid = pending.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            log = open(out / f"{rid}.log", "a")
            cmd = [sys.executable, "scripts/run_ccas.py", "--config", args.config,
                   "--results-dir", str(out), "--override", *overrides]
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, proc.pid)
            running[gpu] = (proc, rid, log)
            print(f"[start] gpu={gpu} pid={proc.pid} {tag} {rid}", flush=True)
        for gpu in list(running):
            proc, rid, log = running[gpu]
            if proc.poll() is not None:
                log.close()
                gpulease.release(gpu, pid=proc.pid)
                print(f"[exit] {rid} rc={proc.returncode}", flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()
