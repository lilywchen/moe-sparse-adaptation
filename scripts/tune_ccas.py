#!/usr/bin/env python
"""Stage-0 shared full-fine-tuning recipe screen.

This is intentionally a small, predeclared screen rather than per-cell tuning. It varies only
the two optimization quantities most likely to change transfer behavior (base LR and layer-wise
decay), keeps weight decay fixed, and uses approximately 10% warmup for each dataset. Candidate
results live outside the factorial directory and are selected on OOD validation only.

    python scripts/tune_ccas.py --dry-run
    python scripts/tune_ccas.py --dataset rxrx1 --gpus 0,1 --max-concurrent 2
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


DATASETS = {"rxrx1": "configs/ccas_rxrx1.yaml",
            "camelyon17": "configs/ccas_camelyon17.yaml"}
LRS = (3.0e-5, 1.0e-4, 3.0e-4)
LLRDS = (0.70, 0.85)
WARMUP = {"rxrx1": 3, "camelyon17": 1}
HPO_ROOT = Path(os.environ.get(
    "MOE_HPO_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/hpo",
))


def candidates(dataset, seed=0):
    cfg_path = DATASETS[dataset]
    rows = []
    for lr, llrd in itertools.product(LRS, LLRDS):
        tag = f"hpoA_lr{lr:.0e}_llrd{llrd:.2f}".replace("+", "")
        ov = [f"seed={seed}", "model.variant=original", "model.pressure=canonical",
              f"train.optim.lr={lr}", f"train.llrd={llrd}",
              f"train.warmup_epochs={WARMUP[dataset]}", f"run_tag={tag}"]
        rid = run_id_from(apply_overrides(load_config(cfg_path), ov))
        rows.append((cfg_path, ov, rid, lr, llrd))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--shard", default="0/1", help="candidate shard i/n")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = HPO_ROOT / args.dataset / "phase_a"
    out.mkdir(parents=True, exist_ok=True)
    i, n = (int(x) for x in args.shard.split("/"))
    mine = candidates(args.dataset, args.seed)[i::n]
    pending = [r for r in mine if not (out / f"{r[2]}.json").exists()]
    if args.dry_run:
        print(f"{args.dataset}: {len(mine)} candidates, {len(pending)} pending -> {out}")
        for _, _, rid, lr, llrd in mine:
            print(f"  {rid}: lr={lr:g} llrd={llrd:g}")
        return

    # GPUs are leased GLOBALLY (see moe_shift/utils/gpulease.py), not divided up locally.
    # --max-concurrent is only this launcher's own courtesy limit; the lease is what stops two
    # `--shard` launchers from putting four jobs on two GPUs and OOM-killing the dataloaders.
    slots = [x.strip() for x in args.gpus.split(",")]
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:                            # every GPU is busy, ours or not
                break
            cfg_path, ov, rid, _, _ = pending.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            log = open(out / f"{rid}.log", "a")
            cmd = [sys.executable, "scripts/run_ccas.py", "--config", cfg_path,
                   "--results-dir", str(out), "--override", *ov]
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, proc.pid)              # lease now tracks the job, not us
            running[gpu] = (proc, rid, log)
            print(f"[start] gpu={gpu} pid={proc.pid} {rid}", flush=True)
        for gpu in list(running):
            proc, rid, log = running[gpu]
            if proc.poll() is not None:
                log.close()
                gpulease.release(gpu, pid=proc.pid)
                print(f"[exit] {rid} rc={proc.returncode}", flush=True)
                del running[gpu]
        time.sleep(10)


if __name__ == "__main__":
    main()
