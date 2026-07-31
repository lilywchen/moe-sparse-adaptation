#!/usr/bin/env python
"""CCAS Stage 1: the 36-cell MoE grid + matched dense-wide + original-dense controls.

    3 placements x 2 routing units x 2 router geometries x 3 training pressures = 36 MoE cells
  + 6 dense-wide controls (placement x canonical/output; route has no dense analogue)
  + 1 original-dense reference
  = 43 cells per dataset per seed.

Stage 1 budget: 216 MoE runs, 36 dense-wide, 6 original = 258 runs across both datasets.

Idempotent: a cell whose result JSON exists is skipped, so relaunching is always safe.

    python scripts/sweep_ccas.py --dry-run
    python scripts/sweep_ccas.py --gpus 0,1 --max-concurrent 2
"""
import argparse, itertools, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe_shift.capacity.naming import run_id_from

from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config

PLACEMENTS = ["early", "middle", "late"]
ROUTING_UNITS = ["image", "token"]
GEOMETRIES = ["linear", "cosine"]
PRESSURES = ["canonical", "route", "output"]

DATASETS = {"rxrx1": "configs/ccas_rxrx1.yaml", "camelyon17": "configs/ccas_camelyon17.yaml"}
SEEDS = [0, 1, 2]                       # Stage 1: three paired seeds
RESULTS = Path(os.environ.get("MOE_RESULTS", "./RESULTS/ccas"))


def cells(datasets=None, seeds=None):
    """(label, cfg_path, overrides, run_id) for every Stage-1 cell."""
    out = []
    for ds in (datasets or DATASETS):
        cfg_path = DATASETS[ds]
        for seed in (seeds or SEEDS):
            # --- 36 MoE cells ---
            for pl, ru, ge, pressure in itertools.product(
                    PLACEMENTS, ROUTING_UNITS, GEOMETRIES, PRESSURES):
                balance = "within_environment" if pressure == "route" else "global"
                ov = [f"seed={seed}", "model.variant=moe", f"model.placement={pl}",
                      f"model.routing_unit={ru}", f"model.geometry={ge}",
                      f"model.pressure={pressure}", f"model.balance={balance}"]
                out.append((f"moe_{pl}_{ru}_{ge}_{pressure}", cfg_path, ov, ds, seed))
            # --- depth-matched dense-wide controls ---
            # canonical matches canonical and route-level MoE; output matches DANN-for-DANN.
            for pl, pressure in itertools.product(PLACEMENTS, ["canonical", "output"]):
                ov = [f"seed={seed}", "model.variant=dense_wide", f"model.placement={pl}",
                      f"model.pressure={pressure}", "model.balance=global"]
                out.append((f"dense_wide_{pl}_{pressure}", cfg_path, ov, ds, seed))
            # --- original dense reference (lower budget P0) ---
            out.append(("original", cfg_path, [f"seed={seed}", "model.variant=original"], ds, seed))
    resolved = []
    for lab, cfg_path, ov, ds, seed in out:
        rid = run_id_from(apply_overrides(load_config(cfg_path), ov))
        resolved.append((lab, cfg_path, ov, rid, ds, seed))
    return resolved


def pending(all_cells):
    return [c for c in all_cells if not (RESULTS / f"{c[3]}.json").exists()]


def launch(cfg_path, ov, rid, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MOE_RESULTS=str(RESULTS))
    cmd = [sys.executable, "scripts/run_ccas.py", "--config", cfg_path, "--override", *ov]
    log = open(RESULTS / f"{rid}.log", "a")
    log.write(f"\n=== launch {time.strftime('%Y-%m-%d %H:%M:%S')} gpu={gpu} ===\n"); log.flush()
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    ap.add_argument("--max-concurrent", type=int, default=2, help="2 x H100 container -> 2")
    ap.add_argument("--dataset", default=None, choices=list(DATASETS))
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 0 or 0,1,2")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                    help="stage gate: below 3 the OOD test split is never evaluated (PLAN.md)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    ds = [args.dataset] if args.dataset else None
    sd = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    allc = cells(ds, sd)
    if args.stage != 1:
        # Stage 3 runs are a DIFFERENT experiment from the Stage-1 cell of the same name (they
        # reveal the test split), so they carry their own run_tag and never overwrite Stage-1 JSONs.
        allc = [(lab, cfg, ov + [f"stage={args.stage}", f"run_tag=stage{args.stage}"],
                 f"{rid}_stage{args.stage}", dset, seed)
                for lab, cfg, ov, rid, dset, seed in allc]
    i, n = (int(x) for x in args.shard.split("/"))
    mine = allc[i::n]
    todo = pending(mine)

    if args.dry_run:
        print(f"CCAS Stage {args.stage}: {len(mine)} cells, "
              f"{len(mine)-len(todo)} done, {len(todo)} pending")
        print(f"results: {RESULTS}")
        by_ds = {}
        for lab, _, _, rid, dset, seed in mine:
            done = (RESULTS / f"{rid}.json").exists()
            by_ds.setdefault(dset, [0, 0])[0 if done else 1] += 1
        for dset, (d, p) in by_ds.items():
            print(f"  {dset}: {d} done / {p} pending")
        for lab, _, _, rid, dset, seed in todo[:40]:
            print(f"    [pending] {rid}")
        if len(todo) > 40:
            print(f"    ... and {len(todo)-40} more")
        return

    # One lease per physical GPU, held across ALL launcher processes (see gpulease.py). Stage 1 is
    # 168 cells and is normally sharded; without the lease each shard enforced --max-concurrent
    # against its own copy of --gpus, so N shards ran 2N jobs and the host cgroup OOM-killed the
    # dataloader workers. --max-concurrent survives as this launcher's own ceiling.
    slots = [g.strip() for g in args.gpus.split(",")]
    print(f"CCAS: {len(todo)} pending, {len(slots)} GPUs, "
          f"max {args.max_concurrent} here -> {RESULTS}", flush=True)
    running = {}
    while todo or running:
        while todo and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:
                break
            lab, cfg_path, ov, rid, dset, seed = todo.pop(0)
            p = launch(cfg_path, ov, rid, gpu)
            gpulease.adopt(gpu, p.pid)
            running[gpu] = (p, rid)
            print(f"[start] gpu{gpu} pid={p.pid} {rid}", flush=True)
        for gpu in list(running):
            p, rid = running[gpu]
            if p.poll() is not None:
                gpulease.release(gpu, pid=p.pid)
                print(f"[exit ] gpu{gpu} {rid} rc={p.returncode}", flush=True)
                del running[gpu]
        time.sleep(10)
    print("CCAS sweep complete")
    subprocess.run([sys.executable, "scripts/aggregate_ccas.py"], env=dict(os.environ, MOE_RESULTS=str(RESULTS)))


if __name__ == "__main__":
    main()
