#!/usr/bin/env python
"""Twelve-hour, four-container RxRx1 batch-correction campaign.

Run the same one-liner in each 2xH100 container with a different ``--shard-index``.  The campaign
is idempotent, writes only persistent storage, caps concurrency at one process per GPU, and shard
zero can wait for all shards, run the test-blind geometry audit, and aggregate the final report.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config

CONFIG = "configs/ccas_rxrx1_cell_dino_native.yaml"
CAMPAIGN = "batch_corrector_12h_20260819"
DEFAULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/rxrx1_batch_corrector_12h_20260819")


DISCOVERY = [
    ("original_random", "none", 4, 16, False, False, 0.0),
    ("original_grouped", "none", 4, 16, True, False, 0.0),
    ("PairedERM", "none", 4, 16, True, True, 0.0),
    ("H1_center", "center", 4, 16, True, False, 0.0),
    ("H2_adabn", "adabn", 4, 16, True, False, 0.0),
    ("H3_lowrank_r16", "lowrank", 1, 16, True, False, 0.0),
    ("H4_batch_E4r16", "moe_batch", 4, 16, True, False, 0.0),
    ("HarmonyDG_w010", "none", 4, 16, True, True, 0.10),
    ("HarmonyDG_w020", "none", 4, 16, True, True, 0.20),
    ("HarmonyAdaBN_w010", "adabn", 4, 16, True, True, 0.10),
    ("TransportMoE_w010", "moe_batch", 4, 16, True, True, 0.10),
    ("TransportMoE_w020", "moe_batch", 4, 16, True, True, 0.20),
]

# Predeclared before discovery numbers are seen.  This prevents four OOD-validation experiments
# from becoming a hyperparameter oracle and ensures every rung of H0/H2/H3/H4 gets replication.
CORE = [
    ("original_grouped", "none", 4, 16, True, False, 0.0),
    ("H2_adabn", "adabn", 4, 16, True, False, 0.0),
    ("HarmonyDG", "none", 4, 16, True, True, 0.10),
    ("TransportMoE", "moe_batch", 4, 16, True, True, 0.10),
]


def _common(label, phase, seed, epochs, stage, grouped, paired, consistency_w, context_sizes):
    return [
        f"seed={seed}", f"stage={stage}", "model.variant=original",
        "model.pressure=canonical", "model.freeze_backbone=false",
        "train.objective=erm",
        f"train.cross_experiment_pairs={'true' if paired else 'false'}",
        f"train.paired_experiment_batches={'true' if paired else 'false'}",
        f"train.experiment_batching={'true' if grouped else 'false'}",
        f"train.epochs={epochs}", f"train.milestone_epochs=[{epochs}]",
        f"train.save_checkpoint_epochs={'[' + str(epochs) + ']' if stage >= 3 else '[]'}",
        f"train.warmup_epochs={max(1, round(epochs * 0.05))}",
        "train.batch_size=64", "train.num_workers=8", "train.llrd=1.0",
        "train.optim.lr=1.0e-4", "train.optim.weight_decay=0.05",
        "train.label_smoothing=0.1", "model.drop_path=0.1",
        "losses.balance_w=1.0e-2", "losses.zloss_w=1.0e-3",
        f"losses.cross_experiment_contrastive_w={consistency_w}",
        "analysis.record_train_accuracy=false", "analysis.run_mechanism=false",
        f"analysis.context_sizes={context_sizes}",
        "analysis.context_repeats=8",
        f"run_tag={CAMPAIGN}_{phase}_{label}",
    ]


def _row(label, mode, experts, rank, grouped, paired, consistency_w,
         phase, seed, epochs, stage):
    context_sizes = "[8,16,32,64]" if stage >= 3 and mode != "none" else "[]"
    overrides = _common(label, phase, seed, epochs, stage, grouped, paired,
                        consistency_w, context_sizes) + [
        f"model.batch_corrector.mode={mode}",
        f"model.batch_corrector.n_experts={experts}",
        f"model.batch_corrector.rank={rank}",
        "model.batch_corrector.hidden=128", "model.batch_corrector.temperature=1.0",
    ]
    cfg = apply_overrides(load_config(CONFIG), overrides)
    return {"label": label, "phase": phase, "seed": seed, "epochs": epochs,
            "stage": stage, "mode": mode, "n_experts": experts, "rank": rank,
            "grouped": grouped, "composition_matched_pairs": paired,
            "contrastive_w": consistency_w,
            "overrides": overrides, "run_id": run_id_from(cfg)}


def campaign_rows():
    rows = [_row(*spec, "discovery", 0, 12, 1) for spec in DISCOVERY]
    for seed in (1, 2, 3):
        rows.extend(_row(*spec, "replication", seed, 30, 1) for spec in CORE)
    for seed in (4, 5, 6):
        rows.extend(_row(*spec, "confirmatory", seed, 100, 3) for spec in CORE)
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("campaign contains colliding run ids")
    return rows


def write_manifest(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "campaign": CAMPAIGN,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": CONFIG, "wall_time_target_hours": 12, "gpus": 8,
        "selection_policy": "discovery and replication use OOD val only; all four predeclared "
                            "core methods receive fresh-seed stage-3 test readout",
        "target_context_policy": "correction uses no labels and at most 64 images from the "
                                 "current experiment; final models are audited at 8/16/32/64",
        "ambition_ladder": {
            "L0": "ERM; no target context",
            "L1": "HarmonyDG; learns invariance from composition-matched source pairs, no target context",
            "L2": "AdaBN; unlabelled target moments",
            "L3": "TransportMoE; learned shared operators plus equal unlabelled target moments",
            "L4_ceiling": "label-matched target correction analysis only; never deployable",
        },
        "paper_reference": {"baseline_batch_separated": 0.751, "adabn_batch_separated": 0.871,
                            "paper_epochs": 100, "paper_resolution": 512,
                            "paper_backbone": "DenseNet-161"},
        "estimated_gpu_hours": 85.2,
        "runs": rows,
    }
    path = root / "campaign_manifest.json"
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    os.replace(temporary, path)
    return manifest


def status(root, manifest):
    completed = failed = 0
    by_phase = {}
    for row in manifest["runs"]:
        state = "complete" if (root / f"{row['run_id']}.json").is_file() else (
            "failed" if (root / f"{row['run_id']}.failure.json").is_file() else "pending")
        completed += state == "complete"; failed += state == "failed"
        by_phase.setdefault(row["phase"], {"complete": 0, "failed": 0, "total": 0})
        by_phase[row["phase"]]["total"] += 1
        by_phase[row["phase"]][state] = by_phase[row["phase"]].get(state, 0) + 1
    return {"complete": completed, "failed": failed, "total": len(manifest["runs"]),
            "by_phase": by_phase}


def command(row, root):
    return [sys.executable, "scripts/run_ccas.py", "--config", CONFIG,
            "--results-dir", str(root), "--override", *row["overrides"]]


def run_shard(args, root, manifest):
    rows = [row for index, row in enumerate(manifest["runs"])
            if index % args.num_shards == args.shard_index]
    pending = [row for row in rows if not (root / f"{row['run_id']}.json").is_file()]
    # Longest-first makes three confirmatory jobs balance cleanly over two GPUs per container;
    # every arm is predeclared, so execution order cannot leak results into the design.
    pending.sort(key=lambda row: row["epochs"], reverse=True)
    slots = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if args.max_concurrent > len(slots):
        raise ValueError("max concurrency cannot exceed visible GPU count")
    env_base = dict(os.environ)
    env_base.setdefault("WANDB_MODE", "offline")
    env_base["WANDB_GROUP"] = CAMPAIGN
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:
                break
            row = pending.pop(0)
            log = open(root / f"{row['run_id']}.log", "a")
            process = subprocess.Popen(
                command(row, root), cwd=ROOT, env=dict(env_base, CUDA_VISIBLE_DEVICES=gpu),
                stdout=log, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, row, log)
            print(f"[start] gpu={gpu} {row['phase']} {row['label']} seed={row['seed']}", flush=True)
        for gpu in list(running):
            process, row, log = running[gpu]
            if process.poll() is None:
                continue
            log.close(); gpulease.release(gpu, pid=process.pid)
            if process.returncode != 0 and not (root / f"{row['run_id']}.json").is_file():
                (root / f"{row['run_id']}.failure.json").write_text(json.dumps({
                    "run_id": row["run_id"], "returncode": process.returncode,
                    "log": f"{row['run_id']}.log", "time": time.time()}, indent=2))
            print(f"[exit] gpu={gpu} rc={process.returncode} {row['label']}", flush=True)
            del running[gpu]
        if pending or running:
            time.sleep(10)


def wait_for_global(root, manifest, timeout_hours):
    deadline = time.time() + 3600 * timeout_hours
    while time.time() < deadline:
        current = status(root, manifest)
        print(f"[global] {current['complete']}/{current['total']} complete, "
              f"{current['failed']} failed", flush=True)
        if current["complete"] + current["failed"] == current["total"]:
            return current
        time.sleep(60)
    return status(root, manifest)


def finish_campaign(root, manifest, gpu):
    audit = root / "hypothesis_audit"
    cache = root / "cache" / "cell_dino_train_id_val_features.npz"
    if not (audit / "batch_hypotheses.json").is_file():
        subprocess.run([
            sys.executable, "scripts/analyze_rxrx1_batch_hypotheses.py", "--extract",
            "--config", CONFIG, "--cache", str(cache), "--output", str(audit),
        ], cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), check=False,
           stdout=open(root / "hypothesis_audit.log", "a"), stderr=subprocess.STDOUT)
    subprocess.run([
        sys.executable, "scripts/aggregate_rxrx1_batch_correctors.py",
        "--result-root", str(root)], cwd=ROOT, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-global", action="store_true")
    parser.add_argument("--global-timeout-hours", type=float, default=14.0)
    args = parser.parse_args()
    root = Path(args.result_root).expanduser().resolve()
    manifest = write_manifest(root, campaign_rows())
    if args.status:
        print(json.dumps(status(root, manifest), indent=2)); return
    if args.dry_run:
        for index, row in enumerate(manifest["runs"]):
            if index % args.num_shards == args.shard_index:
                print(row["phase"], row["label"], row["seed"], row["run_id"])
        return
    run_shard(args, root, manifest)
    if args.wait_for_global:
        wait_for_global(root, manifest, args.global_timeout_hours)
        finish_campaign(root, manifest, args.gpus.split(",")[0])


if __name__ == "__main__":
    main()
