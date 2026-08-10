#!/usr/bin/env python
"""Seed-1/2 16-experiment midpoint for the RxRx1 domain-count curve."""
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
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_rxrx1_domain_midpoint import render_report
from aggregate_rxrx1_domain_scaling_replicate import normalized_config
from moe_shift.capacity.naming import run_id_from
from moe_shift.data.rxrx1_scaling import (
    audit_environment_subset,
    full_environment_subset,
    midpoint_environment_subset,
    quarter_environment_subset,
)
from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config
from sweep_rxrx1_domain_scaling_replicate import (
    ARMS,
    CONFIG,
    FULL_ANCHOR_ROOT,
    SEEDS,
    _common,
    _environment_override,
    full_anchor_rows,
)


CAMPAIGN = "rxrx1_domain_midpoint30_20260810"
DEFAULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/rxrx1_domain_midpoint30_20260810"
)
QUARTER_ANCHOR_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/rxrx1_domain_scaling_replicate30_20260810"
)
WANDB_GROUP = "rxrx1-cell-dino-domain-midpoint30-20260810"
HF_PREFIX = "rxrx1/cell_dino_cp5/rxrx1_domain_midpoint30_20260810"


def _source_identity():
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True).strip())
    return sha, dirty


def wave_rows(config=CONFIG):
    environments = midpoint_environment_subset()
    rows = []
    for seed in SEEDS:
        for arm, intervention in ARMS:
            label = f"midpoint_{arm}_s{seed}"
            common = _common(seed, arm)
            common[-1] = f"run_tag={CAMPAIGN}_{label}"
            overrides = [*common, *intervention, _environment_override(environments)]
            cfg = apply_overrides(load_config(config), overrides)
            rows.append((label, arm, seed, environments, overrides, run_id_from(cfg), cfg))
    if len({row[5] for row in rows}) != len(rows):
        raise ValueError("domain-midpoint run identities collide")
    validate_planned_pairs(rows, config)
    return rows


def validate_planned_pairs(rows, config=CONFIG):
    anchors = {(row[1], int(row[2])): row[5] for row in full_anchor_rows(config)}
    for _label, arm, seed, _environments, _overrides, _run_id, cfg in rows:
        anchor = anchors.get((arm, int(seed)))
        if anchor is None:
            raise ValueError(f"missing completed full anchor declaration for {arm}/seed{seed}")
        if normalized_config(cfg) != normalized_config(anchor):
            raise ValueError(f"planned midpoint/full config drift for {arm}/seed{seed}")
    return True


def sharded_rows(rows, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def _dataset_audit(config):
    try:
        from wilds import get_dataset
        cfg = load_config(config)
        dataset = get_dataset(dataset="rxrx1", root_dir=cfg["data_root"], download=False)
    except Exception as error:
        return {"unavailable": f"{type(error).__name__}: {error}"}
    split_indices = {
        split: set(map(int, dataset.get_subset(split).indices))
        for split in ("train", "id_test", "val", "test")
    }
    overlaps = {}
    names = tuple(split_indices)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            count = len(split_indices[left] & split_indices[right])
            overlaps[f"{left}__{right}"] = count
            if count:
                raise ValueError(f"RxRx1 split-index leakage: {left}/{right} share {count}")
    audits = {
        "quarter": audit_environment_subset(dataset, quarter_environment_subset()),
        "midpoint": audit_environment_subset(dataset, midpoint_environment_subset()),
        "full": audit_environment_subset(dataset, full_environment_subset()),
    }
    if not (set(audits["quarter"]["environment_ids"]) <
            set(audits["midpoint"]["environment_ids"]) <
            set(audits["full"]["environment_ids"])):
        raise ValueError("RxRx1 environment subsets are not strictly nested")
    if {value["n_classes_observed"] for value in audits.values()} != {
            audits["full"]["n_classes_expected"]}:
        raise ValueError("domain-count curve loses perturbation classes")
    if any(set(map(int, value["cell_environment_counts"])) != {0, 1, 2, 3}
           for value in audits.values()):
        raise ValueError("domain-count curve loses cell types")
    return {"scales": audits,
            "split_index_counts": {key: len(value) for key, value in split_indices.items()},
            "split_index_overlaps": overlaps}


def write_manifest(out, rows, config=CONFIG):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    anchors = {(row[1], int(row[2])): row[4] for row in full_anchor_rows(config)}
    source_sha, source_dirty = _source_identity()
    payload = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "config": config,
        "expected_runs": 8,
        "seeds": list(SEEDS),
        "axis": "number_of_independent_training_experiments",
        "atomic_sampling_unit": "RxRx1 field, nested by experiment",
        "quarter_environment_subset": list(quarter_environment_subset()),
        "midpoint_environment_subset": list(midpoint_environment_subset()),
        "full_environment_subset": list(full_environment_subset()),
        "quarter_anchor_root": str(QUARTER_ANCHOR_ROOT),
        "full_anchor_root": str(FULL_ANCHOR_ROOT),
        "checkpoint_rule": "terminal epoch 30; all predeclared arms receive stage-3 readout",
        "headline_endpoints": ["acc_heldout", "worst_env_heldout", "active_ffn_params"],
        "primary_contrast": "slope of shared-minus-dense OOD test gap over log2(train experiments)",
        "stopping_rule": "exactly 30 epochs for every new arm; no adaptive extension",
        "source_git_commit": source_sha,
        "source_git_dirty": source_dirty,
        "compute_accounting": {
            "dense_E4_late2": {"total_params": 29493881,
                                "active_ffn_params": 9454854,
                                "estimated_active_ffn_flops_relative": 2.0},
            "replace_E4k2_late2": {"total_params": 29494645,
                                    "active_ffn_params": 4729346,
                                    "estimated_active_ffn_flops_relative": 1.0},
            "shared_E3k1_late2": {"total_params": 29493877,
                                   "active_ffn_params": 4728578,
                                   "estimated_active_ffn_flops_relative": 1.0},
            "note": "Relative active FFN FLOPs use common token/sequence shapes; exact end-to-end FLOPs are not claimed. Terminal artifacts re-audit parameter counts.",
        },
        "dataset_audit": _dataset_audit(config),
        "runs": [
            {"label": label, "scale": "midpoint", "arm": arm, "seed": seed,
             "run_id": run_id, "environment_subset": list(environments),
             "overrides": overrides, "variant": cfg["model"]["variant"],
             "resolved_config": cfg,
             "full_anchor_run_id": anchors[(arm, int(seed))]}
            for label, arm, seed, environments, overrides, run_id, cfg in rows
        ],
    }
    path = out / "wave_manifest.json"
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)
    return payload


def tracking_environment(require_tracking=True):
    env = dict(os.environ)
    if not env.get("HF_TOKEN"):
        try:
            from huggingface_hub import get_token
            env["HF_TOKEN"] = get_token() or ""
        except Exception:
            pass
    if not env.get("WANDB_API_KEY"):
        try:
            import wandb
            env["WANDB_API_KEY"] = wandb.api.api_key or ""
        except Exception:
            pass
    missing = [name for name in ("WANDB_API_KEY", "HF_TOKEN", "CCAS_HF_REPO")
               if not env.get(name)]
    if missing and require_tracking:
        raise RuntimeError("tracking unavailable (" + ", ".join(missing)
                           + "); pass --allow-untracked for local-first execution")
    if missing:
        env["WANDB_MODE"] = "offline"
        print(f"[tracking] local-first mode; missing: {', '.join(missing)}", flush=True)
    env["WANDB_GROUP"] = WANDB_GROUP
    env["WANDB_JOB_TYPE"] = "rxrx1_domain_midpoint30"
    env["WANDB_TAGS"] = "rxrx1,cell-dino,domain-scaling,midpoint,matched-controls,stage3"
    env["CCAS_HF_PREFIX"] = HF_PREFIX
    return env


def command_for(row, config, out):
    return [sys.executable, "scripts/run_ccas.py", "--config", config,
            "--results-dir", str(out), "--override", *row[4]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--result-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--quarter-root", default=str(QUARTER_ANCHOR_ROOT))
    parser.add_argument("--full-root", default=str(FULL_ANCHOR_ROOT))
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--allow-untracked", action="store_true")
    args = parser.parse_args()

    out = Path(args.result_root).expanduser().resolve()
    rows = wave_rows(args.config)
    write_manifest(out, rows, args.config)
    if args.status:
        print(render_report(out, args.quarter_root, args.full_root))
        return
    selected = sharded_rows(rows, args.shard_index, args.num_shards)
    pending = [row for row in selected if not (out / f"{row[5]}.json").exists()]
    print(render_report(out, args.quarter_root, args.full_root), flush=True)
    print(f"shard {args.shard_index}/{args.num_shards}: {len(selected)} planned, "
          f"{len(pending)} pending on GPUs {args.gpus}", flush=True)
    if args.dry_run:
        for row in selected:
            print(f"  {row[0]}: {row[5]}")
        return

    slots = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if args.max_concurrent > len(slots):
        raise ValueError("max-concurrent cannot exceed visible GPU slots")
    base_env = tracking_environment(require_tracking=not args.allow_untracked)
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:
                break
            row = pending.pop(0)
            env = dict(base_env, CUDA_VISIBLE_DEVICES=gpu)
            log_handle = open(out / f"{row[5]}.log", "a")
            process = subprocess.Popen(command_for(row, args.config, out), env=env,
                                       stdout=log_handle, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, row, log_handle)
            print(f"[start] gpu={gpu} pid={process.pid} {row[0]} {row[5]}", flush=True)
        for gpu in list(running):
            process, row, log_handle = running[gpu]
            if process.poll() is not None:
                log_handle.close()
                gpulease.release(gpu, pid=process.pid)
                print(f"[exit] gpu={gpu} rc={process.returncode} {row[0]} {row[5]}", flush=True)
                print(render_report(out, args.quarter_root, args.full_root), flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()
