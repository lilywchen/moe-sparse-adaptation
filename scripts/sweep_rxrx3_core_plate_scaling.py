#!/usr/bin/env python3
"""Two-seed matched RxRx3-core plate-count scaling wave.

Each invocation launches one predeclared plate point (1, 2, or 4 plates).  The
completed 8-plate competence pilot is the frozen full-data anchor.
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
sys.path.insert(0, str(ROOT / "scripts"))

import sweep_rxrx3_core_pilot as pilot
from aggregate_rxrx3_core_pilot import render_report
from moe_shift.capacity.naming import run_id_from
from moe_shift.data.rxrx3_core import read_rxrx3_manifest
from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config


CONFIG = pilot.CONFIG
MANIFEST_DIR = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/datasets/rxrx3-core/manifests"
)
RESULT_PARENT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx3_core/cell_dino_cp5"
)
FULL_ANCHOR_ROOT = RESULT_PARENT / "rxrx3_core_pilot10_20260810"
EXPECTED_TRAIN_ROWS = {1: 2696, 2: 5376, 4: 10706, 8: 21404}
PLATE_COUNTS = tuple(EXPECTED_TRAIN_ROWS)
ALLOWED_SCALE_DIFFERENCES = {
    *pilot.ALLOWED_ARCHITECTURE_DIFFERENCES,
    "rxrx3_manifest",
}


def campaign(plates):
    return f"rxrx3_core_plate{int(plates)}_scale10_20260810"


def result_root(plates):
    return RESULT_PARENT / campaign(plates)


def manifest_path(plates):
    plates = int(plates)
    if plates not in PLATE_COUNTS:
        raise ValueError(f"plates must be one of {PLATE_COUNTS}")
    return MANIFEST_DIR / f"rxrx3_core_gene_plate_{plates}.tsv"


def wave_rows(plates, config=CONFIG):
    plates = int(plates)
    rows = []
    tag = campaign(plates)
    for seed in pilot.SEEDS:
        for arm, intervention in pilot.ARMS:
            label = f"{arm}_s{seed}"
            overrides = [
                *pilot._common(seed, label),
                f"rxrx3_manifest={manifest_path(plates)}",
                f"run_tag={tag}_{label}",
                *intervention,
            ]
            cfg = apply_overrides(load_config(config), overrides)
            rows.append((label, arm, seed, overrides, run_id_from(cfg), cfg))
    if len(rows) != 8 or len({row[4] for row in rows}) != 8:
        raise ValueError("each RxRx3 plate point requires eight collision-free run ids")
    pilot.validate_resolved_configs(rows)
    return rows


def _split_sets(records):
    return {
        split: {row["well_id"] for row in records if row["split"] == split}
        for split in ("train", "id_val", "ood_test")
    }


def dataset_audit(plates):
    """Audit the complete nested plate curve without opening image bytes."""
    plates = int(plates)
    summaries = {}
    sets = {}
    for count in PLATE_COUNTS:
        records, summary = read_rxrx3_manifest(manifest_path(count))
        split_sets = _split_sets(records)
        if len(records) != len({row["well_id"] for row in records}):
            raise ValueError(f"plate {count}: duplicate manifest wells")
        if any(split_sets[a] & split_sets[b] for a, b in (
                ("train", "id_val"), ("train", "ood_test"), ("id_val", "ood_test"))):
            raise ValueError(f"plate {count}: split leakage")
        expected = {
            "train": EXPECTED_TRAIN_ROWS[count], "id_val": 2708, "ood_test": 23855,
        }
        if summary["split_counts"] != expected:
            raise ValueError(
                f"plate {count}: split counts {summary['split_counts']} != {expected}"
            )
        if (summary["classes"] != 674 or summary["train_experiments"] != 85
                or summary["ood_test_experiments"] != 85):
            raise ValueError(f"plate {count}: frozen task coverage drift")
        summaries[count] = summary
        sets[count] = split_sets
    for smaller, larger in zip(PLATE_COUNTS, PLATE_COUNTS[1:]):
        if not sets[smaller]["train"] < sets[larger]["train"]:
            raise ValueError(f"train wells are not strictly nested: {smaller} -> {larger}")
    for count in PLATE_COUNTS[1:]:
        if (sets[count]["id_val"] != sets[1]["id_val"]
                or sets[count]["ood_test"] != sets[1]["ood_test"]):
            raise ValueError(f"plate {count}: evaluation wells drift")

    pixel_audit = json.loads(
        (MANIFEST_DIR.parent / "rxrx3_core_image_audit.json").read_text()
    )
    if (pixel_audit.get("passed") is not True
            or pixel_audit.get("channel_rows") != 1335606
            or pixel_audit.get("manifest_union_wells") != 47967):
        raise ValueError("frozen six-channel pixel gate is not closed")
    selected = summaries[plates]
    return {
        "passed": True,
        "axis": "train_plate_count_with_four_guides_fixed",
        "selected_plates": plates,
        "selected_manifest": selected["manifest"],
        "selected_manifest_sha256": selected["manifest_sha256"],
        "selected_well_set_sha256": selected["well_set_sha256"],
        "split_counts": selected["split_counts"],
        "classes": selected["classes"],
        "train_experiments": selected["train_experiments"],
        "ood_test_experiments": selected["ood_test_experiments"],
        "cell_types": selected["cell_types"],
        "curve_manifest_sha256": {
            str(count): summaries[count]["manifest_sha256"] for count in PLATE_COUNTS
        },
        "curve_well_set_sha256": {
            str(count): summaries[count]["well_set_sha256"] for count in PLATE_COUNTS
        },
        "evaluation_wells_fixed": True,
        "train_wells_strictly_nested": True,
        "six_channel_pixel_gate": True,
    }


def _source_identity():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, text=True,
    ).strip())
    return sha, dirty


def write_manifest(out, rows, plates, audit=None, capacity=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sha, dirty = _source_identity()
    if dirty:
        raise ValueError("RxRx3 plate scaling requires a clean tracked worktree")
    audit = dataset_audit(plates) if audit is None else audit
    capacity = pilot.capacity_accounting(rows) if capacity is None else capacity
    payload = {
        "schema_version": 1,
        "campaign": campaign(plates),
        "config": CONFIG,
        "expected_runs": 8,
        "seeds": list(pilot.SEEDS),
        "axis": "train_plate_count_with_four_guides_fixed",
        "plate_count": int(plates),
        "guide_count_fixed": 4,
        "class_count_fixed": 674,
        "train_experiments_fixed": 85,
        "atomic_unit": "well with exactly six joined stain rows",
        "full_anchor_root": str(FULL_ANCHOR_ROOT),
        "headline_endpoints": ["acc_heldout", "worst_env_heldout", "active_ffn_params"],
        "primary_contrast": "shared-residual E3/top-1 versus dense E4 at matched total capacity",
        "secondary_contrasts": ["shared versus replacement", "dense versus original"],
        "stopping_rule": "exactly 10 epochs; terminal checkpoint; no topology adaptation",
        "checkpoint_rule": "fixed ID-validation plates; terminal epoch 10",
        "allowed_config_differences": sorted(ALLOWED_SCALE_DIFFERENCES | {"seed"}),
        "source_git_commit": sha,
        "source_git_dirty": dirty,
        "dataset_audit": audit,
        "compute_accounting": capacity,
        "runs": [
            {
                "label": label, "arm": arm, "seed": seed, "overrides": overrides,
                "run_id": run_id, "variant": cfg["model"]["variant"],
                "resolved_config": cfg,
            }
            for label, arm, seed, overrides, run_id, cfg in rows
        ],
    }
    path = out / "wave_manifest.json"
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)
    return payload


def tracking_environment(plates, require_tracking=True):
    env = pilot.tracking_environment(require_tracking=require_tracking)
    env["WANDB_GROUP"] = f"rxrx3-core-cell-dino-plate{int(plates)}-scale10-20260810"
    env["WANDB_JOB_TYPE"] = f"rxrx3_core_plate{int(plates)}_scale10"
    env["WANDB_TAGS"] = "rxrx3-core,cell-dino,plate-scaling,matched-controls,stage3"
    env["CCAS_HF_PREFIX"] = f"rxrx3_core/cell_dino_cp5/plate{int(plates)}_scale10_20260810"
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plates", type=int, choices=PLATE_COUNTS, required=True)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--result-root")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--allow-untracked", action="store_true")
    args = parser.parse_args()

    out = Path(args.result_root or result_root(args.plates)).expanduser().resolve()
    rows = wave_rows(args.plates, args.config)
    write_manifest(out, rows, args.plates)
    if args.status:
        print(render_report(out))
        return
    selected = pilot.sharded_rows(rows, args.shard_index, args.num_shards)
    pending = [row for row in selected if not (out / f"{row[4]}.json").exists()]
    print(render_report(out), flush=True)
    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(selected)} planned, "
        f"{len(pending)} pending on GPUs {args.gpus}", flush=True,
    )
    if args.dry_run:
        for row in selected:
            print(f"  {row[0]}: {row[4]}")
        return

    slots = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if args.max_concurrent > len(slots):
        raise ValueError("max-concurrent cannot exceed visible GPU slots")
    base_env = tracking_environment(
        args.plates, require_tracking=not args.allow_untracked
    )
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:
                break
            row = pending.pop(0)
            env = dict(base_env, CUDA_VISIBLE_DEVICES=gpu)
            log_handle = open(out / f"{row[4]}.log", "a")
            process = subprocess.Popen(
                pilot.command_for(row, args.config, out), env=env,
                stdout=log_handle, stderr=subprocess.STDOUT,
            )
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, row, log_handle)
            print(f"[start] gpu={gpu} pid={process.pid} {row[0]} {row[4]}", flush=True)
        for gpu in list(running):
            process, row, log_handle = running[gpu]
            if process.poll() is not None:
                log_handle.close()
                gpulease.release(gpu, pid=process.pid)
                print(
                    f"[exit] gpu={gpu} rc={process.returncode} {row[0]} {row[4]}",
                    flush=True,
                )
                print(render_report(out), flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()
