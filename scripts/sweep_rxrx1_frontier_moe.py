#!/usr/bin/env python
"""Eight-arm frontier-MoE wave: each arm targets one MEASURED failure of the completed campaigns.

Why these eight
---------------
The completed evidence is specific, so the arms are too:

==  ================================  ==================================================
#   arm                               failure it addresses
==  ================================  ==================================================
1   ``oracle_cell_type``              CEILING on content routing. Cell type is the one
                                      conditioning variable that recurs across train and
                                      test and is orthogonal to the batch nuisance.
2   ``oracle_environment``            CEILING on nuisance absorption. Unseen experiments
                                      have no expert, so this reads out shared-only.
3   ``condln_stats``                  1.18M-param experts against ~35.7 images/class;
                                      affine experts cost 2*C and match the physics of a
                                      batch effect.
4   ``lowrank_E24k8``                 expert granularity + identical-at-init deepcopies.
5   ``soft_moe_E8``                   vanishing straight-through gradient once CE
                                      saturates; aux loss then dominating routing.
6   ``lowrank_div_anneal``            balance loss balances USAGE not FUNCTION; experts
                                      start starved. Adds diversity + dense-to-sparse.
7   ``btx_specialists``               experts never differentiate. BTX makes them distinct
                                      by construction (see scripts/btx_rxrx1.py).
8   ``shared_E3k1_dro``               the only reproducible MoE effect is a mean/tail
                                      trade; GroupDRO makes the tail the objective.
==  ================================  ==================================================

Protocol notes
--------------
* ``analysis.run_mechanism=true`` on every arm. The previous wave shipped with it false and
  produced eight accuracy numbers with no route reliance, expert usage, entropy or leakage. That
  blind spot is the single most important thing not to repeat.
* Every arm keeps the pretrained dense FFN active for all inputs. Replacement is the one design
  the completed campaigns showed to be actively harmful, so it is not reintroduced as a confound;
  this is asserted, not merely intended.
* ``label_smoothing=0.1`` on every arm. Train accuracy reaches 1.0 by epoch 30 and the
  selected-ST router gradient is scaled by ``d(loss)/d(topv)``, so a saturated task loss stops
  training the router. Smoothing keeps that signal alive and is held constant across arms so it
  cannot explain any between-arm difference.
* Arm 7 is a three-phase pipeline (cluster, specialists, mix) and therefore runs a different
  entry point. It is slower than the others; it is placed last in the shard order so it does not
  hold up the single-phase arms.

Launch (one command PER CONTAINER -- see --help epilogue for why).
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

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config

CONFIG = "configs/ccas_rxrx1_cell_dino_native.yaml"
CAMPAIGN = "frontier_moe30_20260807"
DEFAULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/frontier_moe30_20260807"
)
WANDB_GROUP = "rxrx1-cell-dino-frontier-moe30-20260807"
HF_PREFIX = "rxrx1/cell_dino_cp5/frontier_moe30_20260807"

#: Blocks 10-11 are the reproducible gradient-conflict peak from
#: analysis/gradient_conflict_profile_validation.json, and the placement of the best completed
#: arm. Held FIXED across this wave so placement cannot explain any between-arm difference.
LATE2 = (10, 11)

#: Variants that keep the pretrained dense FFN active for every input.
SHARED_PATH_VARIANTS = ("shared_moe", "oracle_moe", "condln_moe", "soft_moe", "lowrank_moe")


def _blocks(indices):
    return "model.ffn_block_indices=[" + ",".join(str(index) for index in indices) + "]"


def _common(label):
    """Overrides shared by every arm. Anything here CANNOT explain a between-arm difference."""
    return [
        "seed=0", "stage=3",
        "model.routing_estimator=selected_st",
        "model.routing_unit=token", "model.geometry=cosine",
        "model.pressure=canonical", "model.balance=global",
        "model.freeze_backbone=false", "model.unfreeze_last_n_blocks=0",
        "model.sym_break_moe=0.0", "model.feature_stat_mix_prob=0.0",
        "train.objective=erm", "train.cross_experiment_pairs=false",
        "train.epochs=30", "train.milestone_epochs=[5,10,20,30]",
        "train.save_checkpoint_epochs=[30]", "train.warmup_epochs=3",
        "train.llrd=1.0", "train.batch_size=64", "train.optim.lr=1.0e-4",
        "train.optim.weight_decay=0.05", "model.drop_path=0.1",
        # Keeps cross-entropy off exactly zero so the router keeps receiving a task gradient
        # after training accuracy saturates. Constant across arms by design.
        "train.label_smoothing=0.1",
        "train.anneal_top_k_epochs=0",
        "losses.balance_w=1.0e-2", "losses.zloss_w=1.0e-3",
        "losses.cross_experiment_contrastive_w=0.0",
        "losses.cross_experiment_contrastive_temperature=0.1",
        # NOT false. The previous wave's biggest gap was shipping without routing diagnostics.
        "analysis.run_mechanism=true", "analysis.record_train_accuracy=true",
        f"run_tag={CAMPAIGN}_{label}",
    ]


#: (label, entry point, overrides). ``entry`` is "run" for the ordinary single-phase runner and
#: "btx" for the three-phase Branch-Train-MiX pipeline.
SPECS = [
    # ---- ceilings: read these first; they bound what any learned router can achieve ----
    ("oracle_cell_type", "run", [
        "model.variant=oracle_moe", "model.n_experts=4", "model.top_k=1",
        "model.group_source=cell_type", "model.expert_dropout=0.5", _blocks(LATE2),
    ]),
    ("oracle_environment", "run", [
        # 33 training experiments -> 33 routed experts. Every OOD row has site=-1, matches no
        # expert, and therefore reads out the shared path alone. That is the intended measurement.
        "model.variant=oracle_moe", "model.n_experts=33", "model.top_k=1",
        "model.group_source=environment", "model.expert_dropout=0.5", _blocks(LATE2),
    ]),
    # ---- the bet: experts small enough for this dataset's supervision budget ----
    ("condln_stats", "run", [
        "model.variant=condln_moe", "model.n_experts=8", "model.top_k=1",
        "model.condln_descriptor=token_stats", "model.condln_modulate=input", _blocks(LATE2),
    ]),
    ("lowrank_E24k8", "run", [
        "model.variant=lowrank_moe", "model.n_experts=24", "model.top_k=8",
        "model.expert_rank=16", "model.diversity_w=0.0", _blocks(LATE2),
    ]),
    # ---- estimator fixes: was the null the router, or the way it was trained? ----
    ("soft_moe_E8", "run", [
        "model.variant=soft_moe", "model.n_experts=8", "model.top_k=1",
        "model.slots_per_expert=1", "model.expert_rank=0", "model.soft_temperature=1.0",
        _blocks(LATE2),
    ]),
    ("lowrank_div_anneal", "run", [
        "model.variant=lowrank_moe", "model.n_experts=24", "model.top_k=8",
        "model.expert_rank=16", "model.diversity_w=0.05",
        "train.anneal_top_k_epochs=10", _blocks(LATE2),
    ]),
    # ---- objective: make the one reproducible effect the target ----
    ("shared_E3k1_dro", "run", [
        "model.variant=shared_moe", "model.n_experts=3", "model.top_k=1",
        "train.objective=group_dro", "train.group_dro_step_size=0.01", _blocks(LATE2),
    ]),
    # ---- experts distinct by construction; three phases, so it runs last ----
    ("btx_specialists", "btx", [
        "model.variant=shared_moe", "model.n_experts=4", "model.top_k=1", _blocks(LATE2),
    ]),
]


def wave_rows(config=CONFIG):
    """-> [(label, entry, overrides, run_id, cfg)] with run ids resolved and asserted unique."""
    rows = []
    for label, entry, intervention in SPECS:
        overrides = [*_common(label), *intervention]
        cfg = apply_overrides(load_config(config), overrides)
        variant = cfg["model"]["variant"]
        if variant not in SHARED_PATH_VARIANTS:
            raise ValueError(
                f"arm {label!r} uses variant {variant!r}, which does not keep the pretrained "
                "dense FFN active; replacement is excluded from this wave by design")
        if not bool(cfg.get("analysis", {}).get("run_mechanism")):
            raise ValueError(f"arm {label!r} must run with analysis.run_mechanism=true")
        rows.append((label, entry, overrides, run_id_from(cfg), cfg))
    run_ids = [row[3] for row in rows]
    duplicates = {rid for rid in run_ids if run_ids.count(rid) > 1}
    if duplicates:
        # Two arms sharing a run id would silently make the sweep skip the second one, and the
        # result table would show a "complete" arm that was never trained.
        raise ValueError(f"arms collide on run id(s): {sorted(duplicates)}")
    return rows


def sharded_rows(rows, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def write_manifest(out, rows):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "wave_manifest.json"
    payload = {
        "schema_version": 1, "campaign": CAMPAIGN, "config": CONFIG,
        "selection_split": "ood_val", "test_readout": "all_predefined_arms",
        "placement_fixed": list(LATE2),
        "label_smoothing_fixed": 0.1,
        "mechanism_auditing": True,
        "runs": [
            {"label": label, "entry": entry, "run_id": run_id, "overrides": overrides,
             "variant": cfg["model"]["variant"], "n_experts": cfg["model"]["n_experts"],
             "top_k": cfg["model"]["top_k"]}
            for label, entry, overrides, run_id, cfg in rows
        ],
    }
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
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
    if missing:
        if require_tracking:
            raise RuntimeError(
                "tracking is required for this wave; missing configured value(s): "
                + ", ".join(missing)
                + ". Pass --allow-untracked to proceed with local artifacts only "
                  "(the previous wave ran untracked because credentials were absent inside "
                  "the containers, so this is an explicit choice rather than a silent default).")
        # Offline W&B still produces a local run directory that can be synced later.
        env.setdefault("WANDB_MODE", "offline")
        print(f"[tracking] proceeding untracked; missing: {', '.join(missing)}", flush=True)
    env["WANDB_GROUP"] = WANDB_GROUP
    env["WANDB_JOB_TYPE"] = "rxrx1_frontier_moe30"
    env["WANDB_TAGS"] = (
        "rxrx1,cell-dino,frontier-moe,oracle-ceiling,soft-moe,lowrank,group-dro,stage3")
    env["CCAS_HF_PREFIX"] = HF_PREFIX
    return env


def command_for(row, config, out):
    """Build the subprocess command for one arm."""
    label, entry, overrides, run_id, _cfg = row
    if entry == "run":
        return [sys.executable, "scripts/run_ccas.py", "--config", config,
                "--results-dir", str(out), "--override", *overrides]
    if entry == "btx":
        return [sys.executable, "scripts/btx_rxrx1.py", "run-all", "--config", config,
                "--results-dir", str(out), "--n-clusters", "4",
                "--override", *overrides]
    raise ValueError(f"unknown entry point {entry!r} for arm {label!r}")


def render_table(out):
    """One table over every arm present on disk, including the routing diagnostics."""
    out = Path(out)
    manifest_path = out / "wave_manifest.json"
    if not manifest_path.is_file():
        return f"{CAMPAIGN} - no manifest at {out}"
    manifest = json.loads(manifest_path.read_text())
    header = ("| Arm | State | Epoch | OOD val | OOD test | Worst test | ID | Train "
              "| Reliance | Experts | Cos |")
    lines = [f"{CAMPAIGN} - {_completed(out, manifest)}/{len(manifest['runs'])} complete",
             header, "|" + "---|" * 11]
    for entry in manifest["runs"]:
        lines.append(_row(out, entry))
    return "\n".join(lines)


def _completed(out, manifest):
    return sum(1 for entry in manifest["runs"] if (out / f"{entry['run_id']}.json").is_file())


def _fmt(value, width=7, percent=True):
    if value is None:
        return "-".rjust(width)
    if percent:
        return f"{100.0 * float(value):.3f}%".rjust(width)
    return f"{float(value):.4f}".rjust(width)


def _row(out, entry):
    label, run_id = entry["label"], entry["run_id"]
    result_path = out / f"{run_id}.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        return (
            f"| {label} | complete | {result.get('config', {}).get('train', {}).get('epochs')} "
            f"| {_fmt(result.get('acc_val'))} | {_fmt(result.get('acc_heldout'))} "
            f"| {_fmt(result.get('worst_env_heldout'))} | {_fmt(result.get('acc_within'))} "
            f"| {_fmt(result.get('acc_train'))} "
            f"| {_fmt(result.get('route_reliance'), percent=False)} "
            f"| {_fmt(result.get('experts_used'), percent=False)} "
            f"| {_fmt(result.get('expert_output_cosine'), percent=False)} |")
    milestone_path = out / f"{run_id}.milestones.jsonl"
    epoch = "-"
    if milestone_path.is_file():
        rows = [json.loads(line) for line in milestone_path.read_text().splitlines() if line]
        if rows:
            epoch = rows[-1]["epoch"]
    log_path = out / f"{run_id}.log"
    state = "training" if log_path.is_file() else "pending"
    return (f"| {label} | {state} | {epoch} | " + " | ".join(["-"] * 8) + " |")


EPILOGUE = """
Launching across four 2xH100 containers
---------------------------------------
Run ONE command per container. This is not a convenience choice:

  * each SciServer container sees only its own two GPUs, both numbered 0 and 1, so a single
    launcher cannot reach another container's devices; and
  * gpulease keeps its lock files in CCAS_GPU_LOCK_DIR, default /tmp/ccas_gpu_locks, which is
    container-local -- so per-container leases are independent, which is what makes this safe.

  container 0:  python scripts/sweep_rxrx1_frontier_moe.py --shard-index 0 --num-shards 4 \\
                  --gpus 0,1 --max-concurrent 2
  container 1:  ... --shard-index 1 ...
  container 2:  ... --shard-index 2 ...
  container 3:  ... --shard-index 3 ...

Do NOT raise --max-concurrent above the number of GPUs in the container. gpulease.py documents
the incident: four jobs on two GPUs exhausted the host cgroup, the kernel OOM-killed
pt_data_worker, and five runs were lost silently -- the log stops mid-epoch with no traceback and
no result JSON, so the cells look "pending" rather than "failed". Two arms per GPU is fine as a
QUEUE (which is what this loop already does), never as co-resident processes.

Aggregation is a single command from ANY container, because the result root is shared storage:

  python scripts/aggregate_frontier_moe.py --result-root <root>
"""


def main():
    parser = argparse.ArgumentParser(
        epilog=EPILOGUE, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--result-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--allow-untracked", action="store_true",
                        help="proceed without W&B/HF credentials (local artifacts only)")
    args = parser.parse_args()

    out = Path(args.result_root).expanduser().resolve()
    all_rows = wave_rows(args.config)
    write_manifest(out, all_rows)
    if args.status:
        print(render_table(out))
        return
    rows = sharded_rows(all_rows, args.shard_index, args.num_shards)
    pending = [row for row in rows if not (out / f"{row[3]}.json").exists()]
    print(render_table(out), flush=True)
    print(f"shard {args.shard_index}/{args.num_shards}: {len(rows)} planned, "
          f"{len(pending)} pending on GPUs {args.gpus}", flush=True)
    if args.dry_run:
        for label, entry, overrides, run_id, _cfg in rows:
            print(f"  {label} [{entry}]: {run_id}")
            print(f"    {' '.join(command_for((label, entry, overrides, run_id, _cfg), args.config, out))}")
        return

    slots = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if args.max_concurrent > len(slots):
        raise ValueError(
            f"--max-concurrent {args.max_concurrent} exceeds {len(slots)} GPU(s); co-resident "
            "jobs exhausted the host cgroup and silently lost five runs (see gpulease.py)")
    base_env = tracking_environment(require_tracking=not args.allow_untracked)
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:
                break
            row = pending.pop(0)
            label, entry, _overrides, run_id, _cfg = row
            env = dict(base_env, CUDA_VISIBLE_DEVICES=gpu)
            log_handle = open(out / f"{run_id}.log", "a")
            process = subprocess.Popen(
                command_for(row, args.config, out), env=env,
                stdout=log_handle, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, run_id, label, log_handle)
            print(f"[start] gpu={gpu} pid={process.pid} {label} [{entry}] {run_id}", flush=True)

        for gpu in list(running):
            process, run_id, label, log_handle = running[gpu]
            if process.poll() is not None:
                log_handle.close()
                gpulease.release(gpu, pid=process.pid)
                print(f"[exit] gpu={gpu} rc={process.returncode} {label} {run_id}", flush=True)
                print(render_table(out), flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()
