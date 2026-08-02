#!/usr/bin/env python
"""Locked fresh-seed confirmation of the route-balanced E16 versus E8 signal.

The seed-0 low-temperature expert-count screen found a small, directionally
consistent mean-OOD and worst-experiment gain for route-balanced E16 over its
exact route-balanced E8 anchor at epochs 30 and 60 without an ID penalty.  This
registry freezes that comparison at seeds 1 and 2.  It is not a new tuning
screen: both expert counts use route pressure, temperature 0.03, zero load-
balance loss, router z-loss 0.001, and the same optimizer, data order,
60-epoch horizon, and 10/30/60 checkpoints.

The estimand is active-compute matched; total parameters intentionally differ.
Selection remains OOD validation only, OOD test is sealed, and no configuration
change is permitted after these confirmation jobs begin.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_router_aux as refill
from scripts.sweep_rxrx1_cell_dino import CONFIG


RESULT_ROOT = Path(os.environ.get(
    "MOE_RX_TEMPERATURE_EXPERT_CONFIRM_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/temperature_expert_count_confirm60_20260802",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-temperature-expert-count-confirm60-20260802"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/temperature_expert_count_confirm60_20260802"
)
WANDB_JOB_TYPE = "rxrx1_temperature_expert_count_confirm60"
WANDB_TAGS = (
    "rxrx1,cell-dino,temperature-expert-count-confirm60,confirmatory,"
    "active-compute-matched,ood-test-blind"
)

SEEDS = (1, 2)
EXPERT_COUNTS = (8, 16)
TEMPERATURE = 3.0e-2
BALANCE_W = 0.0
ZLOSS_W = 1.0e-3


def cells(config=CONFIG):
    rows = []
    base = refill._base_overrides(config, "route")
    for seed in SEEDS:
        seed_rows = {}
        for n_experts in EXPERT_COUNTS:
            overrides = [
                *base,
                f"seed={seed}",
                f"model.n_experts={n_experts}",
                f"model.temperature={TEMPERATURE}",
                f"losses.balance_w={BALANCE_W}",
                f"losses.zloss_w={ZLOSS_W}",
                "analysis.run_mechanism=true",
                "train.save_checkpoint_epochs=[10,30,60]",
                f"run_tag=temperature_expert_count_confirm60_route_E{n_experts}_s{seed}_20260802",
            ]
            cfg = apply_overrides(load_config(config), overrides)
            seed_rows[n_experts] = (overrides, run_id_from(cfg))
        comparator = seed_rows[8][1]
        for n_experts in EXPERT_COUNTS:
            overrides, run_id = seed_rows[n_experts]
            rows.append((f"route_E{n_experts}_s{seed}", overrides, run_id, comparator))
    return rows


def launch_cells(config=CONFIG):
    """Adapter for the shared refill engine; comparator IDs remain metadata."""
    return [(tag, overrides, run_id) for tag, overrides, run_id, _ in cells(config)]


def main():
    refill.RESULT_ROOT = RESULT_ROOT
    refill.WANDB_GROUP = WANDB_GROUP
    refill.HF_PREFIX = HF_PREFIX
    refill.WANDB_JOB_TYPE = WANDB_JOB_TYPE
    refill.WANDB_TAGS = WANDB_TAGS
    refill.cells = launch_cells
    refill.main()


if __name__ == "__main__":
    main()
