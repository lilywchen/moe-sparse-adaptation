#!/usr/bin/env python
"""Locked fresh-seed confirmation of route-balanced E2 versus the shared E8 anchor.

Route E2 retained a positive worst-experiment direction at epochs 30 and 60 and
closed epoch 60 with a small positive mean OOD difference and effectively tied
ID versus route E8.  This bounded registry freezes E2 at seeds 1 and 2 and
shares the already launched/predeclared route-E8 anchors from the E16
confirmation family.  No setting is tuned: route pressure, temperature 0.03,
zero balance loss, z-loss 0.001, optimizer, data order, horizon, and checkpoint
policy all remain fixed.  The estimand is active-compute matched; OOD test is
sealed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_router_aux as refill
from scripts import sweep_rxrx1_temperature_expert_count_confirm as shared
from scripts.sweep_rxrx1_cell_dino import CONFIG


RESULT_ROOT = shared.RESULT_ROOT
WANDB_GROUP = shared.WANDB_GROUP
HF_PREFIX = shared.HF_PREFIX
WANDB_JOB_TYPE = shared.WANDB_JOB_TYPE
WANDB_TAGS = shared.WANDB_TAGS
SEEDS = shared.SEEDS


def comparator_run_ids(config=CONFIG):
    return {
        seed: run_id
        for tag, _, run_id, _ in shared.cells(config)
        for seed in SEEDS
        if tag == f"route_E8_s{seed}"
    }


def cells(config=CONFIG):
    rows = []
    base = refill._base_overrides(config, "route")
    comparators = comparator_run_ids(config)
    for seed in SEEDS:
        overrides = [
            *base,
            f"seed={seed}",
            "model.n_experts=2",
            f"model.temperature={shared.TEMPERATURE}",
            f"losses.balance_w={shared.BALANCE_W}",
            f"losses.zloss_w={shared.ZLOSS_W}",
            "analysis.run_mechanism=true",
            "train.save_checkpoint_epochs=[10,30,60]",
            f"run_tag=temperature_expert_count_confirm60_route_E2_s{seed}_20260802",
        ]
        cfg = apply_overrides(load_config(config), overrides)
        rows.append((f"route_E2_s{seed}", overrides, run_id_from(cfg), comparators[seed]))
    return rows


def launch_cells(config=CONFIG):
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
