#!/usr/bin/env python
"""Locked fresh-seed check of the bounded router-temperature survivor.

The seed-0 temperature screen selected canonical pressure, temperature 0.03, zero balance loss,
and router z-loss 0.001 after it improved both mean OOD validation and worst-experiment accuracy
over the exact early dense control.  This registry freezes that recipe at seeds 1 and 2.  It does
not duplicate dense controls: the already predeclared seed-matched early dense arms in
``tail_safe_confirm60_20260802`` are the exact comparators.

The two sparse jobs use the same data order, optimizer, 60-epoch horizon, and 10/30/60 checkpoint
policy as those dense anchors.  The fairness estimand is exact total parameters up to the already
validated 378-parameter (0.001232%) mismatch.  Selection remains OOD validation only; OOD test is
sealed.  No further tuning is permitted from these confirmation jobs.
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
from scripts.sweep_rxrx1_tail_safe_confirm import cells as tail_safe_cells


RESULT_ROOT = Path(os.environ.get(
    "MOE_RX_TEMPERATURE_LEADER_CONFIRM_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/temperature_leader_confirm60_20260802",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-temperature-leader-confirm60-20260802"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/temperature_leader_confirm60_20260802"
)
WANDB_JOB_TYPE = "rxrx1_temperature_leader_confirm60"
WANDB_TAGS = (
    "rxrx1,cell-dino,temperature-leader-confirm60,confirmatory,ood-test-blind"
)

SEEDS = (1, 2)
TEMPERATURE = 3.0e-2
BALANCE_W = 0.0
ZLOSS_W = 1.0e-3


def comparator_run_ids(config=CONFIG):
    """Return the predeclared seed-matched exact dense anchors without duplicating them."""
    return {
        seed: run_id
        for tag, _, run_id in tail_safe_cells(config)
        for seed in SEEDS
        if tag == f"dense_s{seed}"
    }


def cells(config=CONFIG):
    rows = []
    base = refill._base_overrides(config, "canonical")
    comparators = comparator_run_ids(config)
    for seed in SEEDS:
        overrides = [
            *base,
            f"seed={seed}",
            f"model.temperature={TEMPERATURE}",
            f"losses.balance_w={BALANCE_W}",
            f"losses.zloss_w={ZLOSS_W}",
            "analysis.run_mechanism=true",
            "train.save_checkpoint_epochs=[10,30,60]",
            f"run_tag=temperature_leader_confirm60_moe_s{seed}_20260802",
        ]
        cfg = apply_overrides(load_config(config), overrides)
        rows.append((f"moe_s{seed}", overrides, run_id_from(cfg), comparators[seed]))
    return rows


def launch_cells(config=CONFIG):
    """Adapter for the shared refill engine; comparator IDs remain registry metadata."""
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
