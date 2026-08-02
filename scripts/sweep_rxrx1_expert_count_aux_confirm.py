#!/usr/bin/env python
"""Locked fresh-seed confirmation of the canonical-E4 auxiliary effect.

The seed-0 canonical E4 pair showed a small positive mean-OOD and worst-
experiment difference for balance/z-loss 0.01 versus zero auxiliary at epochs
10, 30, and 60. This registry freezes the exact sparse pair at seeds 1 and 2.
It is an auxiliary-regularization estimand, not a sparse-versus-dense or
routing-specific claim; exact dense objective controls remain a separate
required falsifier. Selection is OOD validation only and OOD test stays sealed.
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
    "MOE_RX_EXPERT_AUX_CONFIRM_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/expert_count_aux_confirm60_20260802",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-expert-count-aux-confirm60-20260802"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/expert_count_aux_confirm60_20260802"
)
WANDB_JOB_TYPE = "rxrx1_expert_count_aux_confirm60"
WANDB_TAGS = (
    "rxrx1,cell-dino,expert-count-aux-confirm60,confirmatory,"
    "active-compute-matched,ood-test-blind"
)

SEEDS = (1, 2)
AUX_SETTINGS = (
    ("tail_safe", 1.0e-2, 1.0e-2),
    ("zero_aux", 0.0, 0.0),
)


def cells(config=CONFIG):
    rows = []
    base = refill._base_overrides(config, "canonical")
    for seed in SEEDS:
        seed_rows = {}
        for label, balance_w, zloss_w in AUX_SETTINGS:
            overrides = [
                *base,
                f"seed={seed}",
                "model.n_experts=4",
                f"losses.balance_w={balance_w}",
                f"losses.zloss_w={zloss_w}",
                "analysis.run_mechanism=true",
                "train.save_checkpoint_epochs=[10,30,60]",
                f"run_tag=expert_count_aux_confirm60_canonical_E4_{label}_s{seed}_20260802",
            ]
            cfg = apply_overrides(load_config(config), overrides)
            seed_rows[label] = (overrides, run_id_from(cfg))
        comparator = seed_rows["zero_aux"][1]
        for label, _, _ in AUX_SETTINGS:
            overrides, run_id = seed_rows[label]
            rows.append((f"canonical_E4_{label}_s{seed}", overrides, run_id, comparator))
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
