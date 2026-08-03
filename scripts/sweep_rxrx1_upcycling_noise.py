#!/usr/bin/env python
"""Bounded expert-symmetry-breaking screen for RxRx1 Cell-DINO sparse upcycling.

The active temperature-0.3 moderate-bank sweep supplies zero-noise tail-safe anchors.  This
addendum perturbs cloned expert input weights only, asking whether a small symmetry-breaking
initialization produces meaningful conditional specialization without sacrificing the pretrained
representation.  It is exploratory, active-compute matched, and OOD-test blind.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_expert_count as base
from scripts import sweep_rxrx1_router_aux as refill
from scripts.sweep_rxrx1_cell_dino import CONFIG


RESULT_ROOT = Path(os.environ.get(
    "MOE_RX_UPCYCLING_NOISE_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/upcycling_noise60_20260803",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-upcycling-noise60-20260803"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/upcycling_noise60_20260803"
)
WANDB_JOB_TYPE = "rxrx1_upcycling_noise60"
WANDB_TAGS = (
    "rxrx1,cell-dino,upcycling-noise60,active-compute-matched,sparse-upcycling,"
    "exploratory,ood-test-blind"
)

TEMPERATURE = 0.3
BALANCE_W = 1.0e-2
ZLOSS_W = 1.0e-2
CELLS = (
    ("route", 4, 1.0e-2),
    ("canonical", 4, 1.0e-2),
    ("route", 8, 1.0e-3),
    ("canonical", 8, 1.0e-3),
    ("route", 8, 1.0e-2),
    ("canonical", 8, 1.0e-2),
    ("route", 16, 1.0e-2),
    ("canonical", 16, 1.0e-2),
)


def cells(config=CONFIG):
    rows = []
    for pressure, n_experts, noise in CELLS:
        fixed = refill._base_overrides(config, pressure)
        noise_label = str(noise).replace(".", "p").replace("-", "m")
        label = f"{pressure}_E{n_experts}_temp03_noise{noise_label}_tail_safe"
        overrides = [
            *fixed,
            f"model.n_experts={n_experts}",
            f"model.temperature={TEMPERATURE}",
            f"model.sym_break_moe={noise}",
            f"losses.balance_w={BALANCE_W}",
            f"losses.zloss_w={ZLOSS_W}",
            "analysis.run_mechanism=true",
            "train.save_checkpoint_epochs=[10,30,60]",
            f"run_tag=upcycling_noise60_{label}_20260803",
        ]
        cfg = apply_overrides(load_config(config), overrides)
        rows.append((label, overrides, run_id_from(cfg)))
    return rows


def main():
    base.RESULT_ROOT = RESULT_ROOT
    base.WANDB_GROUP = WANDB_GROUP
    base.HF_PREFIX = HF_PREFIX
    base.WANDB_JOB_TYPE = WANDB_JOB_TYPE
    base.WANDB_TAGS = WANDB_TAGS
    base.cells = cells
    base.main()


if __name__ == "__main__":
    main()
