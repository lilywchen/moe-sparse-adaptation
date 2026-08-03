#!/usr/bin/env python
"""Temperature-0.3 moderate-bank pressure controls for RxRx1 Cell-DINO.

The extreme E32/E64 temperature-0.3 route/canonical screen tests whether a smoother router helps
large expert banks.  This predeclared twelve-cell addendum asks whether the same setting has a
bounded optimum at E4/E8/E16.  It crosses route/canonical pressure, three expert counts, and
tail-safe/zero-auxiliary objectives while fixing token top-1 cosine routing, seed 0, data order,
optimizer, temperature, and 10/30/60 milestones.  Comparisons are active-compute matched only;
OOD test remains sealed and all results are exploratory.
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
    "MOE_RX_TEMPERATURE03_MODERATE_BANK_AUX_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/temperature03_moderate_bank_aux60_20260803",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-temperature03-moderate-bank-aux60-20260803"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX",
    "rxrx1/cell_dino_cp5/temperature03_moderate_bank_aux60_20260803",
)
WANDB_JOB_TYPE = "rxrx1_temperature03_moderate_bank_aux60"
WANDB_TAGS = (
    "rxrx1,cell-dino,temperature03-moderate-bank-aux60,active-compute-matched,"
    "router-design,optimization-screen,exploratory,ood-test-blind"
)

PRESSURES = ("route", "canonical")
EXPERT_COUNTS = (4, 8, 16)
TEMPERATURE = 0.3
AUX_SETTINGS = (
    ("tail_safe", 1.0e-2, 1.0e-2),
    ("no_aux", 0.0, 0.0),
)


def cells(config=CONFIG):
    rows = []
    for pressure in PRESSURES:
        fixed = refill._base_overrides(config, pressure)
        for n_experts in EXPERT_COUNTS:
            for aux_label, balance_w, zloss_w in AUX_SETTINGS:
                label = f"{pressure}_E{n_experts}_temp03_{aux_label}"
                overrides = [
                    *fixed,
                    f"model.n_experts={n_experts}",
                    f"model.temperature={TEMPERATURE}",
                    f"losses.balance_w={balance_w}",
                    f"losses.zloss_w={zloss_w}",
                    "analysis.run_mechanism=true",
                    "train.save_checkpoint_epochs=[10,30,60]",
                    f"run_tag=temperature03_moderate_bank_aux60_{label}_20260803",
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
