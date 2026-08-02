#!/usr/bin/env python
"""Bounded router-temperature initialization screen after router-aux60.

The router-aux60 trajectories show that pressure/auxiliary-loss gains can be transient by epoch
60.  This follow-up asks whether the learned cosine router starts in an overly sharp or overly
soft regime.  It crosses canonical/global and route/within-experiment pressure with two new
initial temperatures and three representative auxiliary settings.  The existing 0.07 runs are
shared references, not duplicated.

All 12 new cells are seed-0 E8 top-1 MoEs, use the same data order and optimizer, run to 60 epochs,
and save checkpoints at 10/30/60.  Temperature changes neither total parameters nor active
compute, so pressure pairs remain exact architectural comparisons.  This is exploratory and OOD
validation selected; any winner still requires a locked fresh-seed confirmation.
"""
import os
from pathlib import Path

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_router_aux as refill
from scripts.sweep_rxrx1_cell_dino import CONFIG


RESULT_ROOT = Path(os.environ.get(
    "MOE_RX_ROUTER_TEMP_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/router_temperature60_20260802",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-router-temperature60-20260802"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/router_temperature60_20260802"
)

PRESSURES = ("route", "canonical")
TEMPERATURES = (("t3em2", 3.0e-2), ("t2em1", 2.0e-1))
AUX_SETTINGS = (
    ("bw0_z1em3", 0.0, 1.0e-3),
    ("bw1em2_z0", 1.0e-2, 0.0),
    ("bw0_z0", 0.0, 0.0),
)


def cells(config=CONFIG):
    rows = []
    for pressure in PRESSURES:
        base = refill._base_overrides(config, pressure)
        for temp_label, temperature in TEMPERATURES:
            for aux_label, balance_w, zloss_w in AUX_SETTINGS:
                label = f"{pressure}_{temp_label}_{aux_label}"
                overrides = [
                    *base,
                    f"model.temperature={temperature}",
                    f"losses.balance_w={balance_w}",
                    f"losses.zloss_w={zloss_w}",
                    "analysis.run_mechanism=true",
                    "train.save_checkpoint_epochs=[10,30,60]",
                    f"run_tag=router_temperature60_{label}_20260802",
                ]
                cfg = apply_overrides(load_config(config), overrides)
                rows.append((label, overrides, run_id_from(cfg)))
    return rows


def main():
    refill.RESULT_ROOT = RESULT_ROOT
    refill.WANDB_GROUP = WANDB_GROUP
    refill.HF_PREFIX = HF_PREFIX
    refill.cells = cells
    refill.main()


if __name__ == "__main__":
    main()
