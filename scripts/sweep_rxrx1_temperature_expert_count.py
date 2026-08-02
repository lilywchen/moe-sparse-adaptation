#!/usr/bin/env python
"""Expert-count test at the provisional low-temperature router optimum.

The seed-0 temperature screen selected canonical pressure with temperature 0.03, zero load-
balance loss, and router z-loss 0.001.  This bounded architecture screen asks whether that gain
depends on the size of the available expert bank.  It crosses 4 versus 16 experts with canonical
versus within-experiment route pressure.  The completed E8 pressure pair is the shared anchor and
is not duplicated.

All four jobs are seed-0, top-1, early token-cosine MoEs with the same optimizer, data order,
60-epoch horizon, and 10/30/60 checkpoint policy.  Active expert compute is matched while total
parameters intentionally vary, so this family supports only an active-compute-matched claim.
Selection remains OOD validation; OOD test is sealed and every result is exploratory.
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
    "MOE_RX_TEMPERATURE_EXPERT_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/temperature_expert_count60_20260802",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-temperature-expert-count60-20260802"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/temperature_expert_count60_20260802"
)
WANDB_JOB_TYPE = "rxrx1_temperature_expert_count60"
WANDB_TAGS = (
    "rxrx1,cell-dino,temperature-expert-count60,exploratory,active-compute-matched,"
    "ood-test-blind"
)

PRESSURES = ("canonical", "route")
EXPERT_COUNTS = (4, 16)
TEMPERATURE = 3.0e-2
BALANCE_W = 0.0
ZLOSS_W = 1.0e-3


def cells(config=CONFIG):
    rows = []
    for pressure in PRESSURES:
        base = refill._base_overrides(config, pressure)
        for n_experts in EXPERT_COUNTS:
            overrides = [
                *base,
                f"model.n_experts={n_experts}",
                f"model.temperature={TEMPERATURE}",
                f"losses.balance_w={BALANCE_W}",
                f"losses.zloss_w={ZLOSS_W}",
                "analysis.run_mechanism=true",
                "train.save_checkpoint_epochs=[10,30,60]",
                f"run_tag=temperature_expert_count60_{pressure}_E{n_experts}_20260802",
            ]
            cfg = apply_overrides(load_config(config), overrides)
            rows.append((f"{pressure}_E{n_experts}", overrides, run_id_from(cfg)))
    return rows


def main():
    refill.RESULT_ROOT = RESULT_ROOT
    refill.WANDB_GROUP = WANDB_GROUP
    refill.HF_PREFIX = HF_PREFIX
    refill.WANDB_JOB_TYPE = WANDB_JOB_TYPE
    refill.WANDB_TAGS = WANDB_TAGS
    refill.cells = cells
    refill.main()


if __name__ == "__main__":
    main()
