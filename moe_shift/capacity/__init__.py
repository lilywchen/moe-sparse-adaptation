"""Fixed-parameter-budget conditional-capacity toolkit.

    convert_block(model, variant=...)   variant in {original, dense_wide, moe, moe_frozen}

Torch-dependent symbols are imported LAZILY (PEP 562) so that planning utilities -- sweeps,
aggregators, dry-runs -- can import `run_id_from` on machines without torch installed.
"""
from .naming import run_id_from                      # torch-free

_LAZY = {
    "Router": ".routers",
    "WideFFN": ".ffn", "MoEFFN": ".ffn",
    "global_lbl": ".balance", "within_batch_lbl": ".balance",
    "within_environment_lbl": ".balance", "z_loss": ".balance",
    "convert_block": ".surgery", "convert_blocks": ".surgery",
    "placement_index": ".surgery", "resolve_block_indices": ".surgery", "check_budget": ".surgery",
    "budget_delta_pct": ".surgery", "CapacityReport": ".surgery", "moe_blocks": ".surgery",
    "PLACEMENTS": ".surgery", "VARIANTS": ".surgery",
    "CCASModel": ".model", "build_ccas": ".model",
}

__all__ = ["run_id_from", *sorted(_LAZY)]


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
