"""Convert ONE FFN block of a pretrained ViT into dense-wide / learned-MoE / frozen-router MoE,
and account for the parameter budget exactly.

Placement (plan factor 1) selects WHICH block is converted. Exactly one block is ever modified,
so depth and capacity are never confounded.
"""
from dataclasses import dataclass, asdict
from typing import Optional

import torch.nn as nn

from .ffn import MoEFFN, WideFFN, _mlp_parts
from .routers import Router

PLACEMENTS = ("early", "middle", "late")
VARIANTS = ("original", "dense_wide", "moe", "moe_frozen")


def placement_index(n_blocks: int, placement: str) -> int:
    """early ~ quarter depth, middle ~ half depth, late ~ near the end (one block of shared
    computation is deliberately left after the late placement so expert outputs can be
    recombined at all)."""
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    return {"early": max(0, n_blocks // 4),
            "middle": n_blocks // 2,
            "late": max(0, n_blocks - 2)}[placement]


@dataclass
class CapacityReport:
    variant: str
    placement: str
    block_index: int
    n_experts: int
    total_params: int
    ffn_block_params: int
    router_params: int
    bias_replica_params: int        # (E-1) * d_out: fc2-bias copies MoE has and dense-wide lacks
    active_ffn_params: int          # params touched by ONE input at the converted block
    delta_vs_reference_pct: Optional[float] = None   # vs the fixed budget P*, None if no reference

    def as_dict(self):
        return asdict(self)


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def convert_block(model, variant: str, placement: str = "middle", n_experts: int = 8,
                  top_k: int = 1, routing_unit: str = "token", geometry: str = "cosine",
                  balance: str = "global", temperature: float = 0.07,
                  sym_break_wide: float = 0.1, sym_break_moe: float = 0.0,
                  reference_total: Optional[int] = None):
    """Convert one block in-place. `model` must expose `.blocks` (list of ViT blocks with .mlp).

    `reference_total` is the fixed budget P* (normally the MoE variant's `total_params`); when
    given, `delta_vs_reference_pct` is populated, otherwise it stays None rather than a
    misleading 0.0.

    Returns (model, CapacityReport).
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("model has no .blocks — cannot place conditional capacity")

    idx = placement_index(len(blocks), placement)
    block = blocks[idx]
    mlp = block.mlp
    fc1_orig, _, fc2_orig = _mlp_parts(mlp)
    d_out, has_out_bias = fc2_orig.out_features, fc2_orig.bias is not None

    if variant == "original":
        new = mlp
    elif variant == "dense_wide":
        # The target is the corresponding MoE block, including router parameters and replicated
        # output biases. Dense width is the nearest realizable integer hidden dimension.
        target_block_params = n_experts * _count(mlp) + _count(
            Router(fc1_orig.in_features, n_experts, geometry, temperature))
        new = WideFFN(mlp, n_experts=n_experts, sym_break=sym_break_wide,
                      target_params=target_block_params)
    else:
        new = MoEFFN(mlp, n_experts=n_experts, top_k=top_k, routing_unit=routing_unit,
                     geometry=geometry, balance=balance, temperature=temperature,
                     router_frozen=(variant == "moe_frozen"), sym_break=sym_break_moe)
    block.mlp = new

    router_params = _count(new.router) if isinstance(new, MoEFFN) else 0
    block_params = _count(new)
    if isinstance(new, MoEFFN):
        active = (block_params - router_params) // n_experts * top_k + router_params
        # An MoE holds one output bias PER EXPERT; dense-wide, being a single wide Linear, holds
        # exactly one. That (E-1)*d_out excess is intrinsic to the construction, not a bug, and
        # it is the reason `moe_total - wide_total != router_params`. It is reported so the
        # budget difference can be stated exactly rather than hand-waved.
        bias_replicas = (n_experts - 1) * d_out if has_out_bias else 0
    else:
        active = block_params
        bias_replicas = 0

    total = _count(model)
    # Fixed budget P* = the MoE total (experts + router). dense-wide is compared against it.
    report = CapacityReport(
        variant=variant, placement=placement, block_index=idx, n_experts=n_experts,
        total_params=total, ffn_block_params=block_params, router_params=router_params,
        bias_replica_params=bias_replicas, active_ffn_params=active,
        delta_vs_reference_pct=(None if reference_total is None else
                                100.0 * (total - reference_total) / reference_total),
    )
    model._capacity_report = report
    model._moe_block = new if isinstance(new, MoEFFN) else None
    model._converted_index = idx
    return model, report


def budget_delta_pct(report_a: CapacityReport, report_b: CapacityReport) -> float:
    """Signed % difference in TOTAL trainable parameters (a vs b). The plan predeclares a
    tolerance below 0.1%; `check_budget` enforces it."""
    return 100.0 * (report_a.total_params - report_b.total_params) / report_b.total_params


def check_budget(report_a, report_b, tol_pct: float = 0.1):
    d = budget_delta_pct(report_a, report_b)
    ok = abs(d) < tol_pct
    return ok, d


def moe_blocks(model):
    b = getattr(model, "_moe_block", None)
    return [b] if b is not None else []
