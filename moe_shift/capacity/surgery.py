"""Convert one or more ViT FFNs into matched dense-wide or sparse expert blocks.

The original single-block API remains the default.  ``convert_blocks`` makes the intervention
depth explicit so a multi-layer sparse hypothesis is compared with a dense control modified at
the exact same indices, rather than silently confounding placement and capacity.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import torch.nn as nn

from .ffn import MoEFFN, SharedResidualMoEFFN, WideFFN, _mlp_parts
from .routers import Router

PLACEMENTS = ("early", "middle", "late")
VARIANTS = ("original", "dense_wide", "moe", "moe_frozen", "shared_moe")


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
    block_indices: tuple[int, ...] = ()
    placements: tuple[str, ...] = ()
    n_converted_blocks: int = 1

    def as_dict(self):
        return asdict(self)


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def resolve_block_indices(n_blocks: int, placement: str = "middle",
                          placements: Optional[Sequence[str]] = None,
                          block_indices: Optional[Sequence[int]] = None):
    """Resolve a declarative placement list to unique, validated zero-based block indices."""
    if placements is not None and block_indices is not None:
        raise ValueError("specify placements or block_indices, not both")
    if block_indices is not None:
        indices = tuple(int(i) for i in block_indices)
        labels = tuple(f"block_{i}" for i in indices)
    else:
        labels = tuple(str(p) for p in (placements if placements is not None else (placement,)))
        indices = tuple(placement_index(n_blocks, p) for p in labels)
    if not indices:
        raise ValueError("at least one FFN block must be selected")
    if any(i < 0 or i >= n_blocks for i in indices):
        raise ValueError(f"block_indices must lie in [0, {n_blocks - 1}], got {indices}")
    if len(set(indices)) != len(indices):
        raise ValueError(f"block selections must resolve to unique indices, got {indices}")
    return indices, labels


def _convert_mlp(mlp, variant, n_experts, top_k, routing_unit, geometry, balance,
                 temperature, sym_break_wide, sym_break_moe, routing_estimator,
                 feature_stat_mix_prob, feature_stat_mix_alpha):
    fc1_orig, _, fc2_orig = _mlp_parts(mlp)
    d_out, has_out_bias = fc2_orig.out_features, fc2_orig.bias is not None

    if variant == "original":
        new = mlp
    elif variant == "dense_wide":
        target_block_params = n_experts * _count(mlp) + _count(
            Router(fc1_orig.in_features, n_experts, geometry, temperature))
        new = WideFFN(mlp, n_experts=n_experts, sym_break=sym_break_wide,
                      target_params=target_block_params)
    elif variant == "shared_moe":
        new = SharedResidualMoEFFN(
            mlp, n_experts=n_experts, top_k=top_k, routing_unit=routing_unit,
            geometry=geometry, balance=balance, temperature=temperature,
            routing_estimator=routing_estimator,
            feature_stat_mix_prob=feature_stat_mix_prob,
            feature_stat_mix_alpha=feature_stat_mix_alpha,
        )
    else:
        new = MoEFFN(mlp, n_experts=n_experts, top_k=top_k, routing_unit=routing_unit,
                     geometry=geometry, balance=balance, temperature=temperature,
                     router_frozen=(variant == "moe_frozen"), sym_break=sym_break_moe,
                     routing_estimator=routing_estimator)

    routed_types = (MoEFFN, SharedResidualMoEFFN)
    router_params = _count(new.router) if isinstance(new, routed_types) else 0
    block_params = _count(new)
    if isinstance(new, SharedResidualMoEFFN):
        shared_params = _count(new.shared)
        residual_params = sum(_count(expert) for expert in new.experts)
        active = shared_params + residual_params // n_experts * top_k + router_params
        bias_replicas = n_experts * d_out if has_out_bias else 0
    elif isinstance(new, MoEFFN):
        active = (block_params - router_params) // n_experts * top_k + router_params
        bias_replicas = (n_experts - 1) * d_out if has_out_bias else 0
    else:
        active = block_params
        bias_replicas = 0
    return new, block_params, router_params, bias_replicas, active


def convert_blocks(model, variant: str, placement: str = "middle",
                   placements: Optional[Sequence[str]] = None,
                   block_indices: Optional[Sequence[int]] = None,
                   n_experts: int = 8, top_k: int = 1, routing_unit: str = "token",
                   geometry: str = "cosine", balance: str = "global",
                   temperature: float = 0.07, sym_break_wide: float = 0.1,
                   sym_break_moe: float = 0.0, routing_estimator: str = "selected_st",
                   feature_stat_mix_prob: float = 0.0,
                   feature_stat_mix_alpha: float = 0.1,
                   reference_total: Optional[int] = None):
    """Convert selected blocks in-place; ``model`` must expose mutable blocks with ``.mlp``.

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

    indices, labels = resolve_block_indices(len(blocks), placement, placements, block_indices)
    converted = []
    block_params = router_params = bias_replicas = active = 0
    for idx in indices:
        new, bp, rp, br, ap = _convert_mlp(
            blocks[idx].mlp, variant, n_experts, top_k, routing_unit, geometry, balance,
            temperature, sym_break_wide, sym_break_moe, routing_estimator,
            feature_stat_mix_prob, feature_stat_mix_alpha)
        blocks[idx].mlp = new
        converted.append(new)
        block_params += bp
        router_params += rp
        bias_replicas += br
        active += ap

    total = _count(model)
    # Fixed budget P* = the MoE total (experts + router). dense-wide is compared against it.
    report = CapacityReport(
        variant=variant, placement="+".join(labels), block_index=indices[0], n_experts=n_experts,
        total_params=total, ffn_block_params=block_params, router_params=router_params,
        bias_replica_params=bias_replicas, active_ffn_params=active,
        delta_vs_reference_pct=(None if reference_total is None else
                                100.0 * (total - reference_total) / reference_total),
        block_indices=indices, placements=labels, n_converted_blocks=len(indices),
    )
    model._capacity_report = report
    model._moe_blocks = [
        module for module in converted
        if isinstance(module, (MoEFFN, SharedResidualMoEFFN))
    ]
    model._moe_block = model._moe_blocks[0] if model._moe_blocks else None
    model._converted_indices = indices
    model._converted_index = indices[0]
    return model, report


def convert_block(model, variant: str, placement: str = "middle", n_experts: int = 8,
                  top_k: int = 1, routing_unit: str = "token", geometry: str = "cosine",
                  balance: str = "global", temperature: float = 0.07,
                  sym_break_wide: float = 0.1, sym_break_moe: float = 0.0,
                  routing_estimator: str = "selected_st",
                  feature_stat_mix_prob: float = 0.0,
                  feature_stat_mix_alpha: float = 0.1,
                  reference_total: Optional[int] = None):
    """Backward-compatible single-block wrapper around :func:`convert_blocks`."""
    return convert_blocks(
        model, variant, placement=placement, n_experts=n_experts, top_k=top_k,
        routing_unit=routing_unit, geometry=geometry, balance=balance,
        temperature=temperature, sym_break_wide=sym_break_wide,
        sym_break_moe=sym_break_moe, routing_estimator=routing_estimator,
        feature_stat_mix_prob=feature_stat_mix_prob,
        feature_stat_mix_alpha=feature_stat_mix_alpha,
        reference_total=reference_total)


def budget_delta_pct(report_a: CapacityReport, report_b: CapacityReport) -> float:
    """Signed % difference in TOTAL trainable parameters (a vs b). The plan predeclares a
    tolerance below 0.1%; `check_budget` enforces it."""
    return 100.0 * (report_a.total_params - report_b.total_params) / report_b.total_params


def check_budget(report_a, report_b, tol_pct: float = 0.1):
    d = budget_delta_pct(report_a, report_b)
    ok = abs(d) < tol_pct
    return ok, d


def moe_blocks(model):
    blocks = getattr(model, "_moe_blocks", None)
    if blocks is not None:
        return list(blocks)
    block = getattr(model, "_moe_block", None)
    return [block] if block is not None else []
