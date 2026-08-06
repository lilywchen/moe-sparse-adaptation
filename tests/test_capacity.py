"""Protocol checks for the fixed-parameter-budget conditional-capacity study.

These assert the two properties the whole comparison rests on:
  (1) FUNCTION PRESERVATION -- dense-wide, learned MoE and frozen MoE compute the same function
      as the original dense block at initialisation, so any divergence is caused by adaptation
      rather than by initialisation.
  (2) PARAMETER MATCHING    -- dense-wide and MoE hold the same total trainable parameters to
      within the predeclared 0.1% tolerance using the nearest realizable dense hidden width.
"""
import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from moe_shift.audit.routing import capture
from moe_shift.capacity import (MoEFFN, Router, WideFFN, budget_delta_pct, check_budget,
                                convert_block, convert_blocks, global_lbl, placement_index,
                                within_batch_lbl)

D, H, E, B, T = 32, 64, 4, 6, 5


class Mlp(nn.Module):
    def __init__(self, d=D, h=H):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(d, h), nn.GELU(), nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TinyBlock(nn.Module):
    def __init__(self, d=D, h=H):
        super().__init__()
        self.mlp = Mlp(d, h)

    def forward(self, x):
        return self.mlp(x)


class TinyViT(nn.Module):
    def __init__(self, n=8, d=D, h=H):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(d, h) for _ in range(n)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


@pytest.fixture
def mlp():
    torch.manual_seed(0)
    return Mlp()


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(B, T, D)


# ---------------------------------------------------------------- function preservation
def test_wide_is_function_preserving_without_symbreak(mlp, x):
    wide = WideFFN(mlp, n_experts=E, sym_break=0.0)
    assert torch.allclose(wide(x), mlp(x), atol=1e-6), "dense-wide must equal dense at init"


def test_moe_is_function_preserving(mlp, x):
    for unit in ("image", "token"):
        moe = MoEFFN(mlp, n_experts=E, routing_unit=unit, sym_break=0.0)
        assert torch.allclose(moe(x), mlp(x), atol=1e-6), f"MoE({unit}) must equal dense at init"


@pytest.mark.parametrize("routing_unit", ["image", "token"])
def test_selected_st_top1_receives_task_gradient_and_preserves_forward(mlp, x, routing_unit):
    """Regression: hard top-1 must not erase the task gradient to the router."""
    torch.manual_seed(2)
    moe = MoEFFN(
        mlp, n_experts=E, top_k=1, routing_unit=routing_unit,
        routing_estimator="selected_st", sym_break=0.0,
    )
    target = torch.randn_like(x)
    out = moe(x)
    assert torch.allclose(out, mlp(x), atol=1e-6)
    loss = (out * target).sum()
    grads = torch.autograd.grad(loss, tuple(moe.router.parameters()), allow_unused=False)
    norm = torch.stack([grad.float().square().sum() for grad in grads]).sum().sqrt()
    assert float(norm) > 1e-6, "classification loss must train a selected-ST top-1 router"


def test_routing_estimator_rejects_unknown_value(mlp):
    with pytest.raises(ValueError, match="routing_estimator"):
        MoEFFN(mlp, routing_estimator="unknown")


def test_frozen_router_is_function_preserving_and_has_no_grads(mlp, x):
    moe = MoEFFN(mlp, n_experts=E, router_frozen=True, sym_break=0.0)
    assert torch.allclose(moe(x), mlp(x), atol=1e-6)
    assert all(not p.requires_grad for p in moe.router.parameters())
    assert any(p.requires_grad for p in moe.experts.parameters()), "experts must still train"


@pytest.mark.parametrize("routing_unit, repeats", [("image", 1), ("token", T)])
def test_routing_capture_aligns_image_labels_to_assignments(mlp, x, routing_unit, repeats):
    class AuditModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_block = MoEFFN(mlp, n_experts=E, routing_unit=routing_unit)

        def forward(self, batch):
            return self.moe_block(batch)

    labels = torch.arange(B)
    sites = torch.arange(B) + 10
    expert, captured_site, captured_label = capture(
        AuditModel(), [(x, labels, sites)], torch.device("cpu"))

    assert len(expert) == B * repeats
    assert np.array_equal(captured_site, np.repeat(sites.numpy(), repeats))
    assert np.array_equal(captured_label, np.repeat(labels.numpy(), repeats))


def test_symbreak_preserves_function_exactly(mlp, x):
    wide = WideFFN(mlp, n_experts=E, sym_break=0.1)
    assert torch.allclose(wide(x), mlp(x), atol=1e-6), (
        "asymmetric outgoing splits must preserve the pretrained function")


def test_symbreak_actually_differentiates_replicas(mlp):
    """Without sym_break the widened replicas receive identical gradients and can never
    differentiate -- dense-wide would silently collapse to the original dense model."""
    torch.manual_seed(0)
    tied = WideFFN(mlp, n_experts=E, sym_break=0.0)
    torch.manual_seed(0)
    broken = WideFFN(mlp, n_experts=E, sym_break=0.1)
    xx = torch.randn(4, D)
    for m, expect_tied in ((tied, True), (broken, False)):
        m.zero_grad()
        m(xx).pow(2).sum().backward()
        g = m.fc1.weight.grad.view(E, H, D)
        same = torch.allclose(g[0], g[1], atol=1e-9)
        assert same is expect_tied, (
            "tied replicas must have identical grads; broken replicas must not")


# ---------------------------------------------------------------- parameter budget
def test_dense_width_is_the_closest_realizable_parameter_match():
    """No adjacent integer dense hidden width may be closer to the MoE block budget."""
    torch.manual_seed(0)
    _, r_moe = convert_block(TinyViT(), "moe", placement="middle", n_experts=E)
    torch.manual_seed(0)
    _, r_wide = convert_block(TinyViT(), "dense_wide", placement="middle", n_experts=E)

    gap = abs(r_moe.ffn_block_params - r_wide.ffn_block_params)
    one_hidden_unit = D + D + 1
    assert r_moe.bias_replica_params == (E - 1) * D
    assert r_wide.bias_replica_params == 0, "dense-wide has a single output bias"
    assert gap <= one_hidden_unit / 2, (
        f"a closer integer dense width exists: gap {gap} > half-step {one_hidden_unit / 2}")


@pytest.mark.parametrize("d,h,n_blocks,n_exp", [(192, 768, 12, 8), (384, 1536, 12, 8)])
def test_budget_matched_within_tolerance_at_study_scale(d, h, n_blocks, n_exp):
    """The 0.1% tolerance is a claim about the backbone the study actually runs (DINOv2 ViT-S/14:
    d=384, h=1536, 12 blocks). On the toy fixture the router plus bias replicas are ~0.5% of a
    46k-param model, so checking the tolerance there measures the fixture, not the protocol.
    """
    torch.manual_seed(0)
    _, r_moe = convert_block(TinyViT(n=n_blocks, d=d, h=h), "moe", n_experts=n_exp)
    torch.manual_seed(0)
    _, r_wide = convert_block(TinyViT(n=n_blocks, d=d, h=h), "dense_wide", n_experts=n_exp)
    ok, delta = check_budget(r_wide, r_moe, tol_pct=0.1)
    assert ok, f"dense-wide vs MoE budget differs by {delta:.4f}% (tolerance 0.1%)"


def test_delta_vs_reference_is_none_unless_a_reference_is_given():
    """A silently-zero delta would be logged into every result JSON and read as 'perfectly
    matched'. Absent a reference it must be None."""
    torch.manual_seed(0)
    _, r_moe = convert_block(TinyViT(), "moe", n_experts=E)
    assert r_moe.delta_vs_reference_pct is None

    torch.manual_seed(0)
    _, r_wide = convert_block(TinyViT(), "dense_wide", n_experts=E,
                              reference_total=r_moe.total_params)
    assert r_wide.delta_vs_reference_pct is not None
    assert r_wide.delta_vs_reference_pct < 0, "dense-wide sits just under the MoE budget P*"
    assert abs(r_wide.delta_vs_reference_pct - budget_delta_pct(r_wide, r_moe)) < 1e-9


def test_original_is_smaller_than_fixed_budget():
    torch.manual_seed(0)
    _, r_orig = convert_block(TinyViT(), "original")
    torch.manual_seed(0)
    _, r_moe = convert_block(TinyViT(), "moe", n_experts=E)
    assert r_orig.total_params < r_moe.total_params
    assert budget_delta_pct(r_orig, r_moe) < 0


def test_active_compute_ordering():
    """top-1 MoE activates ~one expert; dense-wide activates all E."""
    torch.manual_seed(0)
    _, r_moe = convert_block(TinyViT(), "moe", n_experts=E, top_k=1)
    torch.manual_seed(0)
    _, r_wide = convert_block(TinyViT(), "dense_wide", n_experts=E)
    assert r_moe.active_ffn_params < r_wide.active_ffn_params


# ---------------------------------------------------------------- routers
def test_routers_are_parameter_matched():
    lin, cos = Router(D, E, "linear"), Router(D, E, "cosine")
    assert sum(p.numel() for p in lin.parameters()) == sum(p.numel() for p in cos.parameters())


def test_cosine_router_is_scale_invariant_linear_is_not():
    torch.manual_seed(0)
    z = torch.randn(4, D)
    cos, lin = Router(D, E, "cosine"), Router(D, E, "linear")
    assert torch.allclose(cos(z), cos(z * 7.0), atol=1e-5), "cosine must ignore feature norm"
    assert not torch.allclose(lin(z), lin(z * 7.0), atol=1e-3), "linear must use feature norm"


# ---------------------------------------------------------------- balance losses
def test_within_batch_equals_global_when_single_environment():
    torch.manual_seed(0)
    probs = torch.rand(24, E).softmax(-1)
    assign = probs.argmax(-1)
    env = torch.zeros(24, dtype=torch.long)
    assert torch.allclose(within_batch_lbl(probs, assign, E, env), global_lbl(probs, assign, E))


def test_within_batch_penalises_batch_to_expert_monopoly():
    """Routes that are globally balanced but perfectly batch-partitioned should score worse
    under within-batch balancing than under global balancing."""
    n_env = E
    probs = torch.full((4 * n_env, E), 1e-6)
    assign, env = [], []
    for b in range(n_env):                      # environment b -> expert b, every time
        for _ in range(4):
            probs[len(assign), b] = 1.0
            assign.append(b)
            env.append(b)
    probs = probs / probs.sum(-1, keepdim=True)
    assign, env = torch.tensor(assign), torch.tensor(env)
    g = global_lbl(probs, assign, E)
    w = within_batch_lbl(probs, assign, E, env)
    assert w > g, f"within-batch ({w:.3f}) must penalise monopolies more than global ({g:.3f})"


# ---------------------------------------------------------------- placement
def test_placement_indices_distinct_and_ordered():
    n = 12
    e, m, l = (placement_index(n, p) for p in ("early", "middle", "late"))
    assert e < m < l < n, (e, m, l)
    assert l <= n - 2, "late placement must leave shared computation after the block"


def test_convert_touches_exactly_one_block():
    torch.manual_seed(0)
    model, rep = convert_block(TinyViT(n=8), "moe", placement="late", n_experts=E)
    converted = [i for i, b in enumerate(model.blocks) if isinstance(b.mlp, (MoEFFN, WideFFN))]
    assert converted == [rep.block_index], converted


def test_convert_multiple_blocks_is_explicit_and_function_preserving():
    torch.manual_seed(0)
    reference = TinyViT(n=8)
    model = copy.deepcopy(reference)
    inputs = torch.randn(3, 5, D)
    model, report = convert_blocks(
        model, "moe", block_indices=[1, 5], n_experts=E, sym_break_moe=0.0)
    converted = [i for i, block in enumerate(model.blocks) if isinstance(block.mlp, MoEFFN)]
    assert converted == [1, 5]
    assert report.block_indices == (1, 5)
    assert report.n_converted_blocks == 2
    assert len(model._moe_blocks) == 2
    assert torch.allclose(model(inputs), reference(inputs), atol=1e-6)


def test_multiblock_dense_uses_nearest_realisable_width_per_layer():
    torch.manual_seed(0)
    _, sparse = convert_blocks(TinyViT(n=8), "moe", block_indices=[1, 5], n_experts=E)
    torch.manual_seed(0)
    _, dense = convert_blocks(TinyViT(n=8), "dense_wide", block_indices=[1, 5], n_experts=E)
    # The toy model is too small for a whole-model percentage tolerance: its irreducible
    # one-hidden-unit rounding step is itself about 0.1%.  Check the exact realizability bound.
    one_hidden_unit = D + D + 1
    assert abs(dense.ffn_block_params - sparse.ffn_block_params) <= one_hidden_unit


def test_multiblock_dense_and_moe_are_matched_at_study_scale():
    torch.manual_seed(0)
    _, sparse = convert_blocks(
        TinyViT(n=12, d=192, h=768), "moe", block_indices=[1, 5, 9], n_experts=8)
    torch.manual_seed(0)
    _, dense = convert_blocks(
        TinyViT(n=12, d=192, h=768), "dense_wide", block_indices=[1, 5, 9], n_experts=8)
    ok, delta = check_budget(dense, sparse)
    assert ok, delta


# ---------------------------------------------------------------- routing units
def test_image_routing_is_constant_within_an_image(mlp, x):
    moe = MoEFFN(mlp, n_experts=E, routing_unit="image", sym_break=0.0)
    moe(x)
    assert moe.last["assign"].shape[0] == B, "image routing = one decision per image"


def test_token_routing_decides_per_token(mlp, x):
    moe = MoEFFN(mlp, n_experts=E, routing_unit="token", sym_break=0.0)
    moe(x)
    assert moe.last["assign"].shape[0] == B * T, "token routing = one decision per token"
