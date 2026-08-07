"""Guards for the frontier MoE variants.

The properties tested here are the ones whose violation would silently invalidate the wave rather
than crash it: function preservation at initialisation, per-image softmax scoping in Soft MoE,
shared-only fallback for unseen oracle groups, and the fact that the diversity measure actually
responds to expert differences.
"""
import copy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from moe_shift.capacity.frontier import (
    CondLNMoEFFN,
    LowRankResidualMoEFFN,
    SharedRoutedOracleFFN,
    SoftMoEResidualFFN,
    _mean_pairwise_cosine,
)
from moe_shift.capacity.surgery import (
    SHARED_PATH_VARIANTS,
    VARIANTS,
    convert_blocks,
    set_shared_only,
    set_top_k,
)

DIM, HIDDEN = 32, 64


class Mlp(nn.Module):
    def __init__(self, c=DIM, h=HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(c, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, c)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Mlp()

    def forward(self, x):
        return self.mlp(x)


class Net(nn.Module):
    def __init__(self, depth=4):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


FRONTIER_SPECS = {
    "oracle_moe": dict(n_experts=4, expert_dropout=0.0, group_source="cell_type"),
    "condln_moe": dict(n_experts=8, top_k=1),
    "soft_moe": dict(n_experts=8, slots_per_expert=1, expert_rank=0),
    "lowrank_moe": dict(n_experts=24, top_k=8, expert_rank=16, diversity_w=0.05),
}


def _build(variant, **kwargs):
    torch.manual_seed(0)
    net = Net()
    net, report = convert_blocks(net, variant, block_indices=[2, 3], **kwargs)
    return net, report


def _reference_output(x):
    torch.manual_seed(0)
    net = Net()
    net.eval()
    with torch.no_grad():
        return net(x).clone()


# ------------------------------------------------------------------ function preservation
@pytest.mark.parametrize("variant", sorted(FRONTIER_SPECS))
def test_frontier_variants_are_function_preserving_at_init(variant):
    """Every frontier variant must equal the pretrained FFN exactly at initialisation.

    Without this, an accuracy difference could come from perturbing the pretrained representation
    rather than from conditional computation, and no arm in the wave would be interpretable.
    """
    x = torch.randn(6, 10, DIM)
    expected = _reference_output(x)
    net, _ = _build(variant, **FRONTIER_SPECS[variant])
    net.eval()
    if variant == "oracle_moe":
        for block in net._moe_blocks:
            block.set_group(torch.tensor([0, 1, 2, 3, 0, 1]))
    with torch.no_grad():
        got = net(x)
    assert torch.allclose(got, expected, atol=1e-6), f"{variant} is not function-preserving"


@pytest.mark.parametrize("variant", sorted(FRONTIER_SPECS))
def test_frontier_variants_registered_and_keep_shared_path(variant):
    assert variant in VARIANTS
    assert variant in SHARED_PATH_VARIANTS


@pytest.mark.parametrize("variant", sorted(FRONTIER_SPECS))
def test_frontier_variants_accept_4d_token_layout(variant):
    """BxHxWxC must round-trip identically to BxTxC: MoEFFN's contract accepts both."""
    x = torch.randn(4, 3, 5, DIM)
    net, _ = _build(variant, **FRONTIER_SPECS[variant])
    net.eval()
    if variant == "oracle_moe":
        for block in net._moe_blocks:
            block.set_group(torch.tensor([0, 1, 2, 3]))
    with torch.no_grad():
        out = net(x)
    assert out.shape == x.shape


# ------------------------------------------------------------------ oracle semantics
def test_oracle_unseen_group_falls_through_to_shared_path():
    """site == -1 on every OOD row must match no expert and use the shared path alone.

    This is the mechanism by which the environment-indexed ceiling reads out on held-out data.
    """
    block = SharedRoutedOracleFFN(Mlp(), n_experts=3, expert_dropout=0.0,
                                  group_source="environment")
    with torch.no_grad():
        for expert in block.experts:
            expert.fc2.weight.fill_(1.0)          # make any routed contribution obvious
            expert.fc2.bias.fill_(1.0)
    block.eval()
    x = torch.randn(4, 6, DIM)

    block.set_group(torch.tensor([-1, -1, -1, -1]))
    with torch.no_grad():
        unseen = block(x)
    block.set_shared_only = None                   # ensure we are not accidentally shared-only
    block.shared_only = True
    with torch.no_grad():
        shared = block(x)
    assert torch.allclose(unseen, shared, atol=1e-6)

    block.shared_only = False
    block.set_group(torch.tensor([0, 1, 2, 0]))
    with torch.no_grad():
        seen = block(x)
    assert not torch.allclose(seen, shared, atol=1e-6)


def test_oracle_records_applied_assignment_including_dropout():
    """`assign` must record what was APPLIED (-1 for shared-only), not what was requested."""
    block = SharedRoutedOracleFFN(Mlp(), n_experts=2, expert_dropout=1.0,
                                  group_source="cell_type")
    block.train()
    block.set_group(torch.tensor([0, 1, 0]))
    block(torch.randn(3, 4, DIM))
    assert block.top1().tolist() == [-1, -1, -1], "expert_dropout=1.0 must apply no experts"


def test_oracle_requires_group_ids():
    block = SharedRoutedOracleFFN(Mlp(), n_experts=2)
    with pytest.raises(RuntimeError, match="set_group"):
        block(torch.randn(2, 3, DIM))


def test_oracle_group_length_must_match_batch():
    block = SharedRoutedOracleFFN(Mlp(), n_experts=2)
    block.set_group(torch.tensor([0, 1, 0]))
    with pytest.raises(ValueError, match="must match batch"):
        block(torch.randn(2, 3, DIM))


def test_set_shared_only_and_set_top_k_report_what_they_touched():
    net, _ = _build("oracle_moe", **FRONTIER_SPECS["oracle_moe"])
    assert set_shared_only(net, True) == 2
    assert all(block.shared_only for block in net._moe_blocks)
    assert set_shared_only(net, False) == 2
    assert set_top_k(net, 3) == {}, "oracle blocks have no settable top-k"

    net, _ = _build("lowrank_moe", **FRONTIER_SPECS["lowrank_moe"])
    assert set_shared_only(net, True) == 0
    applied = set_top_k(net, 5)
    assert applied == {"2": 5, "3": 5}


# ------------------------------------------------------------------ Soft MoE scoping
def test_soft_moe_softmax_is_scoped_within_each_image():
    """A cross-image softmax would leak acquisition-batch information between samples.

    Concatenating two images into one batch must not change either one's output.
    """
    block = SoftMoEResidualFFN(Mlp(), n_experts=4, slots_per_expert=2)
    with torch.no_grad():
        for expert in block.experts:
            nn.init.normal_(expert.fc2.weight, std=0.05)
            nn.init.normal_(expert.fc2.bias, std=0.05)
    block.eval()
    a = torch.randn(1, 7, DIM)
    b = torch.randn(1, 7, DIM)
    with torch.no_grad():
        alone_a = block(a)
        alone_b = block(b)
        together = block(torch.cat([a, b], dim=0))
    assert torch.allclose(together[0:1], alone_a, atol=1e-5)
    assert torch.allclose(together[1:2], alone_b, atol=1e-5)


def test_soft_moe_has_no_auxiliary_loss():
    """Removing the balance/z loss is the point of this arm, so it is asserted, not assumed."""
    block = SoftMoEResidualFFN(Mlp(), n_experts=4)
    block(torch.randn(2, 5, DIM))
    assert float(block.aux_loss(balance_w=1.0, zloss_w=1.0)) == 0.0


def test_soft_moe_counts_phi_as_router_parameters():
    _net, report = _build("soft_moe", **FRONTIER_SPECS["soft_moe"])
    assert report.router_params > 0, "phi performs the routing and must appear in the budget"


def test_soft_moe_reports_all_experts_active():
    _net, report = _build("soft_moe", **FRONTIER_SPECS["soft_moe"])
    assert report.active_ffn_params == report.ffn_block_params


def test_soft_moe_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature must be positive"):
        SoftMoEResidualFFN(Mlp(), n_experts=2, temperature=0.0)


# ------------------------------------------------------------------ conditional LayerNorm
def test_condln_experts_are_affine_and_cheap():
    block = CondLNMoEFFN(Mlp(), n_experts=8, top_k=1)
    expert_params = block.gamma.numel() + block.beta.numel()
    assert expert_params == 8 * 2 * DIM
    full_ffn = sum(p.numel() for p in Mlp().parameters())
    assert expert_params < full_ffn, "affine experts must cost far less than an FFN copy"


def test_condln_descriptor_uses_token_statistics_not_content_order():
    """The descriptor must be permutation-invariant over tokens: it is a statistic, not content."""
    block = CondLNMoEFFN(Mlp(), n_experts=4, top_k=1)
    x = torch.randn(2, 9, DIM)
    shuffled = x[:, torch.randperm(9)]
    assert torch.allclose(block._descriptor(x), block._descriptor(shuffled), atol=1e-5)


def test_condln_modulate_output_is_also_function_preserving():
    block = CondLNMoEFFN(Mlp(), n_experts=4, top_k=2, modulate="output")
    reference = copy.deepcopy(block.mlp)
    block.eval()
    x = torch.randn(3, 6, DIM)
    with torch.no_grad():
        assert torch.allclose(block(x), reference(x), atol=1e-6)


def test_condln_rejects_unknown_descriptor():
    with pytest.raises(ValueError, match="descriptor"):
        CondLNMoEFFN(Mlp(), descriptor="control_wells")


# ------------------------------------------------------------------ diversity + annealing
def test_mean_pairwise_cosine_detects_identical_and_orthogonal_rows():
    identical = torch.ones(4, 8)
    assert float(_mean_pairwise_cosine(identical)) == pytest.approx(1.0, abs=1e-5)
    orthogonal = torch.eye(4, 8)
    assert float(_mean_pairwise_cosine(orthogonal)) == pytest.approx(0.0, abs=1e-5)
    assert float(_mean_pairwise_cosine(torch.ones(1, 8))) == 0.0


def test_identical_experts_score_maximum_similarity():
    """The failure the balance loss cannot see: N identical experts used uniformly.

    Copying one expert's weights into all of them must drive the measure to ~1.0.
    """
    block = LowRankResidualMoEFFN(Mlp(), n_experts=4, top_k=2, expert_rank=8, diversity_w=1.0)
    with torch.no_grad():
        for expert in block.experts:
            expert.up.weight.normal_(std=0.1)          # leave zero-init so outputs are nonzero
        source = block.experts[0].state_dict()
        for expert in block.experts[1:]:
            expert.load_state_dict(source)
    block.train()
    block(torch.randn(2, 8, DIM))
    assert float(block.expert_diversity_loss()) == pytest.approx(1.0, abs=1e-3)


def test_diversity_loss_is_measurable_without_the_diversity_term_enabled():
    """Arms with and without the diversity TERM must be comparable on the same MEASURE."""
    block = LowRankResidualMoEFFN(Mlp(), n_experts=4, top_k=2, expert_rank=8, diversity_w=0.0)
    with torch.no_grad():
        for expert in block.experts:
            expert.up.weight.normal_(std=0.1)
    block.eval()
    block(torch.randn(2, 8, DIM))
    assert block.last.get("probe") is not None
    assert float(block.expert_diversity_loss()) != 0.0


def test_diversity_term_enters_aux_loss_only_when_weighted():
    """Same weights, same input, same RNG: only diversity_w may change the aux loss."""
    x = torch.randn(2, 8, DIM)
    off = LowRankResidualMoEFFN(Mlp(), n_experts=4, top_k=2, expert_rank=8, diversity_w=0.0)
    with torch.no_grad():
        for expert in off.experts:
            expert.up.weight.normal_(std=0.1)
    on = LowRankResidualMoEFFN(Mlp(), n_experts=4, top_k=2, expert_rank=8, diversity_w=1.0)
    on.load_state_dict(off.state_dict())          # copy AFTER randomising, so both are identical

    torch.manual_seed(0); off.train(); off(x)
    torch.manual_seed(0); on.train(); on(x)
    # With balance and z weights at zero, the only remaining term is the diversity penalty.
    assert float(off.aux_loss(0.0, 0.0)) == 0.0
    assert float(on.aux_loss(0.0, 0.0)) == pytest.approx(
        float(on.expert_diversity_loss()), abs=1e-6)
    assert float(on.aux_loss(0.0, 0.0)) != 0.0


def test_top_k_annealing_is_clamped_and_changes_active_k():
    block = LowRankResidualMoEFFN(Mlp(), n_experts=6, top_k=2, expert_rank=8)
    assert block.set_top_k(6) == 6
    block(torch.randn(2, 4, DIM))
    assert block.last["active_top_k"] == 6
    assert block.set_top_k(99) == 6
    assert block.set_top_k(0) == 1
    assert block.target_top_k == 2, "the target must survive annealing"


def test_lowrank_experts_are_distinct_at_init():
    """Random low-rank factors must NOT be identical at init, unlike deepcopy(mlp) experts."""
    block = LowRankResidualMoEFFN(Mlp(), n_experts=4, top_k=1, expert_rank=8)
    first = block.experts[0].down.weight
    others = [expert.down.weight for expert in block.experts[1:]]
    assert all(not torch.allclose(first, other) for other in others)


def test_lowrank_rejects_zero_rank():
    with pytest.raises(ValueError, match="expert_rank"):
        LowRankResidualMoEFFN(Mlp(), n_experts=4, top_k=1, expert_rank=0)


def test_lowrank_rejects_top_k_above_expert_count():
    with pytest.raises(ValueError, match="top_k"):
        LowRankResidualMoEFFN(Mlp(), n_experts=2, top_k=3, expert_rank=8)


# ------------------------------------------------------------------ audit surface
@pytest.mark.parametrize("variant", sorted(FRONTIER_SPECS))
def test_every_variant_exposes_the_audit_interface(variant):
    """run_ccas calls these unconditionally; a missing one would abort the mechanism audit.

    Exercised at BLOCK level: CCASModel forwards to these, and building a real CCASModel would
    require a pretrained backbone download.
    """
    net, _ = _build(variant, **FRONTIER_SPECS[variant])
    assert net._moe_blocks, f"{variant} registered no routed block"
    for block in net._moe_blocks:
        assert callable(block.set_env)
        assert callable(block.set_group)
        assert callable(block.aux_loss)
        assert callable(block.top1)
        block.set_env(torch.zeros(4, dtype=torch.long))
        block.set_group(torch.zeros(4, dtype=torch.long))
    net.eval()
    with torch.no_grad():
        net(torch.randn(4, 5, DIM))
    for block in net._moe_blocks:
        assert block.top1() is not None
        # aux_loss must be callable after a forward pass for every variant, including the two
        # that deliberately return exactly zero.
        assert float(block.aux_loss(1e-2, 1e-3)) >= 0.0


def test_within_environment_balance_is_honoured_by_routed_variants():
    block = LowRankResidualMoEFFN(Mlp(), n_experts=4, top_k=1, expert_rank=8,
                                  balance="within_environment")
    block.set_env(torch.tensor([0, 0, 1, 1]))
    block(torch.randn(4, 3, DIM))
    assert float(block.aux_loss(1.0, 0.0)) > 0.0


def test_within_batch_alias_still_accepted():
    """Configs written before the terminology was clarified must keep working."""
    block = LowRankResidualMoEFFN(Mlp(), n_experts=2, top_k=1, expert_rank=4,
                                  balance="within_batch")
    assert block.balance == "within_environment"
