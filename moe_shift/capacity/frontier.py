"""Frontier MoE variants. Each targets one MEASURED failure of the replacement and
shared/residual arms, and each is EXACTLY function-preserving at initialisation.

Why these four exist
--------------------
Completed campaigns established three constraints on any new design:

1. ``route_reliance`` never exceeded ``0.0065`` against a predeclared ``0.01`` gate, and the
   learned-minus-frozen reliance gap was negative on average.  Randomising the routes costs
   almost nothing, so experts had converged to near-interchangeable functions.
2. Train accuracy reaches ``1.0`` by epoch 30 (``0.48`` at epoch 10).  The selected-ST estimator
   is the ONLY path from cross-entropy to the router and its gradient is scaled by
   ``d(loss)/d(topv)``.  When the task loss saturates that gradient vanishes, while ``balance_w``
   and ``zloss_w`` keep pushing toward uniform assignment for the rest of the run.
3. One full-width FFN expert is ~1.18M parameters and RxRx1 supplies ~35.7 images per class
   (~1.1 per class per experiment).  Dispatching to E such experts divides supervision that was
   already thin, consistent with every "more experts" arm losing.

The four variants attack (1), (2) and (3) from different directions:

``SharedRoutedOracleFFN``
    Ground-truth-indexed routed experts over an always-active pretrained FFN.  Not deployable --
    it is the CEILING bounding what any learned router could achieve.  ViT port of
    ``moe_shift/models/oracle.py``, whose decision rule it preserves.

``CondLNMoEFFN``
    Experts are LayerNorm-style affine modulations (``2*C`` parameters each) routed on a
    descriptor built from token feature STATISTICS rather than content.  Batch effects are
    largely affine shifts of feature statistics, so an affine expert is the matched instrument,
    and 768 parameters per expert is a budget this dataset can actually supervise.

``SoftMoEResidualFFN``
    Fully differentiable Soft MoE (Puigcerver et al., 2023) on the residual branch: no argmax, no
    straight-through estimator, no auxiliary balance loss.  Removes failure (2) by construction.

``LowRankResidualMoEFFN``
    Fine-grained low-rank residual experts (DeepSeekMoE granularity, combined with the shared
    expert ``SharedResidualMoEFFN`` already provides) plus two things the existing code lacks: an
    explicit expert-DIVERSITY objective (the canonical balance loss balances *usage*, not
    *function*, and is satisfied perfectly by N identical experts used uniformly), and
    dense-to-sparse top-k annealing so experts differentiate while still data-rich.
"""
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .balance import global_lbl, within_environment_lbl, z_loss
from .routers import Router


def _mlp_shape(mlp: nn.Module):
    """(d_in, d_hidden, d_out) of a timm-style Mlp. Local to avoid an ffn<->frontier cycle."""
    fc1, fc2 = getattr(mlp, "fc1", None), getattr(mlp, "fc2", None)
    if fc1 is None or fc2 is None:
        raise TypeError(f"expected an Mlp with .fc1/.fc2, got {type(mlp).__name__}")
    return fc1.in_features, fc1.out_features, fc2.out_features


def _as_tokens(x: torch.Tensor):
    """BxHxWxC or BxTxC -> (tokens BxTxC, original shape). Mirrors MoEFFN's contract."""
    orig = x.shape
    if x.ndim == 4:
        return x.reshape(orig[0], orig[1] * orig[2], orig[3]), orig
    if x.ndim == 3:
        return x, orig
    raise ValueError(f"expected BxTxC or BxHxWxC tokens, got {tuple(orig)}")


def _restore(x: torch.Tensor, orig):
    return x.reshape(orig) if len(orig) == 4 else x


def _mean_pairwise_cosine(rows: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal cosine similarity between the rows of ``rows``.

    This is the quantity the canonical balance loss does NOT control.  ``E * sum_e f_e * P_e`` is
    minimised perfectly by N identical experts used uniformly, which is the equilibrium the
    completed runs measured (route reliance <= 0.0065).  A value near 1.0 here means the experts
    are interchangeable no matter how balanced their usage is.
    """
    n = rows.shape[0]
    if n < 2:
        return rows.new_zeros(())
    normalised = F.normalize(rows.float(), dim=-1)
    similarity = normalised @ normalised.t()
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=similarity.device)
    return similarity[off_diagonal].mean()


class _RoutedMixin(nn.Module):
    """Shared bookkeeping for the routed variants: environment ids, stats, audit hooks."""

    #: Tokens kept per forward pass for the expert-diversity probe. Small on purpose: the probe is
    #: a diagnostic and a regulariser, not a second training signal.
    probe_tokens = 64

    def __init__(self):
        super().__init__()
        self._env = None
        self._group = None
        self.last = None

    def _record_probe(self, flat: torch.Tensor):
        """Stash a small DETACHED token probe so expert diversity is measurable every forward.

        Recorded unconditionally, not only when ``diversity_w > 0``: the arm that adds the
        diversity term and the arm that does not must be comparable on the same metric, otherwise
        there is no way to tell whether the term actually changed expert diversity.

        The probe is detached because the objective is to shape the EXPERTS, not to push tokens
        around -- gradient still reaches every expert's parameters through ``expert(probe)``.
        """
        if self.last is None:
            return
        n = min(int(self.probe_tokens), flat.shape[0])
        if n <= 0:
            return
        index = torch.randperm(flat.shape[0], device=flat.device)[:n]
        self.last["probe"] = flat[index].detach()

    def _probe_expert_outputs(self, experts) -> Optional[torch.Tensor]:
        probe = None if self.last is None else self.last.get("probe")
        if probe is None or probe.numel() == 0 or len(experts) < 2:
            return None
        return torch.stack([expert(probe).flatten() for expert in experts])

    def expert_diversity_loss(self) -> torch.Tensor:
        """Mean pairwise cosine similarity between expert OUTPUTS on the stored token probe.

        Minimising this pushes experts toward computing different functions.  Returns exactly zero
        when no probe has been recorded yet, so it is safe to call before the first forward.
        """
        device = next(self.parameters()).device
        outputs = self._probe_expert_outputs(getattr(self, "experts", []))
        if outputs is None:
            return torch.zeros((), device=device)
        return _mean_pairwise_cosine(outputs)

    def set_env(self, env):
        """Per-image acquisition-environment ids; used ONLY by within-environment balancing."""
        self._env = env

    def set_group(self, group):
        """Per-image ORACLE group ids (cell type or train-remapped experiment).

        Consumed only by :class:`SharedRoutedOracleFFN`.  Defined on every routed variant so the
        runner can call it unconditionally without knowing which variant it holds.
        """
        self._group = group

    def top1(self):
        if self.last is None or self.last.get("assign") is None:
            raise RuntimeError("no routing recorded; run a forward pass first")
        return self.last["assign"].detach().cpu()

    def _balance_terms(self, balance_w, zloss_w):
        probs, assign, env = self.last["probs"], self.last["assign"], self.last["env"]
        if probs is None:
            return torch.zeros((), device=self.last["logits"].device)
        if self.balance == "within_environment":
            lb = within_environment_lbl(probs, assign, self.n_experts, env)
        else:
            lb = global_lbl(probs, assign, self.n_experts)
        return balance_w * lb + zloss_w * z_loss(self.last["logits"])


# --------------------------------------------------------------------------------------------
# Arms 1 + 2.  Oracle ceiling: shared pretrained FFN + ground-truth-indexed routed experts.
# --------------------------------------------------------------------------------------------
class SharedRoutedOracleFFN(_RoutedMixin):
    """``out = shared(x) + routed[true_group](x)`` -- a CEILING, not a deployable method.

    ViT-FFN port of :class:`moe_shift.models.oracle.SharedRoutedBlock`, preserving its semantics:

    * routing is by a GROUND-TRUTH group label, so each routed expert can absorb its own group
      and the shared path is maximally free to carry only content;
    * ``expert_dropout`` forces a fraction of in-group samples through the shared path alone
      during training, so ``shared(x)`` becomes self-sufficient and shared-only inference on an
      unseen group is meaningful;
    * a sample whose group id falls outside ``[0, n_experts)`` -- exactly what the train-remapped
      site index is on any OOD split (``-1``) -- matches no expert, so its routed contribution
      stays zero and it falls through to the shared path automatically.

    Decision rule (unchanged from ``oracle.py``): if shared-only held-out accuracy does not beat
    the parameter-matched dense control, batch and content are not separable here even with
    ground truth, and the remedy is illusory.
    """

    def __init__(self, mlp, n_experts: int = 4, expert_dropout: float = 0.5,
                 group_source: str = "cell_type"):
        super().__init__()
        if int(n_experts) < 1:
            raise ValueError("oracle routing requires at least one routed expert")
        if group_source not in ("cell_type", "environment"):
            raise ValueError(f"group_source must be cell_type|environment, got {group_source!r}")
        self.n_experts = int(n_experts)
        self.top_k = 1
        self.expert_dropout = float(expert_dropout)
        self.group_source = str(group_source)
        self.balance = "global"          # no learned router; kept for interface uniformity
        self.shared_only = False

        self.shared = copy.deepcopy(mlp)
        self.experts = nn.ModuleList([copy.deepcopy(mlp) for _ in range(self.n_experts)])
        with torch.no_grad():
            for expert in self.experts:
                # Zeroing only the output projection makes every routed branch exactly zero at
                # init without tying later updates: the oracle index already separates experts.
                expert.fc2.weight.zero_()
                if expert.fc2.bias is not None:
                    expert.fc2.bias.zero_()

    def forward(self, x):
        tokens, orig = _as_tokens(x)
        B, T, C = tokens.shape
        shared_out = self.shared(_restore(tokens, orig))
        if self.shared_only:
            self.last = {"logits": None, "probs": None, "assign": None,
                         "env": self._env, "tokens_per_image": T, "shared_only": True}
            return shared_out

        if self._group is None:
            raise RuntimeError(
                "oracle routing needs group ids; call model.set_group(...) before forward")
        group = self._group.to(tokens.device).reshape(-1)
        if group.shape[0] != B:
            raise ValueError(f"group ids ({group.shape[0]}) must match batch ({B})")

        # Expert dropout: during training a fraction of in-group samples skip their routed expert
        # so the shared path is trained to classify on its own (required for shared-only eval).
        if self.training and self.expert_dropout > 0:
            keep = torch.rand(B, device=tokens.device) > self.expert_dropout
        else:
            keep = torch.ones(B, dtype=torch.bool, device=tokens.device)

        flat = tokens.reshape(B * T, C)
        correction = torch.zeros_like(flat)
        for index in range(self.n_experts):
            mask = (group == index) & keep
            if mask.any():
                token_mask = mask.repeat_interleave(T)
                correction[token_mask] = self.experts[index](flat[token_mask]).to(
                    correction.dtype)
        correction = _restore(correction.reshape(B, T, C), orig)

        # `assign` records the oracle decision actually applied (-1 = shared-only), so the
        # routing audit verifies the assignment instead of trusting it.
        applied = torch.where(keep & (group >= 0) & (group < self.n_experts),
                              group, torch.full_like(group, -1))
        self.last = {"logits": None, "probs": None, "assign": applied,
                     "env": self._env, "tokens_per_image": T, "shared_only": False}
        # Recorded even for the oracle: "do ground-truth-indexed experts actually end up
        # computing different functions?" is one of the things this ceiling is here to answer.
        self._record_probe(flat)
        return shared_out + correction

    def aux_loss(self, balance_w: float, zloss_w: float = 0.0):
        # No learned router: no routing distribution to balance or regularise.
        return torch.zeros((), device=self.shared.fc1.weight.device)


# --------------------------------------------------------------------------------------------
# Arm 3.  Conditional-LayerNorm MoE routed on feature STATISTICS.
# --------------------------------------------------------------------------------------------
class CondLNMoEFFN(_RoutedMixin):
    """Affine (LayerNorm-style) experts routed on a token-statistics descriptor.

    Each expert is a ``(gamma, beta)`` pair of width ``C`` -- ``2*C`` parameters versus ~1.18M for
    a full FFN expert.  At ViT-S width that is 768 per expert, so 8 experts cost 6,144 and 33
    would cost 25,344: a budget RxRx1's ~35.7 images per class can actually supervise.

    The descriptor is ``concat(mean_over_tokens(x), std_over_tokens(x))`` per image.  This routes
    on the direction acquisition batch effects actually occupy (an affine shift of feature
    statistics) rather than on content, and is computable at test time from the input alone -- no
    labels, no control wells, no transduction.

    ``gamma``/``beta`` are zero-initialised and applied as ``x * (1 + gamma) + beta``, so the
    module equals the pretrained FFN exactly at initialisation.
    """

    def __init__(self, mlp, n_experts: int = 8, top_k: int = 1, geometry: str = "cosine",
                 balance: str = "global", temperature: float = 0.07,
                 descriptor: str = "token_stats", modulate: str = "input"):
        super().__init__()
        if balance == "within_batch":
            balance = "within_environment"
        if balance not in ("global", "within_environment"):
            raise ValueError(f"balance must be global|within_environment, got {balance!r}")
        if descriptor != "token_stats":
            raise ValueError(f"unsupported descriptor: {descriptor!r}")
        if modulate not in ("input", "output"):
            raise ValueError(f"modulate must be input|output, got {modulate!r}")
        if not 1 <= int(top_k) <= int(n_experts):
            raise ValueError("conditional-LN MoE requires 1 <= top_k <= n_experts")

        d_in, _, d_out = _mlp_shape(mlp)
        self.n_experts, self.top_k = int(n_experts), int(top_k)
        self.balance = balance
        self.descriptor = str(descriptor)
        self.modulate = str(modulate)
        self.mlp = copy.deepcopy(mlp)
        width = d_in if self.modulate == "input" else d_out
        self.width = int(width)
        # Zero init -> (1 + gamma) = 1, beta = 0 -> exact function preservation at init.
        self.gamma = nn.Parameter(torch.zeros(self.n_experts, width))
        self.beta = nn.Parameter(torch.zeros(self.n_experts, width))
        self.router = Router(2 * d_in, self.n_experts, geometry, temperature)

    def _descriptor(self, tokens: torch.Tensor) -> torch.Tensor:
        mean = tokens.mean(dim=1)
        std = tokens.var(dim=1, unbiased=False).clamp_min(1e-12).sqrt()
        return torch.cat((mean, std), dim=-1)

    def _mix(self, tokens: torch.Tensor):
        """-> (gamma, beta) per image, mixed over the selected experts."""
        B = tokens.shape[0]
        logits = self.router(self._descriptor(tokens))            # [B, E]
        probs = logits.softmax(dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)
        weights = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        gamma = torch.zeros(B, self.width, device=tokens.device, dtype=tokens.dtype)
        beta = torch.zeros(B, self.width, device=tokens.device, dtype=tokens.dtype)
        for slot in range(self.top_k):
            ids, w = topi[:, slot], weights[:, slot].unsqueeze(-1)
            gamma = gamma + w * self.gamma.to(tokens.dtype)[ids]
            beta = beta + w * self.beta.to(tokens.dtype)[ids]
        self.last = {"logits": logits, "probs": probs, "assign": topi[:, 0].detach(),
                     "env": self._env, "tokens_per_image": tokens.shape[1],
                     "descriptor": self.descriptor}
        return gamma.unsqueeze(1), beta.unsqueeze(1)

    def forward(self, x):
        tokens, orig = _as_tokens(x)
        gamma, beta = self._mix(tokens)
        if self.modulate == "input":
            modulated = tokens * (1.0 + gamma) + beta
            return self.mlp(_restore(modulated, orig))
        out, _ = _as_tokens(self.mlp(_restore(tokens, orig)))
        return _restore(out * (1.0 + gamma) + beta, orig)

    def expert_diversity_loss(self) -> torch.Tensor:
        """Affine experts have no probe: their function IS their ``(gamma, beta)`` row, so the
        cosine is taken directly on the parameters rather than on sampled outputs."""
        return _mean_pairwise_cosine(torch.cat((self.gamma, self.beta), dim=-1))

    def aux_loss(self, balance_w: float, zloss_w: float = 0.0):
        if self.last is None:
            return torch.zeros((), device=self.gamma.device)
        return self._balance_terms(balance_w, zloss_w)


# --------------------------------------------------------------------------------------------
# Arm 5.  Soft MoE residual branch -- no argmax, no straight-through, no balance loss.
# --------------------------------------------------------------------------------------------
class _LowRankExpert(nn.Module):
    """Bottleneck residual expert: Linear(C->r) -> GELU -> Linear(r->C), ``up`` zero-initialised.

    At rank 16 and ViT-S width this is ~12k parameters against ~1.18M for a full FFN expert, so
    the same total budget buys far more supervision per parameter.  Random ``down`` factors are
    also distinct from step zero, unlike ``copy.deepcopy(mlp)`` experts with ``sym_break=0``,
    which are identical at init and separate only by gradient noise.
    """

    def __init__(self, dim: int, rank: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = dim if out_dim is None else int(out_dim)
        self.down = nn.Linear(dim, int(rank))
        self.act = nn.GELU()
        self.up = nn.Linear(int(rank), out_dim)
        nn.init.normal_(self.down.weight, std=dim ** -0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


class SoftMoEResidualFFN(_RoutedMixin):
    """Always-active pretrained FFN plus a Soft MoE residual branch.

    Soft MoE (Puigcerver et al., 2023) replaces discrete assignment with two softmaxes over a
    learned ``[C, E*S]`` slot projection ``phi``::

        logits   = tokens @ phi                          [T, E*S]
        dispatch = softmax(logits, over TOKENS)  ->  slot inputs = dispatch^T @ tokens
        combine  = softmax(logits, over SLOTS)   ->  output      = combine @ slot_out

    Both softmaxes are taken WITHIN an image, never across the minibatch.  A cross-image softmax
    would let one sample's tokens determine another's representation, which on this dataset would
    silently leak acquisition-batch information between samples.

    No argmax means no straight-through estimator, so the branch keeps a dense gradient path even
    after the task loss saturates.  There is likewise no routing distribution to balance, so
    ``aux_loss`` is exactly zero and ``balance_w``/``zloss_w`` cannot dominate the late run.
    """

    def __init__(self, mlp, n_experts: int = 8, slots_per_expert: int = 1,
                 expert_rank: int = 0, temperature: float = 1.0):
        super().__init__()
        if int(n_experts) < 1 or int(slots_per_expert) < 1:
            raise ValueError("soft MoE requires n_experts >= 1 and slots_per_expert >= 1")
        if float(temperature) <= 0:
            raise ValueError("soft MoE temperature must be positive")
        d_in, _, d_out = _mlp_shape(mlp)
        if d_in != d_out:
            raise ValueError("soft MoE residual branch requires d_in == d_out")
        self.n_experts = int(n_experts)
        self.slots_per_expert = int(slots_per_expert)
        self.n_slots = self.n_experts * self.slots_per_expert
        self.expert_rank = int(expert_rank)
        self.temperature = float(temperature)
        self.balance = "global"
        self.top_k = self.n_experts          # every expert is active, by construction

        self.shared = copy.deepcopy(mlp)
        self.norm = nn.LayerNorm(d_in)
        if self.expert_rank > 0:
            self.experts = nn.ModuleList(
                [_LowRankExpert(d_in, self.expert_rank) for _ in range(self.n_experts)])
        else:
            experts = []
            for _ in range(self.n_experts):
                expert = copy.deepcopy(mlp)
                with torch.no_grad():
                    expert.fc2.weight.zero_()
                    if expert.fc2.bias is not None:
                        expert.fc2.bias.zero_()
                experts.append(expert)
            self.experts = nn.ModuleList(experts)
        self.phi = nn.Parameter(torch.empty(d_in, self.n_slots))
        nn.init.normal_(self.phi, std=d_in ** -0.5)

    def forward(self, x):
        tokens, orig = _as_tokens(x)
        B, T, C = tokens.shape
        shared_out = self.shared(_restore(tokens, orig))

        normed = self.norm(tokens)                                       # [B, T, C]
        logits = normed @ self.phi.to(normed.dtype) / self.temperature   # [B, T, n_slots]
        dispatch = logits.softmax(dim=1)                            # over TOKENS, within image
        combine = logits.softmax(dim=2)                             # over SLOTS,  within image
        slots = torch.einsum("bts,btc->bsc", dispatch, tokens)      # [B, n_slots, C]

        slot_out = torch.zeros_like(slots)
        for index, expert in enumerate(self.experts):
            lo = index * self.slots_per_expert
            hi = lo + self.slots_per_expert
            slot_out[:, lo:hi] = expert(slots[:, lo:hi]).to(slot_out.dtype)
        correction = torch.einsum("bts,bsc->btc", combine, slot_out)

        # Soft MoE makes no discrete choice; record the dominant slot's expert so the routing
        # audit still has a well-defined per-token assignment to measure usage/entropy on.
        dominant = combine.reshape(B * T, self.n_slots).argmax(dim=-1) // self.slots_per_expert
        self.last = {"logits": logits.reshape(B * T, self.n_slots),
                     "probs": None, "assign": dominant.detach(),
                     "env": (None if self._env is None else self._env.repeat_interleave(T)),
                     "tokens_per_image": T, "soft": True}
        self._record_probe(tokens.reshape(B * T, C))
        return shared_out + _restore(correction, orig)

    def aux_loss(self, balance_w: float, zloss_w: float = 0.0):
        # Soft assignment has no load to balance. Returning exactly zero is the point of the arm.
        return torch.zeros((), device=self.phi.device)


# --------------------------------------------------------------------------------------------
# Arms 4 + 6 + 8 base.  Fine-grained low-rank residual MoE, diversity, dense-to-sparse anneal.
# --------------------------------------------------------------------------------------------
class LowRankResidualMoEFFN(_RoutedMixin):
    """Shared pretrained FFN plus fine-grained low-rank routed residual experts.

    Three additions over :class:`~moe_shift.capacity.ffn.SharedResidualMoEFFN`:

    ``expert_rank``
        Residual experts are rank-limited bottlenecks rather than full FFN copies, so E can be
        large without dividing supervision into unsupportable slices (DeepSeekMoE granularity,
        combined with the always-active shared expert this class keeps).

    ``diversity_w``
        The canonical balance loss ``E * sum_e f_e * P_e`` balances expert USAGE, not expert
        FUNCTION, and is satisfied perfectly by N identical experts used uniformly -- the
        equilibrium the completed runs actually measured (route reliance <= 0.0065).  This term
        penalises mean pairwise cosine similarity between expert OUTPUTS on a random token probe,
        so differing is rewarded directly rather than hoped for.

    ``set_top_k``
        Dense-to-sparse annealing.  Training can start with every expert active (each expert sees
        all tokens while data is still plentiful) and anneal to the target ``top_k``, instead of
        starting sparse and starving every expert from step zero.
    """

    def __init__(self, mlp, n_experts: int = 24, top_k: int = 8, expert_rank: int = 16,
                 routing_unit: str = "token", geometry: str = "cosine",
                 balance: str = "global", temperature: float = 0.07,
                 routing_estimator: str = "selected_st", diversity_w: float = 0.0,
                 diversity_probe_tokens: int = 64):
        super().__init__()
        if routing_unit not in ("image", "token"):
            raise ValueError(f"routing_unit must be image|token, got {routing_unit!r}")
        if balance == "within_batch":
            balance = "within_environment"
        if balance not in ("global", "within_environment"):
            raise ValueError(f"balance must be global|within_environment, got {balance!r}")
        if routing_estimator not in ("selected_st", "legacy_renorm"):
            raise ValueError(
                f"routing_estimator must be selected_st|legacy_renorm, got {routing_estimator!r}")
        if int(expert_rank) < 1:
            raise ValueError("expert_rank must be >= 1; use SharedResidualMoEFFN for full width")
        if not 1 <= int(top_k) <= int(n_experts):
            raise ValueError("low-rank residual MoE requires 1 <= top_k <= n_experts")

        d_in, _, d_out = _mlp_shape(mlp)
        self.n_experts = int(n_experts)
        self.target_top_k = int(top_k)
        self.top_k = int(top_k)
        self.expert_rank = int(expert_rank)
        self.routing_unit, self.balance = routing_unit, balance
        self.routing_estimator = str(routing_estimator)
        self.diversity_w = float(diversity_w)
        self.probe_tokens = int(diversity_probe_tokens)
        self.d_out = int(d_out)

        self.shared = copy.deepcopy(mlp)
        self.experts = nn.ModuleList(
            [_LowRankExpert(d_in, self.expert_rank, out_dim=d_out)
             for _ in range(self.n_experts)])
        self.router = Router(d_in, self.n_experts, geometry, temperature)

    def set_top_k(self, top_k: int):
        """Set the active top-k (dense-to-sparse annealing). Clamped to ``[1, n_experts]``."""
        self.top_k = max(1, min(int(top_k), self.n_experts))
        return self.top_k

    def forward(self, x):
        tokens, orig = _as_tokens(x)
        B, T, C = tokens.shape
        shared_out = self.shared(_restore(tokens, orig))

        if self.routing_unit == "image":
            logits = self.router(tokens.mean(dim=1))
            env_dec = self._env
        else:
            logits = self.router(tokens.reshape(B * T, C))
            env_dec = None if self._env is None else self._env.repeat_interleave(T)

        probs = logits.softmax(dim=-1)
        top_k = self.top_k
        topv, topi = probs.topk(top_k, dim=-1)
        if top_k == 1 and self.routing_estimator == "selected_st":
            # Exactly one in the forward pass (preserving the zero-initialised residual) while
            # still carrying d(topv)/d(logits) in the backward pass.
            topv = topv + (torch.ones_like(topv) - topv).detach()
        else:
            topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        self.last = {"logits": logits, "probs": probs, "assign": topi[:, 0].detach(),
                     "env": env_dec, "tokens_per_image": T,
                     "routing_estimator": self.routing_estimator, "active_top_k": top_k}

        flat = tokens.reshape(B * T, C)
        if self.routing_unit == "image":
            idx = topi.repeat_interleave(T, dim=0)
            wts = topv.repeat_interleave(T, dim=0)
        else:
            idx, wts = topi, topv

        correction = torch.zeros(B * T, self.d_out, device=flat.device, dtype=flat.dtype)
        for slot in range(top_k):
            ids, weights = idx[:, slot], wts[:, slot]
            contribution = torch.zeros_like(correction)
            for index, expert in enumerate(self.experts):
                mask = ids == index
                if mask.any():
                    contribution[mask] = expert(flat[mask]).to(contribution.dtype)
            correction = correction + weights.unsqueeze(-1) * contribution

        self._record_probe(flat)
        return shared_out + _restore(correction.reshape(B, T, self.d_out), orig)

    def aux_loss(self, balance_w: float, zloss_w: float = 0.0):
        if self.last is None:
            return torch.zeros((), device=self.router.weight.device)
        total = self._balance_terms(balance_w, zloss_w)
        if self.diversity_w > 0:
            total = total + self.diversity_w * self.expert_diversity_loss()
        return total


FRONTIER_TYPES = (
    SharedRoutedOracleFFN,
    CondLNMoEFFN,
    SoftMoEResidualFFN,
    LowRankResidualMoEFFN,
)

#: Variants whose routing has no trainable ``Router`` module.
ROUTERLESS_TYPES = (SharedRoutedOracleFFN,)
