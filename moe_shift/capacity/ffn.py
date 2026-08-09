"""Fixed-parameter-budget FFN variants: dense-wide, learned-router MoE, frozen-router MoE.

All three are FUNCTION-PRESERVING at initialisation. The dense control uses a Net2Wider-style
mapping: hidden units are copied and each original unit's outgoing weight is divided among its
copies. Unequal (but sum-preserving) splits break replica symmetry without changing the function.

    dense-wide(x) == learned-MoE(x) == frozen-MoE(x) == original-dense(x)   at init.

    W1_wide = [W1; ... ; W1]        W2_wide = [W2, ... , W2] / E

------------------------------------------------------------------------------------------------
IMPORTANT -- dense-wide symmetry. Exact equal output splits would be a degenerate
reparameterisation. `sym_break` perturbs only the outgoing split coefficients, with coefficients
renormalized to sum to one for every source unit. The widened FFN is therefore exactly
function-preserving up to floating-point roundoff while its replicas receive unequal gradients.
------------------------------------------------------------------------------------------------
"""
import copy
from typing import Optional

import torch
import torch.nn as nn

from .balance import global_lbl, within_environment_lbl, z_loss
from .routers import Router


def _mlp_parts(mlp: nn.Module):
    """Pull (fc1, act, fc2) out of a timm Mlp (or anything with that shape)."""
    fc1, fc2 = getattr(mlp, "fc1", None), getattr(mlp, "fc2", None)
    if fc1 is None or fc2 is None:
        raise TypeError(f"expected an Mlp with .fc1/.fc2, got {type(mlp).__name__}")
    return fc1, getattr(mlp, "act", nn.GELU()), fc2


class WideFFN(nn.Module):
    """Dense-wide comparator: the SAME capacity as an E-expert MoE, allocated as shared width.

    Every input activates all E*h hidden units (no conditionality) -- this is the
    fixed-total-parameter control against which `conditional gain` is measured.
    """

    def __init__(self, mlp: nn.Module, n_experts: int = 8, sym_break: float = 0.1,
                 target_params: Optional[int] = None):
        super().__init__()
        fc1, act, fc2 = _mlp_parts(mlp)
        d_in, d_hidden, d_out = fc1.in_features, fc1.out_features, fc2.out_features
        self.n_experts, self.sym_break = n_experts, float(sym_break)
        self.act = copy.deepcopy(act)
        self.drop1 = copy.deepcopy(getattr(mlp, "drop1", nn.Identity()))
        self.norm = copy.deepcopy(getattr(mlp, "norm", nn.Identity()))
        self.drop2 = copy.deepcopy(getattr(mlp, "drop2", nn.Identity()))
        if any(p.numel() for p in self.norm.parameters()):
            raise TypeError("parameterized hidden normalization is not supported by WideFFN")

        # Pick the closest realizable hidden width to the MoE's full block budget, including its
        # router and replicated output biases. This is strictly closer than blindly using E*h.
        per_hidden = d_in + d_out + int(fc1.bias is not None)
        fixed = d_out if fc2.bias is not None else 0
        if target_params is None:
            wide_hidden = d_hidden * n_experts
        else:
            wide_hidden = max(d_hidden, round((int(target_params) - fixed) / per_hidden))
        self.wide_hidden = int(wide_hidden)

        self.fc1 = nn.Linear(d_in, self.wide_hidden, bias=fc1.bias is not None)
        self.fc2 = nn.Linear(self.wide_hidden, d_out, bias=fc2.bias is not None)
        with torch.no_grad():
            # Map each new unit to an original unit. Counts differ by at most one when the target
            # parameter budget is not an exact multiple of the pretrained hidden width.
            source = torch.arange(self.wide_hidden, device=fc1.weight.device) % d_hidden
            self.fc1.weight.copy_(fc1.weight[source])
            if fc1.bias is not None:
                self.fc1.bias.copy_(fc1.bias[source])

            out = torch.empty(d_out, self.wide_hidden, device=fc2.weight.device,
                              dtype=fc2.weight.dtype)
            for j in range(d_hidden):
                idx = torch.nonzero(source == j, as_tuple=False).flatten()
                # Deterministic unequal coefficients. Centering then normalizing makes their sum
                # exactly one in the working dtype, preserving the source unit's contribution.
                coeff = torch.ones(len(idx), device=out.device, dtype=out.dtype)
                if self.sym_break > 0 and len(idx) > 1:
                    coeff = coeff + self.sym_break * torch.linspace(
                        -1.0, 1.0, len(idx), device=out.device, dtype=out.dtype)
                coeff = coeff / coeff.sum()
                out[:, idx] = fc2.weight[:, j:j + 1] * coeff.unsqueeze(0)
            self.fc2.weight.copy_(out)
            if fc2.bias is not None:
                self.fc2.bias.copy_(fc2.bias)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.norm(x)
        return self.drop2(self.fc2(x))


class MoEFFN(nn.Module):
    """Sparse conditional capacity with experts initialised from the pretrained FFN.

    routing_unit : "image" -> one decision per image (routes on pooled/global appearance)
                   "token" -> one decision per token (routes on local content)
    router_frozen: True    -> router parameters are fixed at init (the frozen-router control that
                             isolates whether the LEARNED assignment policy adds value)
    balance      : "global" | "within_environment"
    routing_estimator:
        "selected_st"  -> top-1 is exactly one in the forward pass but carries the selected
                          softmax probability's gradient (function-preserving straight-through)
        "legacy_renorm" -> historical top-k renormalisation; retained only to reproduce old runs
    """

    def __init__(self, mlp, n_experts=8, top_k=1, routing_unit="token", geometry="cosine",
                 balance="global", temperature=0.07, router_frozen=False, sym_break=0.0,
                 routing_estimator="selected_st"):
        super().__init__()
        if routing_unit not in ("image", "token"):
            raise ValueError(f"routing_unit must be image|token, got {routing_unit!r}")
        if balance == "within_batch":
            balance = "within_environment"
        if balance not in ("global", "within_environment"):
            raise ValueError(f"balance must be global|within_environment, got {balance!r}")
        if routing_estimator not in ("selected_st", "legacy_renorm"):
            raise ValueError(
                "routing_estimator must be selected_st|legacy_renorm, "
                f"got {routing_estimator!r}")
        fc1, _, _ = _mlp_parts(mlp)
        self.n_experts, self.top_k = n_experts, top_k
        self.routing_unit, self.balance = routing_unit, balance
        self.router_frozen, self.sym_break = bool(router_frozen), float(sym_break)
        self.routing_estimator = str(routing_estimator)

        self.experts = nn.ModuleList([copy.deepcopy(mlp) for _ in range(n_experts)])
        if self.sym_break > 0:
            with torch.no_grad():
                for e in self.experts:
                    w = e.fc1.weight
                    w.add_(torch.randn_like(w) * (w.std() * self.sym_break))

        self.router = Router(fc1.in_features, n_experts, geometry, temperature)
        if self.router_frozen:
            for p in self.router.parameters():
                p.requires_grad_(False)

        self._env = None       # per-image environment ids, set via set_env()
        self.last = None       # routing stats for aux loss + audit

    def set_env(self, env):
        """env: LongTensor [B] of acquisition-environment ids for the current batch."""
        self._env = env

    def forward(self, x):
        orig = x.shape
        if x.ndim == 4:                                  # [B,H,W,C] -> tokens
            x = x.reshape(orig[0], orig[1] * orig[2], orig[3])
        B, T, C = x.shape

        if self.routing_unit == "image":
            logits = self.router(x.mean(dim=1))          # [B, E]
            env_dec = self._env
        else:
            logits = self.router(x.reshape(B * T, C))    # [B*T, E]
            env_dec = None if self._env is None else self._env.repeat_interleave(T)

        probs = logits.softmax(dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)
        if self.top_k == 1 and self.routing_estimator == "selected_st":
            # The historical ``topv / topv.sum()`` makes a top-1 gate identically one, so the
            # task loss cannot train the router through the hard expert choice.  This estimator
            # remains exactly one in the forward pass (and therefore preserves the pretrained
            # FFN at upcycling) while using d(topv)/d(logits) in the backward pass.
            topv = topv + (torch.ones_like(topv) - topv).detach()
        else:
            topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        self.last = {"logits": logits, "probs": probs, "assign": topi[:, 0].detach(),
                     "env": env_dec, "tokens_per_image": T,
                     "routing_estimator": self.routing_estimator}

        flat = x.reshape(B * T, C)
        if self.routing_unit == "image":                 # expand per-image choice to its tokens
            idx = topi.repeat_interleave(T, dim=0)
            wts = topv.repeat_interleave(T, dim=0)
        else:
            idx, wts = topi, topv

        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            ids, w = idx[:, slot], wts[:, slot]
            contrib = torch.zeros_like(flat)
            for e in range(self.n_experts):
                m = ids == e
                if m.any():
                    contrib[m] = self.experts[e](flat[m]).to(contrib.dtype)
            out = out + w.unsqueeze(-1) * contrib
        out = out.reshape(B, T, C)
        return out.reshape(orig) if len(orig) == 4 else out

    def aux_loss(self, balance_w: float, zloss_w: float = 0.0):
        if self.last is None:
            return torch.zeros((), device=next(self.parameters()).device)
        probs, assign, env = self.last["probs"], self.last["assign"], self.last["env"]
        if self.balance == "within_environment":
            lb = within_environment_lbl(probs, assign, self.n_experts, env)
        else:
            lb = global_lbl(probs, assign, self.n_experts)
        return balance_w * lb + zloss_w * z_loss(self.last["logits"])

    def top1(self):
        return self.last["assign"].detach().cpu()


def mixstyle_tokens(x: torch.Tensor, probability: float = 0.5,
                    alpha: float = 0.1, eps: float = 1e-6) -> torch.Tensor:
    """Standard MixStyle augmentation for transformer tokens.

    Per-image feature means and standard deviations are mixed across the minibatch while the
    normalized token content is retained.  It is deliberately label- and environment-agnostic,
    and is active only during training through :class:`SharedResidualMoEFFN`.
    """
    if x.ndim != 3:
        raise ValueError(f"MixStyle expects BxTxC tokens, got {tuple(x.shape)}")
    if x.shape[0] < 2 or probability <= 0 or float(torch.rand(())) >= probability:
        return x
    if alpha <= 0:
        raise ValueError("MixStyle alpha must be positive")

    mu = x.mean(dim=1, keepdim=True)
    sigma = (x.var(dim=1, keepdim=True, unbiased=False) + eps).sqrt()
    normalized = (x - mu) / sigma
    permutation = torch.randperm(x.shape[0], device=x.device)
    concentration = x.new_full((x.shape[0],), float(alpha))
    lam = torch.distributions.Beta(concentration, concentration).sample().view(-1, 1, 1)
    mixed_mu = lam * mu + (1.0 - lam) * mu[permutation]
    mixed_sigma = lam * sigma + (1.0 - lam) * sigma[permutation]
    return normalized * mixed_sigma + mixed_mu


class SharedResidualMoEFFN(nn.Module):
    """Always-active pretrained FFN plus conventionally routed residual experts.

    This is the shared-expert/residual-MoE alternative to replacement upcycling.  ``shared`` is
    the original pretrained FFN.  Each routed expert copies its input projection but starts with
    an exactly-zero output projection, so the complete module equals the pretrained FFN at
    initialization.  ``n_experts`` counts routed residual experts; the total expert banks are
    therefore ``1 + n_experts``.

    The routing implementation intentionally matches :class:`MoEFFN`: cosine or linear router,
    token or image decisions, ordinary top-k dispatch, and the same balancing/z losses.  Optional
    MixStyle is a standard feature-statistics augmentation and does not change inference.
    """

    def __init__(self, mlp, n_experts=3, top_k=1, routing_unit="token", geometry="cosine",
                 balance="global", temperature=0.07, routing_estimator="selected_st",
                 router_frozen=False,
                 feature_stat_mix_prob=0.0, feature_stat_mix_alpha=0.1):
        super().__init__()
        if routing_unit not in ("image", "token"):
            raise ValueError(f"routing_unit must be image|token, got {routing_unit!r}")
        if balance == "within_batch":
            balance = "within_environment"
        if balance not in ("global", "within_environment"):
            raise ValueError(f"balance must be global|within_environment, got {balance!r}")
        if routing_estimator not in ("selected_st", "legacy_renorm"):
            raise ValueError(
                "routing_estimator must be selected_st|legacy_renorm, "
                f"got {routing_estimator!r}")
        if int(n_experts) < 1 or not 1 <= int(top_k) <= int(n_experts):
            raise ValueError("shared residual MoE requires 1 <= top_k <= n_experts")

        fc1, _, _ = _mlp_parts(mlp)
        self.n_experts, self.top_k = int(n_experts), int(top_k)
        self.routing_unit, self.balance = routing_unit, balance
        self.routing_estimator = str(routing_estimator)
        self.feature_stat_mix_prob = float(feature_stat_mix_prob)
        self.feature_stat_mix_alpha = float(feature_stat_mix_alpha)
        self.shared = copy.deepcopy(mlp)
        self.experts = nn.ModuleList([copy.deepcopy(mlp) for _ in range(self.n_experts)])
        with torch.no_grad():
            for expert in self.experts:
                # The pretrained input projection is a useful adapter initialization.  Zeroing
                # only the final projection makes every residual exactly zero without tying later
                # updates: the initial router already sends different examples to each expert.
                expert.fc2.weight.zero_()
                if expert.fc2.bias is not None:
                    expert.fc2.bias.zero_()

        self.router = Router(fc1.in_features, self.n_experts, geometry, temperature)
        self.router_frozen = bool(router_frozen)
        if self.router_frozen:
            for parameter in self.router.parameters():
                parameter.requires_grad_(False)
        # Inference-only mechanism switch.  The routed correction can now be removed without
        # retraining, which distinguishes "the shared path learned everything" from a residual
        # branch that actually contributes to OOD validation accuracy.
        self.shared_only = False
        self._env = None
        self.last = None

    def set_env(self, env):
        self._env = env

    def forward(self, x):
        orig = x.shape
        if x.ndim == 4:
            tokens = x.reshape(orig[0], orig[1] * orig[2], orig[3])
        elif x.ndim == 3:
            tokens = x
        else:
            raise ValueError(f"shared residual MoE expects BxTxC or BxHxWxC, got {tuple(x.shape)}")

        if self.training and self.feature_stat_mix_prob > 0:
            tokens = mixstyle_tokens(
                tokens, probability=self.feature_stat_mix_prob,
                alpha=self.feature_stat_mix_alpha)
        shared_input = tokens.reshape(orig) if len(orig) == 4 else tokens
        shared_out = self.shared(shared_input)

        if self.shared_only:
            self.last = {
                "logits": None, "probs": None, "assign": None,
                "env": self._env, "tokens_per_image": tokens.shape[1],
                "routing_estimator": self.routing_estimator, "shared_only": True,
            }
            return shared_out

        B, T, C = tokens.shape
        if self.routing_unit == "image":
            logits = self.router(tokens.mean(dim=1))
            env_dec = self._env
        else:
            logits = self.router(tokens.reshape(B * T, C))
            env_dec = None if self._env is None else self._env.repeat_interleave(T)

        probs = logits.softmax(dim=-1)
        topv, topi = probs.topk(self.top_k, dim=-1)
        if self.top_k == 1 and self.routing_estimator == "selected_st":
            topv = topv + (torch.ones_like(topv) - topv).detach()
        else:
            topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        self.last = {
            "logits": logits, "probs": probs, "assign": topi[:, 0].detach(),
            "env": env_dec, "tokens_per_image": T,
            "routing_estimator": self.routing_estimator,
        }

        flat = tokens.reshape(B * T, C)
        if self.routing_unit == "image":
            idx = topi.repeat_interleave(T, dim=0)
            wts = topv.repeat_interleave(T, dim=0)
        else:
            idx, wts = topi, topv

        correction = torch.zeros_like(flat)
        for slot in range(self.top_k):
            ids, weights = idx[:, slot], wts[:, slot]
            contribution = torch.zeros_like(flat)
            for expert_index, expert in enumerate(self.experts):
                mask = ids == expert_index
                if mask.any():
                    contribution[mask] = expert(flat[mask]).to(contribution.dtype)
            correction = correction + weights.unsqueeze(-1) * contribution
        correction = correction.reshape(B, T, C)
        if len(orig) == 4:
            correction = correction.reshape(orig)
        return shared_out + correction

    def aux_loss(self, balance_w: float, zloss_w: float = 0.0):
        if self.last is None:
            return torch.zeros((), device=next(self.parameters()).device)
        probs, assign, env = self.last["probs"], self.last["assign"], self.last["env"]
        if self.balance == "within_environment":
            load_balance = within_environment_lbl(probs, assign, self.n_experts, env)
        else:
            load_balance = global_lbl(probs, assign, self.n_experts)
        return balance_w * load_balance + zloss_w * z_loss(self.last["logits"])

    def top1(self):
        return self.last["assign"].detach().cpu()
