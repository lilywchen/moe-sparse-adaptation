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
    """Sparse conditional capacity: E experts, top-1, each initialised from the pretrained FFN.

    routing_unit : "image" -> one decision per image (routes on pooled/global appearance)
                   "token" -> one decision per token (routes on local content)
    router_frozen: True    -> router parameters are fixed at init (the frozen-router control that
                             isolates whether the LEARNED assignment policy adds value)
    balance      : "global" | "within_environment"
    """

    def __init__(self, mlp, n_experts=8, top_k=1, routing_unit="token", geometry="cosine",
                 balance="global", temperature=0.07, router_frozen=False, sym_break=0.0):
        super().__init__()
        if routing_unit not in ("image", "token"):
            raise ValueError(f"routing_unit must be image|token, got {routing_unit!r}")
        if balance == "within_batch":
            balance = "within_environment"
        if balance not in ("global", "within_environment"):
            raise ValueError(f"balance must be global|within_environment, got {balance!r}")
        fc1, _, _ = _mlp_parts(mlp)
        self.n_experts, self.top_k = n_experts, top_k
        self.routing_unit, self.balance = routing_unit, balance
        self.router_frozen, self.sym_break = bool(router_frozen), float(sym_break)

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
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        self.last = {"logits": logits, "probs": probs, "assign": topi[:, 0].detach(),
                     "env": env_dec, "tokens_per_image": T}

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
