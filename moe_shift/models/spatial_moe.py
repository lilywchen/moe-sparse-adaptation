"""GMoE-style spatial-token MoE — the part-by-part mechanism, ported to a CNN.

The key GMoE insight: route EACH local piece independently so experts specialize on local
content *attributes* (which recur across batches), not whole-image properties (which the
global batch effect dominates). Here:

  x -> shared spatial mixing (the original residual block)        # spatial structure preserved
    -> flatten H*W feature-map locations into tokens [B*H*W, C]
    -> per-token routing (cosine, GMoE-style: ℓ2-normalized matched filter to an attribute
       codebook; or linear) -> top-k pointwise FFN experts        # channel mixing, routed
    -> residual

Pointwise (1x1 / per-location) experts are what make per-location routing well-defined: a
3x3 conv expert would need neighbours that may be routed elsewhere. The audit becomes
per-token: mi_site / mi_class are measured between each location's expert choice and the
*image's* site / label (expanded to its tokens).

NOTE: this is the faithful translation of GMoE's *architecture insight* to a CNN. It is NOT
the GMoE *platform* (ViT, ImageNet-pretrained, DomainBed). A null here is informative about
the mechanism in this controlled setting, not a refutation of GMoE at scale.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNExpert(nn.Module):
    """Pointwise FFN expert (per-location), transformer-style: Linear -> GELU -> Linear.
    fc2 is ZERO-INITIALIZED so the routed residual branch starts as a no-op (out = x + 0).
    Without this, a randomly-init FFN residual diverges under the ResNet's SGD lr=0.1 (NaN)."""
    def __init__(self, C, hidden):
        super().__init__()
        self.fc1 = nn.Linear(C, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, C)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)          # branch starts at zero -> stable, bootstraps gently

    def forward(self, x):                      # x: [T, C]
        return self.fc2(self.act(self.fc1(x)))


class SpatialMoEBlock(nn.Module):
    def __init__(self, base_block: nn.Module, N: int = 8, k: int = 2,
                 router: str = "cosine", router_temp: float = 0.07, expert_hidden: int = None):
        super().__init__()
        self.N, self.k = N, k
        self.mix = base_block                                       # shared spatial mixing (+ its residual)
        C = base_block.conv1.in_channels
        h = expert_hidden or C
        self.ln = nn.LayerNorm(C)                                   # pre-FFN norm, per token
        self.experts = nn.ModuleList([FFNExpert(C, h) for _ in range(N)])
        self.router_type = router
        if router == "linear":
            self.router = nn.Linear(C, N)
        elif router == "cosine":
            self.proj = nn.Linear(C, C, bias=False)                 # token -> attribute space
            self.codebook = nn.Parameter(torch.randn(N, C) * 0.1)   # learnable attribute codebook E
            self.log_temp = nn.Parameter(torch.tensor(math.log(router_temp)))
        else:
            raise ValueError(f"unknown router: {router}")
        self.tokens_per_image = None                                # set each forward (H*W)
        self.last = None

    def _route(self, tok):                                          # tok: [T, C] -> [T, N]
        if self.router_type == "linear":
            return self.router(tok).clamp(-20, 20)
        z = F.normalize(self.proj(tok), dim=-1)
        e = F.normalize(self.codebook, dim=-1)
        logits = (z @ e.t()) / self.log_temp.exp().clamp(min=1e-2)
        return logits.clamp(-20, 20)                                # guard softmax against overflow

    def forward(self, x):
        x = self.mix(x)                                             # [B, C, H, W]
        B, C, H, W = x.shape
        self.tokens_per_image = H * W
        tok = x.permute(0, 2, 3, 1).reshape(-1, C)                 # [T, C], T = B*H*W (image-major)
        h_in = self.ln(tok)
        logits = self._route(h_in)                                  # [T, N]
        gates = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(gates, self.k, dim=-1)              # [T, k]
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        self.last = {"logits": logits, "gates": gates, "idx": topi}

        out = torch.zeros_like(tok)
        for slot in range(self.k):
            idx_slot, w_slot = topi[:, slot], topv[:, slot]
            contrib = torch.zeros_like(tok)
            for e in range(self.N):
                m = idx_slot == e                                   # disjoint within a slot
                if m.any():
                    contrib[m] = self.experts[e](h_in[m]).to(contrib.dtype)
            out = out + w_slot.unsqueeze(-1) * contrib              # out-of-place accumulate across slots
        out = tok + out                                            # residual (transformer-style FFN)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def aux_loss(self, zloss_w: float, balance_w: float) -> torch.Tensor:
        logits, gates, idx = self.last["logits"], self.last["gates"], self.last["idx"]
        T = logits.shape[0]
        z = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        dispatch = torch.zeros(self.N, device=logits.device)
        for slot in range(self.k):
            dispatch.scatter_add_(0, idx[:, slot], torch.ones(T, device=logits.device))
        frac = dispatch / (T * self.k)
        meanprob = gates.mean(dim=0)
        balance = self.N * (frac * meanprob).sum()
        return zloss_w * z + balance_w * balance

    def top1(self) -> torch.Tensor:
        """Per-TOKEN top-1 expert id (length B*tokens_per_image, image-major)."""
        return self.last["idx"][:, 0].detach().cpu()
