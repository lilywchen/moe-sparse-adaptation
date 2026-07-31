"""Sample-level Mixture-of-Experts block: ONE routing decision per image (clean audit).

Router options:
  - 'linear'  : standard dot-product gate (Wx). Routes by whatever direction is largest →
                under a confound that's the batch (the liability we measured).
  - 'cosine'  : the GMoE *router component only* — ℓ2-normalize the pooled input and a learnable
                expert codebook, route by cosine similarity / temperature. Removing magnitude
                (where an affine scale lives) is meant to push routing toward content.

  CAVEAT — this is NOT a fair GMoE test. GMoE's mechanism is *part-by-part*: routing is per-PATCH,
  so experts specialize on local visual attributes. Here routing is sample-level (whole image ->
  ONE decision), so the cosine router is only a cheap probe of "does normalization help at the
  image level". A faithful GMoE test additionally requires patch/token-level routing (the planned
  spatial-token block). A null result with cosine here does NOT refute GMoE.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEBlock(nn.Module):
    def __init__(self, base_block: nn.Module, N: int = 8, k: int = 2,
                 router: str = "linear", router_dim: int = None, router_temp: float = 0.07):
        super().__init__()
        self.N, self.k = N, k
        self.experts = nn.ModuleList([copy.deepcopy(base_block) for _ in range(N)])
        in_c = base_block.conv1.in_channels
        self.router_type = router
        if router == "linear":
            self.router = nn.Linear(in_c, N)
        elif router == "cosine":
            d = router_dim or in_c
            self.proj = nn.Linear(in_c, d, bias=False)            # x -> attribute space
            self.codebook = nn.Parameter(torch.randn(N, d) * 0.1)  # learnable attribute codebook E
            self.log_temp = nn.Parameter(torch.tensor(math.log(router_temp)))  # learnable temperature
        else:
            raise ValueError(f"unknown router: {router}")
        self.last = None

    def _route_logits(self, pooled):
        if self.router_type == "linear":
            return self.router(pooled)                            # [B, N]
        z = F.normalize(self.proj(pooled), dim=-1)                # ℓ2-normalized input
        e = F.normalize(self.codebook, dim=-1)                    # ℓ2-normalized codebook
        return (z @ e.t()) / self.log_temp.exp().clamp(min=1e-2)  # cosine sim / temperature

    def forward(self, x):
        B = x.shape[0]
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1).float()   # [B, C]
        logits = self._route_logits(pooled)                       # [B, N]
        gates = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(gates, self.k, dim=-1)            # [B, k]
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)     # renormalize top-k
        self.last = {"logits": logits, "gates": gates, "idx": topi}

        out = None
        for slot in range(self.k):
            idx_slot, w_slot = topi[:, slot], topv[:, slot]
            res = None
            for e in range(self.N):
                mask = idx_slot == e
                if mask.any():
                    ye = self.experts[e](x[mask])
                    if res is None:
                        res = x.new_zeros((B,) + tuple(ye.shape[1:]))
                    res[mask] = ye.to(res.dtype)
            if res is None:
                continue
            contrib = w_slot.view(-1, 1, 1, 1).to(res.dtype) * res
            out = contrib if out is None else out + contrib
        return out

    def aux_loss(self, zloss_w: float, balance_w: float) -> torch.Tensor:
        logits, gates, idx = self.last["logits"], self.last["gates"], self.last["idx"]
        B = logits.shape[0]
        z = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        dispatch = torch.zeros(self.N, device=logits.device)
        for slot in range(self.k):
            dispatch.scatter_add_(0, idx[:, slot], torch.ones(B, device=logits.device))
        frac = dispatch / (B * self.k)
        meanprob = gates.mean(dim=0)
        balance = self.N * (frac * meanprob).sum()
        return zloss_w * z + balance_w * balance

    def top1(self) -> torch.Tensor:
        return self.last["idx"][:, 0].detach().cpu()
