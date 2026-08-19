"""Batch-context feature correction: an AdaBN/Harmony bridge for RxRx1.

The corrector deliberately consumes only *unlabelled* observations from the current acquisition
experiment.  Feature moments provide the AdaBN component; a soft phenotype gate and a soft
batch-context gate mix a shared dictionary of low-rank residual operators, analogous to Harmony's
reuse of soft biological clusters without assigning one private model to every batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


MODES = ("none", "center", "adabn", "lowrank", "moe_batch", "moe_dual")


class BatchFeatureCorrector(nn.Module):
    """Correct pooled image features using experiment-level unlabelled context.

    ``center`` tests an additive shift (H1), ``adabn`` a diagonal affine shift (H2),
    ``lowrank`` a single phenotype-dependent low-rank operator (H3), ``moe_batch`` a shared
    family selected by batch moments (H4), and ``moe_dual`` additionally uses soft phenotype
    memberships.  All MoE residuals are zero-output initialised, preserving the AdaBN path.
    """

    def __init__(self, dim, mode="none", n_experts=4, rank=16, hidden=128,
                 temperature=1.0, eps=1e-5):
        super().__init__()
        self.dim = int(dim)
        self.mode = str(mode)
        self.n_experts = int(n_experts)
        self.rank = int(rank)
        self.temperature = float(temperature)
        self.eps = float(eps)
        if self.mode not in MODES:
            raise ValueError(f"unknown batch corrector mode {self.mode!r}; expected one of {MODES}")
        if self.temperature <= 0:
            raise ValueError("batch corrector temperature must be positive")
        if self.mode in ("lowrank", "moe_batch", "moe_dual") and self.rank < 1:
            raise ValueError("low-rank correction requires rank >= 1")
        if self.mode == "lowrank":
            self.n_experts = 1

        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        if self.mode in ("adabn", "lowrank", "moe_batch", "moe_dual"):
            self.gamma = nn.Parameter(torch.ones(self.dim))
            self.beta = nn.Parameter(torch.zeros(self.dim))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self._environment = None
        self.last = {}

        if self.mode in ("lowrank", "moe_batch", "moe_dual"):
            scale = self.dim ** -0.5
            self.down = nn.Parameter(
                torch.randn(self.n_experts, self.rank, self.dim) * scale)
            self.up = nn.Parameter(torch.zeros(self.n_experts, self.dim, self.rank))
        else:
            self.register_parameter("down", None)
            self.register_parameter("up", None)

        if self.mode in ("moe_batch", "moe_dual"):
            hidden = max(8, int(hidden))
            self.batch_router = nn.Sequential(
                nn.Linear(2 * self.dim, hidden), nn.GELU(),
                nn.Linear(hidden, self.n_experts),
            )
        else:
            self.batch_router = None
        self.phenotype_router = (
            nn.Linear(self.dim, self.n_experts, bias=False)
            if self.mode == "moe_dual" else None)

    @property
    def active(self):
        return self.mode != "none"

    def set_environment(self, environment):
        self._environment = environment

    def _moments(self, x, environment):
        means = torch.empty_like(x)
        stds = torch.empty_like(x)
        descriptors = torch.empty(x.shape[0], 2 * x.shape[1], device=x.device, dtype=x.dtype)
        for value in torch.unique(environment):
            mask = environment == value
            group = x[mask]
            mean = group.mean(dim=0)
            # Population variance is stable for the short final batch of an experiment.
            var = (group - mean).square().mean(dim=0)
            std = (var + self.eps).sqrt()
            means[mask] = mean
            stds[mask] = std
            descriptors[mask] = torch.cat((mean, std.log()), dim=0)
        return means, stds, descriptors

    def forward(self, x):
        if self.mode == "none":
            self.last = {"mode": "none"}
            return x
        environment = self._environment
        if environment is None:
            raise RuntimeError("batch correction requires set_environment() before forward")
        environment = environment.to(x.device).flatten()
        if len(environment) != len(x):
            raise ValueError("batch environment ids and feature batch have different lengths")

        mean, std, descriptor = self._moments(x, environment)
        return self.forward_with_statistics(x, mean, std, descriptor)

    def forward_with_statistics(self, x, mean, std, descriptor=None):
        """Apply fixed support-set moments to query features.

        This is the strict support/query evaluation path: scored queries do not contribute to
        their own correction statistics. ``mean`` and ``std`` may be one vector or one per row.
        """
        if self.mode == "none":
            return x
        if mean.ndim == 1:
            mean = mean.unsqueeze(0).expand_as(x)
        if std.ndim == 1:
            std = std.unsqueeze(0).expand_as(x)
        if descriptor is None:
            descriptor = torch.cat((mean, std.clamp_min(self.eps).log()), dim=-1)
        if self.mode == "center":
            corrected = x - mean
            self.last = {"mode": self.mode, "context_n": int(len(x))}
            return corrected

        normalized = (x - mean) / std
        base = normalized * self.gamma + self.beta
        if self.mode == "adabn":
            self.last = {"mode": self.mode, "context_n": int(len(x))}
            return base

        if self.mode == "lowrank":
            weights = x.new_ones((len(x), 1))
            batch_probs = weights
            phenotype_probs = weights
        else:
            batch_logits = self.batch_router(descriptor) / self.temperature
            batch_probs = F.softmax(batch_logits, dim=-1)
            if self.phenotype_router is None:
                phenotype_probs = x.new_full(batch_probs.shape, 1.0 / self.n_experts)
                weights = batch_probs
            else:
                phenotype_probs = F.softmax(
                    self.phenotype_router(normalized) / self.temperature, dim=-1)
                # Product-of-experts: a correction must be plausible for both the batch and the
                # phenotype.  Renormalisation keeps residual scale independent of entropy.
                weights = batch_probs * phenotype_probs
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        self._last_probabilities = weights
        self._last_logits = batch_logits if self.mode in ("moe_batch", "moe_dual") else None

        hidden = F.gelu(torch.einsum("bd,krd->bkr", normalized, self.down))
        expert_delta = torch.einsum("bkr,kdr->bkd", hidden, self.up)
        delta = torch.einsum("bk,bkd->bd", weights, expert_delta)
        corrected = base + delta
        with torch.no_grad():
            entropy = -(weights.clamp_min(self.eps).log() * weights).sum(dim=-1).mean()
            self.last = {
                "mode": self.mode,
                "context_n": int(len(x)),
                "routing_entropy": float(entropy),
                "effective_experts": float(entropy.exp()),
                "correction_ratio": float(
                    delta.float().norm(dim=1).mean()
                    / base.float().norm(dim=1).mean().clamp_min(self.eps)),
                "batch_route_mean": batch_probs.mean(dim=0).detach().float().cpu().tolist(),
                "phenotype_route_mean": phenotype_probs.mean(dim=0).detach().float().cpu().tolist(),
            }
        return corrected

    def aux_loss(self, balance_w=0.0, zloss_w=0.0):
        """Encourage dictionary use without forcing every individual batch to be uniform."""
        if self.mode not in ("moe_batch", "moe_dual") or not self.last:
            return self._zero
        # ``last`` is detached for reporting, so recomputing a differentiable auxiliary here is
        # impossible.  The forward stores the live probability tensor only until this call.
        probabilities = getattr(self, "_last_probabilities", None)
        logits = getattr(self, "_last_logits", None)
        if probabilities is None:
            return self._zero
        target = probabilities.new_full((self.n_experts,), 1.0 / self.n_experts)
        balance = (probabilities.mean(dim=0) - target).square().mean()
        zloss = logits.logsumexp(dim=-1).square().mean() if logits is not None else balance * 0.0
        return float(balance_w) * balance + float(zloss_w) * zloss
