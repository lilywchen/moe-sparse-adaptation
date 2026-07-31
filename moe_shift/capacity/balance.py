"""Load-balancing objectives.

global_lbl      -- the canonical Switch-style auxiliary loss over the whole minibatch.
within_environment_lbl-- the SAME canonical objective computed separately inside each observed
                   acquisition environment, then averaged across environments (plan:
                   "L(within-batch) = mean over b of L_balance({soft routes i : batch(i) = b})").

Both take soft routing probabilities and the hard top-1 assignment. Neither adds inference
parameters. within-batch guarantees equal *marginal* usage per environment -- it does NOT
guarantee routes carry zero batch information; that stays an empirical outcome.
"""
import torch


def _canonical(probs: torch.Tensor, assign: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Switch aux loss: E * sum_e f_e * P_e.

    probs  : [M, E] soft router probabilities
    assign : [M]    hard top-1 expert index
    """
    if probs.numel() == 0:
        return probs.new_zeros(())
    frac = torch.zeros(n_experts, device=probs.device, dtype=probs.dtype)
    frac.scatter_add_(0, assign, torch.ones_like(assign, dtype=probs.dtype))
    frac = frac / assign.numel()                       # f_e: fraction of decisions to expert e
    mean_prob = probs.mean(dim=0)                      # P_e: mean router probability for e
    return n_experts * (frac * mean_prob).sum()


def global_lbl(probs, assign, n_experts):
    return _canonical(probs, assign, n_experts)


def within_environment_lbl(probs, assign, n_experts, env):
    """env: [M] integer environment id per routing decision. Averages the canonical loss
    computed independently within each environment present in the minibatch."""
    if env is None:
        return global_lbl(probs, assign, n_experts)
    losses = []
    for e in torch.unique(env):
        m = env == e
        if m.any():
            losses.append(_canonical(probs[m], assign[m], n_experts))
    if not losses:
        return probs.new_zeros(())
    return torch.stack(losses).mean()


# Backwards-compatible alias for results/configs created before the terminology was clarified.
# "Batch" in the paper always means an acquisition environment, never a stochastic minibatch.
within_batch_lbl = within_environment_lbl


def z_loss(logits: torch.Tensor) -> torch.Tensor:
    """Router z-loss: keeps logits from drifting large."""
    if logits.numel() == 0:
        return logits.new_zeros(())
    return (torch.logsumexp(logits, dim=-1) ** 2).mean()
