"""DANN-style site-adversary for batch invariance — the optional `losses.invariance_w` term.

The lever that asks: does forcing features batch-invariant close the held-out gap, and does it
saturate the affine problem? Works identically for dense and MoE (it operates on
`forward_features`, which both expose), so MoE+invariance vs dense+invariance is apples-to-apples.

A gradient-reversal layer makes the FEATURES fight site-decodability while the adversary head
learns to decode the site: the adversary trains at full strength, but the gradient flowing back
into the backbone is multiplied by -lambda, so the backbone is pushed toward site-invariance.
"""
import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd):
    return _GradReverse.apply(x, lambd)


class SiteAdversary(nn.Module):
    """Predicts site from features through a gradient-reversal layer → features become
    site-invariant. `lambd` (>=0) sets the reversal strength = the invariance pressure."""
    def __init__(self, in_dim, n_sites, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_sites))

    def forward(self, feats, lambd):
        return self.net(grad_reverse(feats.float(), lambd))


def lambda_schedule(epoch, total, target):
    """Linear warmup 0 → target over the first half of training (stabilizes the adversarial game)."""
    if target <= 0:
        return 0.0
    return float(target) * min(1.0, epoch / max(1, total // 2))
