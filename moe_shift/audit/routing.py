"""Routing audit: does the router send images to experts by their injected site?"""
import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score


@torch.no_grad()
def capture(model, loader, device, block=None):
    """Top-1 expert id and aligned site/class labels for any MoE routing unit.

    Image routing emits one assignment per image. Token routing emits one assignment per token,
    so the image-level labels must be repeated in image-major order before mutual information is
    computed. Treating token assignments as image assignments creates unequal sample counts and
    silently drops the routing diagnostics in ``run_ccas.py``.
    """
    model.eval()
    block = model.moe_block if block is None else block
    if block is None:
        raise ValueError("routing capture requires at least one MoE block")
    experts, sites, labels = [], [], []
    for x, y, site, *_ in loader:
        model(x.to(device))
        assignment = block.top1().numpy()
        site = np.asarray(site)
        label = np.asarray(y)
        if assignment.shape[0] != site.shape[0]:
            if site.shape[0] == 0 or assignment.shape[0] % site.shape[0] != 0:
                raise ValueError(
                    "routing assignments cannot be aligned to image labels: "
                    f"{assignment.shape[0]} assignments for {site.shape[0]} images")
            tokens_per_image = assignment.shape[0] // site.shape[0]
            recorded = (getattr(block, "last", None) or {}).get("tokens_per_image")
            if recorded is not None and int(recorded) != tokens_per_image:
                raise ValueError(
                    f"router recorded {recorded} tokens/image but emitted {tokens_per_image}")
            site = np.repeat(site, tokens_per_image)
            label = np.repeat(label, tokens_per_image)
        experts.append(assignment)
        sites.append(site)
        labels.append(label)
    return np.concatenate(experts), np.concatenate(sites), np.concatenate(labels)


@torch.no_grad()
def capture_spatial(model, loader, device):
    """Spatial-token MoE: per-TOKEN top-1 expert id, with the image's site/class label expanded
    to each of its tokens (token order is image-major, matching SpatialMoEBlock.forward)."""
    model.eval()
    experts, sites, labels = [], [], []
    for x, y, site, *_ in loader:
        model(x.to(device))
        tpi = model.moe_block.tokens_per_image
        experts.append(model.moe_block.top1().numpy())          # [B*tpi]
        sites.append(np.repeat(np.asarray(site), tpi))
        labels.append(np.repeat(np.asarray(y), tpi))
    return np.concatenate(experts), np.concatenate(sites), np.concatenate(labels)


def routing_mi(expert_idx, site):
    """Normalized MI in [0,1] between expert choice and site (0 = independent)."""
    if expert_idx is None:
        return None
    return float(normalized_mutual_info_score(site, expert_idx))


def expert_usage(expert_idx, N):
    """Diagnoses router collapse. Returns (n_experts_used, normalized_entropy in [0,1]).
      n_used == 1 (entropy ~0)  => COLLAPSED to one expert; routing_mi=0 is meaningless.
      n_used ~ N  (entropy ~1)  => router spreads load; routing_mi=0 then means 'not by site'.
    """
    if expert_idx is None:
        return None, None
    counts = np.bincount(expert_idx, minlength=N).astype(float)
    p = counts / counts.sum()
    n_used = int((counts > 0).sum())
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(N)) if N > 1 else 0.0
    return n_used, ent
