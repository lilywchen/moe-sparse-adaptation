"""Site partition + per-site affine nuisance (the ComBat model) — the core apparatus.

rho is the CONFOUND: P(a sample's site is set by its label) vs assigned randomly.
  rho=0 -> site independent of label (no shortcut).
  rho=1 -> site fully determined by class (label % n_sites).
A class's "preferred" site is class % n_sites, so under rho>0 the site predicts the
label among the SEEN sites; the held-out site is unseen, so any site->label shortcut
the model learns fails there.
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader, Dataset, Subset

from .datasets import get_base_dataset, get_normalize


def assign_sites(labels, pool, rho, seed):
    """Per-sample site assignment with label-correlation rho over the given site pool."""
    labels = np.asarray(labels)
    pool = np.asarray(pool)
    K = len(pool)
    rng = np.random.default_rng(seed)
    use_pref = rng.random(len(labels)) < rho            # which samples follow the shortcut
    pref = pool[labels % K]                              # class -> preferred site
    rand = pool[rng.integers(0, K, size=len(labels))]   # else random site in pool
    return np.where(use_pref, pref, rand).astype(np.int64)


class AffineNuisance:
    """Per-site, per-channel location-scale shift (fixed once)."""
    def __init__(self, K, channels, magnitude, sigma_g, sigma_b, seed):
        rng = np.random.default_rng(seed)
        gain = 1.0 + magnitude * rng.normal(0.0, sigma_g, size=(K, channels))
        bias = magnitude * rng.normal(0.0, sigma_b, size=(K, channels))
        self.gain = torch.tensor(gain, dtype=torch.float32)
        self.bias = torch.tensor(bias, dtype=torch.float32)

    def apply(self, img, site):
        g = self.gain[site].view(-1, 1, 1)
        b = self.bias[site].view(-1, 1, 1)
        return torch.clamp(img * g + b, 0.0, 1.0)


class SpatialAffineNuisance:
    """Content-region-dependent per-site affine — the HARD nuisance.

    The per-channel gain/bias depend on each pixel's local luminance region, so the shift
    VARIES SPATIALLY within an image and correlates with content. Two consequences:
      * a single per-image standardization (InstanceNorm) CANNOT cancel it — different
        regions need different corrections, so global mean/std normalization leaves residual;
      * per-batch ComBat (one affine per batch) CANNOT either — the correction depends on
        the content, not just the site.
    This is the regime where CONDITIONAL (content-aware) correction is necessary — i.e. the
    only regime where MoE's conditional computation has a job no global fix can do.
    Region boundaries are fixed global luminance thresholds, and regions are computed from
    the CLEAN image, so the partition is content-driven (bright objects vs dark background)."""
    def __init__(self, K, channels, n_regions, magnitude, sigma_g, sigma_b, seed):
        rng = np.random.default_rng(seed)
        self.n_regions = n_regions
        self.gain = torch.tensor(
            1.0 + magnitude * rng.normal(0.0, sigma_g, size=(K, n_regions, channels)), dtype=torch.float32)
        self.bias = torch.tensor(
            magnitude * rng.normal(0.0, sigma_b, size=(K, n_regions, channels)), dtype=torch.float32)
        self.bounds = torch.linspace(0.0, 1.0, n_regions + 1)[1:-1]    # interior luminance cut points

    def apply(self, img, site):
        lum = img.mean(dim=0, keepdim=True)                            # [1,H,W] content proxy (clean image)
        region = torch.bucketize(lum, self.bounds.to(lum.device))     # [1,H,W] in 0..n_regions-1
        out = img
        for r in range(self.n_regions):
            mask = (region == r)                                       # [1,H,W], broadcasts over channels
            g = self.gain[site, r].view(-1, 1, 1)
            b = self.bias[site, r].view(-1, 1, 1)
            out = torch.where(mask, img * g + b, out)                  # disjoint regions accumulate cleanly
        return torch.clamp(out, 0.0, 1.0)


class StyleDomainNuisance:
    """Per-site STYLE shift (the asset/DG pole) — the GMoE regime, not the batch regime.

    Each site is a label-PRESERVING style (blur + contrast + gamma + colour-cast + noise); the
    held-out site is an UNSEEN style. Unlike the affine batch effect, style is not a single global
    location-scale an InstanceNorm cancels — but the CONTENT (object shape/parts) is shared and
    label-relevant across styles. So a model that routes by recurring CONTENT attributes (GMoE,
    per-patch) should generalize to a new style, where routing by the (whole-image) style would
    not. This is the regime where specialization is an ASSET — the contrast to affine_spatial."""
    def __init__(self, K, channels, magnitude, sigma_g, sigma_b, seed):
        rng = np.random.default_rng(seed)
        self.blur = np.abs(rng.normal(0.0, 0.7 * magnitude, K))               # gaussian-blur sigma >= 0
        self.contrast = 1.0 + magnitude * rng.normal(0.0, sigma_g, K)         # global contrast gain
        self.gamma = np.exp(magnitude * rng.normal(0.0, 0.3, K))              # tone curve
        self.cast = 1.0 + magnitude * rng.normal(0.0, sigma_b, (K, channels)) # per-channel colour cast
        self.noise = np.abs(rng.normal(0.0, 0.04 * magnitude, K))            # additive sensor noise

    def apply(self, img, site):
        x = img
        s = float(self.blur[site])
        if s > 1e-2:
            x = TF.gaussian_blur(x, kernel_size=3, sigma=s)
        x = (x - 0.5) * float(self.contrast[site]) + 0.5
        x = torch.clamp(x, 0.0, 1.0) ** float(self.gamma[site])
        x = x * torch.tensor(self.cast[site], dtype=x.dtype).view(-1, 1, 1)
        if self.noise[site] > 1e-3:
            x = x + torch.randn_like(x) * float(self.noise[site])
        return torch.clamp(x, 0.0, 1.0)


def make_nuisance(shift_cfg, K, channels):
    """Factory: shift.type = 'affine' (homogeneous, trivially correctable) |
    'affine_spatial' (content-region-dependent nuisance; defeats InstanceNorm + ComBat) |
    'style' (per-site label-preserving STYLE — the asset/DG pole, content shared across styles)."""
    t = shift_cfg.get("type", "affine")
    args = (shift_cfg["magnitude"], shift_cfg["sigma_g"], shift_cfg["sigma_b"], shift_cfg["seed"])
    if t == "affine":
        return AffineNuisance(K, channels, *args)
    if t in ("affine_spatial", "spatial"):
        return SpatialAffineNuisance(K, channels, shift_cfg.get("n_regions", 3), *args)
    if t in ("style", "style_domain"):
        return StyleDomainNuisance(K, channels, *args)
    raise ValueError(f"unknown shift.type: {t}")


class SiteShiftedDataset(Dataset):
    """base [0,1] img -> affine nuisance(site) -> clamp -> normalize -> (img, label, site).
    `sites` is a precomputed per-index site array; `normalize` is the per-dataset transform
    (ImageNet stats for the pretrained ViT, CIFAR stats otherwise)."""
    def __init__(self, base, sites, nuisance, normalize):
        self.base = base
        self.sites = sites
        self.nuisance = nuisance
        self.normalize = normalize

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        site = int(self.sites[i])
        img = self.normalize(self.nuisance.apply(img, site))
        return img, label, site


def _verify(sites, labels, K, tag):
    """Print the ACTUAL induced site<->label correlation so the rho axis is never silently broken."""
    mi = normalized_mutual_info_score(np.asarray(labels), np.asarray(sites))
    counts = np.bincount(sites, minlength=K)
    print(f"[sites:{tag}] site<->label NMI={mi:.4f}  per-site counts={counts.tolist()}")


def make_site_loaders(cfg):
    """Returns (train_loader, test_within, test_heldout).
      Train: seen sites {0..K-2}, label-correlated by rho.
      Test: ~1/K of images on the held-out site K-1 (unseen, label-independent),
            the rest on seen sites with the same rho correlation."""
    K = cfg["sites"]["K"]
    s_seed = cfg["sites"]["seed"]
    rho = cfg["shift"]["rho"]
    seen = list(range(K - 1))
    heldout = K - 1

    # resolution: the backbone's expected input (224 for pretrained ViT-S/16; 32 for the CIFAR track)
    img_size = cfg.get("img_size") or cfg["model"].get("vit", {}).get("img", 32)
    normalize = get_normalize(cfg["dataset"])

    train_base = get_base_dataset(cfg["dataset"], train=True, root=cfg["data_root"], img_size=img_size)
    test_base = get_base_dataset(cfg["dataset"], train=False, root=cfg["data_root"], img_size=img_size)
    y_tr = np.asarray(train_base.targets)
    y_te = np.asarray(test_base.targets)

    # train: correlation rho among the seen sites
    train_sites = assign_sites(y_tr, seen, rho, seed=s_seed)

    # test: random ~1/K to held-out site, rest to seen sites with correlation rho
    rng = np.random.default_rng(s_seed + 1)
    to_held = rng.random(len(y_te)) < (1.0 / K)
    seen_sites = assign_sites(y_te, seen, rho, seed=s_seed + 2)
    test_sites = np.where(to_held, heldout, seen_sites).astype(np.int64)

    _verify(train_sites, y_tr, K, f"train rho={rho}")          # <-- safeguard

    nz = cfg["shift"]
    nuisance = make_nuisance(nz, K, 3)            # 'affine' (default) or 'affine_spatial' (hard)
    train_ds = SiteShiftedDataset(train_base, train_sites, nuisance, normalize)
    test_ds = SiteShiftedDataset(test_base, test_sites, nuisance, normalize)

    within = Subset(test_ds, np.where(test_sites != heldout)[0].tolist())
    held = Subset(test_ds, np.where(test_sites == heldout)[0].tolist())

    # DECORRELATED AUDIT set: random (rho=0) site over seen sites, so site _|_ label.
    # Routing-MI and leakage must be measured here, NOT on `within`, or the site<->label
    # correlation at rho>0 inflates mi_site/leakage even for a pure content-router.
    audit_sites = assign_sites(y_te, seen, rho=0.0, seed=s_seed + 3)
    audit_ds = SiteShiftedDataset(test_base, audit_sites, nuisance, normalize)

    bs, nw = cfg["train"]["batch_size"], cfg["train"]["num_workers"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True, pin_memory=True),
        DataLoader(within, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
        DataLoader(held, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
        DataLoader(audit_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True),
    )
