"""Frozen-representation probes — the scIB-style batch-removal vs signal-conservation frontier.

A good batch-robust representation pushes the BATCH out (site hard to decode -> site_leakage low)
while KEEPING the biology (class easy to decode -> class_decodability high). Reporting only one is
gameable: DANN can trivially drive site_leakage to chance by destroying the features. The frontier
(class_decodability - site_leakage) is the honest summary, and it is how the FM field grades
batch correction (scIB: batch removal traded against bio conservation).
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@torch.no_grad()
def features_and_site(model, loader, device):
    model.eval()
    feats, sites = [], []
    for batch in loader:
        x, site = batch[0], batch[2]
        if len(batch) > 3 and hasattr(model, "set_batch_environment"):
            model.set_batch_environment(batch[3].to(device))
        f = model.forward_features(x.to(device))
        feats.append(f.float().cpu().numpy())
        sites.append(np.asarray(site))
    return np.concatenate(feats), np.concatenate(sites)


@torch.no_grad()
def features_site_label(model, loader, device):
    """One pass -> (features, site, label) so both probes share the SAME frozen features."""
    model.eval()
    feats, sites, labels = [], [], []
    for batch in loader:
        x, y, site = batch[0], batch[1], batch[2]
        if len(batch) > 3 and hasattr(model, "set_batch_environment"):
            model.set_batch_environment(batch[3].to(device))
        f = model.forward_features(x.to(device))
        feats.append(f.float().cpu().numpy())
        sites.append(np.asarray(site))
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(sites), np.concatenate(labels)


def _probe(feats, target):
    """Linear-probe accuracy predicting `target` from frozen features (held-out 30% split)."""
    if len(np.unique(target)) < 2:
        return float("nan")
    if not np.isfinite(feats).all():       # a diverged (NaN) run: record nan, don't crash the sweep
        return float("nan")
    Xtr, Xte, ytr, yte = train_test_split(
        feats, target, test_size=0.3, random_state=0, stratify=target)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def site_leakage(feats, site):
    """Linear-probe accuracy predicting SITE from features (chance = 1/K). Lower = more batch-robust."""
    return _probe(feats, site)


def class_decodability(feats, label):
    """Linear-probe accuracy predicting CLASS from features (chance = 1/num_classes). Higher = signal kept."""
    return _probe(feats, label)
