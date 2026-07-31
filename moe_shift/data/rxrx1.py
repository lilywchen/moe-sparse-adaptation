"""RxRx1 (WILDS) loaders — REAL cellular-microscopy batch effects.

Swaps the injected-nuisance apparatus for a real dataset whose NUISANCE is the experimental
batch and whose SIGNAL is the siRNA perturbation (1,139 classes). Because every experiment
runs the full siRNA library, label ⟂ batch BY DESIGN (rho≈0 for free): there is no
batch→label shortcut, so a model that still routes by batch is purely wasting capacity —
exactly the liability this project studies, now in its native domain.

WILDS 'official' split (domain = experiment, 51 experiments across 4 cell types):
    train     -> seen experiments (the training batches)
    id_test   -> held-out IMAGES from the SAME seen experiments   -> test_within + audit
    test      -> OOD experiments (UNSEEN batches)                 -> test_heldout (the headline)

Loader contract (matches run_experiment.py + audit/*): each batch yields (x, y, site) where
`site` is a CONTIGUOUS train-experiment index in 0..K-1 (so the DANN site-adversary and
routing-MI work unchanged). OOD images get site=-1 (unseen; never audited). This function
SETS cfg["sites"]["K"] to the number of train experiments so the adversary and the
site-leakage chance level (1/K) are correct.

Data download is OFF by default (the dataset is large). Set `download: true` in the config
(or once via the WILDS CLI) and point `data_root` at the WILDS root_dir.
"""
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from .datasets import IMAGENET_MEAN, IMAGENET_STD


def _rxrx1_transform(img_size, train, rrc=False):
    """Geometric-only aug (flips/rotations are label-preserving for microscopy); NO photometric —
    colour/contrast jitter would imitate the very batch effect we are studying. Normalize with
    ImageNet stats because both backbones (ResNet-50, ViT-S/16) are ImageNet-pretrained.

    rrc=True swaps the plain Resize for RandomResizedCrop (random scale/crop, then flips). It's still
    purely geometric — no colour change, so batch-effect-safe — and is a strong ViT regularizer
    against the train-set memorization we see on RxRx1. Eval always uses the plain Resize (no rrc)."""
    if train and rrc:
        tf = [T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
              T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    else:
        tf = [T.Resize((img_size, img_size))]
        if train:
            tf += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    tf += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(tf)


class _SiteView(Dataset):
    """Adapt a WILDS subset -> (x, y, site_int), site = remapped experiment index (or -1 if unseen)."""
    def __init__(self, subset, exp_col, remap):
        self.subset = subset
        self.exp_col = exp_col
        self.remap = remap

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y, meta = self.subset[i]              # WILDS: (image, label, metadata_row)
        raw = int(meta[self.exp_col])
        return x, int(y), int(self.remap.get(raw, -1))


def make_rxrx1_loaders(cfg):
    """Returns (train_loader, test_within, test_heldout, audit_loader). Mutates cfg['sites']['K']."""
    from wilds import get_dataset                # imported lazily so the rest of the repo needs no wilds

    img_size = cfg.get("img_size") or cfg["model"].get("vit", {}).get("img", 256)
    ds = get_dataset(dataset="rxrx1", root_dir=cfg["data_root"],
                     download=bool(cfg.get("download", False)))
    exp_col = ds.metadata_fields.index("experiment")    # the domain column

    rrc = bool(cfg["train"].get("rand_resized_crop", False))   # ViT regularizer; default off (ResNet unchanged)
    tf_tr, tf_ev = _rxrx1_transform(img_size, True, rrc), _rxrx1_transform(img_size, False)
    train_sub  = ds.get_subset("train",   transform=tf_tr)
    within_sub = ds.get_subset("id_test", transform=tf_ev)    # seen experiments, held-out images
    ood_sub    = ds.get_subset("test",    transform=tf_ev)    # OOD experiments (unseen batches)

    # contiguous remap of the TRAIN experiments -> 0..K-1 (vectorized over metadata, no pixel reads)
    train_exps = sorted(set(train_sub.metadata_array[:, exp_col].tolist()))
    remap = {e: i for i, e in enumerate(train_exps)}
    K = len(train_exps)
    cfg.setdefault("sites", {})["K"] = K          # adversary n_sites + site-leakage chance (1/K)
    print(f"[rxrx1] {K} train experiments (sites); "
          f"|train|={len(train_sub)} |id_test|={len(within_sub)} |ood_test|={len(ood_sub)}")

    bs, nw = cfg["train"]["batch_size"], cfg["train"]["num_workers"]
    mk = lambda d, sh: DataLoader(d, batch_size=bs, shuffle=sh, num_workers=nw,
                                  pin_memory=True, drop_last=sh, persistent_workers=(nw > 0))
    within = _SiteView(within_sub, exp_col, remap)
    return (
        mk(_SiteView(train_sub, exp_col, remap), True),    # train  (seen experiments)
        mk(within, False),                                 # test_within  (seen, held-out images)
        mk(_SiteView(ood_sub, exp_col, remap), False),     # test_heldout (OOD experiments)
        mk(within, False),                                 # audit: seen experiments, label⟂batch already
    )


def make_rxrx1_val_loader(cfg):
    """OOD VALIDATION loader — WILDS 'val' split (held-out experiments, DISTINCT from 'test').

    This is the honest place to do model/hyperparameter selection and early stopping: tune on
    OOD val, report on OOD test, so the test set is never used to make decisions. Returned
    separately from the 4 core loaders so the dispatcher / injected-nuisance path are unchanged.
    Returns a DataLoader, or None if the dataset has no 'val' split. Sites are remapped exactly
    like the other loaders (val experiments are unseen -> site=-1; accuracy ignores site anyway).
    """
    from wilds import get_dataset

    img_size = cfg.get("img_size") or cfg["model"].get("vit", {}).get("img", 256)
    ds = get_dataset(dataset="rxrx1", root_dir=cfg["data_root"], download=False)
    if "val" not in getattr(ds, "split_dict", {}):
        return None
    exp_col = ds.metadata_fields.index("experiment")
    val_sub = ds.get_subset("val", transform=_rxrx1_transform(img_size, False))
    train_exps = sorted(set(ds.get_subset("train").metadata_array[:, exp_col].tolist()))
    remap = {e: i for i, e in enumerate(train_exps)}
    bs, nw = cfg["train"]["batch_size"], cfg["train"]["num_workers"]
    print(f"[rxrx1] |ood_val|={len(val_sub)} (model/hparam selection; test untouched)")
    return DataLoader(_SiteView(val_sub, exp_col, remap), batch_size=bs, shuffle=False,
                      num_workers=nw, pin_memory=True, persistent_workers=(nw > 0))
