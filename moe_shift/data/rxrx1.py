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

Loader contract (matches run_experiment.py + audit/*): each batch yields (x, y, site, env).
`env` is the RAW experiment id, defined on every split, and is what per-environment accuracy
is bucketed by. `site` is the train-remapped index used by the adversary and the audits, where
`site` is a CONTIGUOUS train-experiment index in 0..K-1 (so the DANN site-adversary and
routing-MI work unchanged). OOD images get site=-1 (unseen; never audited). This function
SETS cfg["sites"]["K"] to the number of train experiments so the adversary and the
site-leakage chance level (1/K) are correct.

Data download is OFF by default (the dataset is large). Set `download: true` in the config
(or once via the WILDS CLI) and point `data_root` at the WILDS root_dir.
"""
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .datasets import IMAGENET_MEAN, IMAGENET_STD


def _cell_dino_cp5(x: torch.Tensor) -> torch.Tensor:
    """Map the three WILDS channels into Cell-DINO's five Cell-Painting slots.

    WILDS exposes nuclei, endoplasmic reticulum, and actin.  The Cell-Painting model was trained
    with DNA, ER, RNA, AGP, and mitochondria.  We therefore use the homologous slots
    ``[DNA=nuclei, ER=ER, RNA=0, AGP=actin, Mito=0]``.  With a linear patch embedding this is
    exactly equivalent to selecting the corresponding three pretrained input kernels; it adds no
    learned adapter and is held fixed for every dense/MoE comparison.
    """
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"cell_dino_cp5 expects a 3-channel CHW tensor, got {tuple(x.shape)}")
    out = x.new_zeros((5, x.shape[1], x.shape[2]))
    out[0], out[1], out[3] = x[0], x[1], x[2]
    return out


def _rxrx1_native6_to_cell_dino_cp5(x: torch.Tensor) -> torch.Tensor:
    """Map native RxRx1 stains into Cell-DINO's five Cell-Painting channels.

    Native RxRx1 stores six grayscale acquisitions in this fixed order:
    Hoechst/DNA, ConA/ER, Phalloidin/actin, Syto14/RNA,
    MitoTracker/mitochondria, and WGA/Golgi.  Cell Painting acquires actin and
    Golgi together in its AGP channel, so we average those two raw acquisitions
    before per-channel standardization.  The average and sum are equivalent
    after standardization, while the average keeps values in the input range.
    """
    if x.ndim != 3 or x.shape[0] != 6:
        raise ValueError(f"native6_to_cell_dino_cp5 expects 6-channel CHW, got {tuple(x.shape)}")
    return torch.stack((x[0], x[1], x[3], 0.5 * (x[2] + x[5]), x[4]), dim=0)


def _standardize_channels(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True)
    return (x - mean) / torch.where(std == 0.0, torch.ones_like(std), std)


def _rxrx1_raw_transform(img_size, train, channel_layout):
    """Transform an already stacked native six-channel RxRx1 tensor.

    Geometric augmentation is sampled once and applied jointly to all stains.
    There is deliberately no photometric augmentation: intensity and contrast
    are part of the acquisition-batch shift under study.
    """
    if channel_layout not in ("native6", "cell_dino_native_cp5"):
        raise ValueError(f"unknown native RxRx1 channel layout: {channel_layout!r}")

    def transform(x):
        if x.ndim != 3 or x.shape[0] != 6:
            raise ValueError(f"native RxRx1 transform expects 6-channel CHW, got {tuple(x.shape)}")
        x = x.to(dtype=torch.float32)
        if train:
            angle = (0, 90, 180, 270)[int(torch.randint(0, 4, (1,)).item())]
            if angle:
                x = TF.rotate(x, angle)
            if bool(torch.rand(()) < 0.5):
                x = TF.hflip(x)
        x = TF.resize(x, [int(img_size), int(img_size)], antialias=True)
        if channel_layout == "cell_dino_native_cp5":
            x = _rxrx1_native6_to_cell_dino_cp5(x)
        return _standardize_channels(x)

    return transform


def _native_channel_paths(raw_root, composite_relative_path):
    """Return the six official PNG paths corresponding to one WILDS composite."""
    relative = Path(str(composite_relative_path))
    stem = relative.stem
    return tuple(Path(raw_root) / relative.parent / f"{stem}_w{i}.png" for i in range(1, 7))


class _RawSiteView(Dataset):
    """Use WILDS labels/splits but read the official six grayscale acquisitions."""

    def __init__(self, subset, exp_col, remap, raw_root, transform):
        self.subset = subset
        self.exp_col = int(exp_col)
        self.remap = remap
        self.raw_root = Path(raw_root)
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        global_idx = int(self.subset.indices[i])
        dataset = self.subset.dataset
        paths = _native_channel_paths(self.raw_root, dataset._input_array[global_idx])
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing native RxRx1 channel(s): {missing[:2]}")
        channels = []
        for path in paths:
            with Image.open(path) as image:
                channels.append(torch.from_numpy(np.asarray(image, dtype=np.float32).copy()) / 255.0)
        x = torch.stack(channels, dim=0)
        if self.transform is not None:
            x = self.transform(x)
        meta = dataset.metadata_array[global_idx]
        raw = int(meta[self.exp_col])
        y = int(dataset.y_array[global_idx])
        return x, y, int(self.remap.get(raw, -1)), raw


def _rxrx1_transform(img_size, train, rrc=False, style="imagenet", channel_layout=None):
    """Geometric-only aug (flips/rotations are label-preserving for microscopy); NO photometric —
    colour/contrast jitter would imitate the very batch effect we are studying. Normalize with
    ImageNet stats because both backbones (ResNet-50, ViT-S/16) are ImageNet-pretrained.

    rrc=True swaps the plain Resize for RandomResizedCrop (random scale/crop, then flips). It's still
    purely geometric — no colour change, so batch-effect-safe — and is a strong ViT regularizer
    against the train-set memorization we see on RxRx1. Eval always uses the plain Resize (no rrc)."""
    if style == "wilds":
        # Faithful to the official WILDS RxRx1 semantics: discrete right-angle rotations,
        # horizontal flip, and per-image/per-channel standardization. DINOv2 requires a fixed
        # spatial size, so the only addition is a deterministic resize to ``img_size``.
        def standardize(x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=(1, 2))
            std = x.std(dim=(1, 2))
            std[std == 0.0] = 1.0
            return TF.normalize(x, mean, std)

        def random_right_angle(x):
            angle = (0, 90, 180, 270)[int(torch.randint(0, 4, (1,)).item())]
            return TF.rotate(x, angle) if angle else x

        tf = []
        if train:
            tf += [T.Lambda(random_right_angle), T.RandomHorizontalFlip()]
        tf += [T.Resize((img_size, img_size)), T.ToTensor(), T.Lambda(standardize)]
        if channel_layout == "cell_dino_cp5":
            tf += [T.Lambda(_cell_dino_cp5)]
        elif channel_layout not in (None, "native3"):
            raise ValueError(f"Unknown RxRx1 channel layout: {channel_layout!r}")
        return T.Compose(tf)
    if style != "imagenet":
        raise ValueError(f"Unknown RxRx1 transform style: {style!r}")
    if channel_layout not in (None, "native3"):
        raise ValueError("non-native channel layouts require rxrx1_transform=wilds")

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
    """Adapt a WILDS subset -> (x, y, site_int, env_int).

    TWO domain fields, deliberately distinct. Conflating them was a real bug: per-environment
    accuracy used to be bucketed by `site`, and on an OOD split every environment is unseen, so
    every row hashed to the single `-1` bucket. `worst_env_*` then came out identically equal to
    overall accuracy, and PLAN.md's cluster bootstrap over environments silently degenerated to a
    point estimate.

    site : train-contiguous index in 0..K-1, or -1 for an environment never seen in training.
           The DANN adversary, the routing-MI audit and the leakage chance level (1/K) all need
           this index space, so the -1 sentinel stays exactly as it was.
    env  : the RAW acquisition-environment id (experiment / hospital) from the WILDS metadata.
           Well defined on every split, and the only correct key for per-environment reporting.
    """
    def __init__(self, subset, exp_col, remap):
        self.subset = subset
        self.exp_col = exp_col
        self.remap = remap

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        x, y, meta = self.subset[i]              # WILDS: (image, label, metadata_row)
        raw = int(meta[self.exp_col])
        return x, int(y), int(self.remap.get(raw, -1)), raw


def make_rxrx1_loaders(cfg):
    """Returns (train_loader, test_within, test_heldout, audit_loader). Mutates cfg['sites']['K']."""
    from wilds import get_dataset                # imported lazily so the rest of the repo needs no wilds

    img_size = cfg.get("img_size") or cfg["model"].get("vit", {}).get("img", 256)
    ds = get_dataset(dataset="rxrx1", root_dir=cfg["data_root"],
                     download=bool(cfg.get("download", False)))
    exp_col = ds.metadata_fields.index("experiment")    # the domain column

    rrc = bool(cfg["train"].get("rand_resized_crop", False))   # ViT regularizer; default off (ResNet unchanged)
    style = str(cfg["train"].get("rxrx1_transform", "imagenet"))
    layout = cfg["train"].get("rxrx1_channel_layout")
    raw_root = cfg.get("rxrx1_raw_root")
    if raw_root:
        if style != "wilds":
            raise ValueError("native RxRx1 channels require rxrx1_transform=wilds")
        tf_tr = _rxrx1_raw_transform(img_size, True, layout)
        tf_ev = _rxrx1_raw_transform(img_size, False, layout)
        # Transforming is delegated to _RawSiteView so the WILDS RGB composites are never read.
        train_sub = ds.get_subset("train")
        within_sub = ds.get_subset("id_test")
        ood_sub = ds.get_subset("test")
    else:
        tf_tr = _rxrx1_transform(img_size, True, rrc, style, layout)
        tf_ev = _rxrx1_transform(img_size, False, False, style, layout)
        train_sub = ds.get_subset("train", transform=tf_tr)
        within_sub = ds.get_subset("id_test", transform=tf_ev)
        ood_sub = ds.get_subset("test", transform=tf_ev)

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
    view = (lambda subset, transform: _RawSiteView(
        subset, exp_col, remap, raw_root, transform)) if raw_root else (
        lambda subset, transform: _SiteView(subset, exp_col, remap))
    within = view(within_sub, tf_ev)
    return (
        mk(view(train_sub, tf_tr), True),                  # train  (seen experiments)
        mk(within, False),                                 # test_within  (seen, held-out images)
        mk(view(ood_sub, tf_ev), False),                   # test_heldout (OOD experiments)
        mk(within, False),                                 # audit: seen experiments, label⟂batch already
    )


def make_rxrx1_val_loader(cfg):
    """OOD VALIDATION loader — WILDS 'val' split (held-out experiments, DISTINCT from 'test').

    This is the honest place to do model/hyperparameter selection and early stopping: tune on
    OOD val, report on OOD test, so the test set is never used to make decisions. Returned
    separately from the 4 core loaders so the dispatcher / injected-nuisance path are unchanged.
    Returns a DataLoader, or None if the dataset has no 'val' split. Sites are remapped exactly
    like the other loaders (val experiments are unseen -> site=-1), but the 4th tuple element
    carries the RAW experiment id, which is what per-environment accuracy is bucketed by.
    """
    from wilds import get_dataset

    img_size = cfg.get("img_size") or cfg["model"].get("vit", {}).get("img", 256)
    ds = get_dataset(dataset="rxrx1", root_dir=cfg["data_root"], download=False)
    if "val" not in getattr(ds, "split_dict", {}):
        return None
    exp_col = ds.metadata_fields.index("experiment")
    style = str(cfg["train"].get("rxrx1_transform", "imagenet"))
    layout = cfg["train"].get("rxrx1_channel_layout")
    raw_root = cfg.get("rxrx1_raw_root")
    if raw_root:
        if style != "wilds":
            raise ValueError("native RxRx1 channels require rxrx1_transform=wilds")
        transform = _rxrx1_raw_transform(img_size, False, layout)
        val_sub = ds.get_subset("val")
    else:
        transform = _rxrx1_transform(img_size, False, False, style, layout)
        val_sub = ds.get_subset("val", transform=transform)
    train_exps = sorted(set(ds.get_subset("train").metadata_array[:, exp_col].tolist()))
    remap = {e: i for i, e in enumerate(train_exps)}
    bs, nw = cfg["train"]["batch_size"], cfg["train"]["num_workers"]
    print(f"[rxrx1] |ood_val|={len(val_sub)} (model/hparam selection; test untouched)")
    view = (_RawSiteView(val_sub, exp_col, remap, raw_root, transform)
            if raw_root else _SiteView(val_sub, exp_col, remap))
    return DataLoader(view, batch_size=bs, shuffle=False,
                      num_workers=nw, pin_memory=True, persistent_workers=(nw > 0))
