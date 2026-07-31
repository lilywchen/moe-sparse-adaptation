"""Base datasets for the shift study.

Geometric augmentation ONLY (no photometric — it would overlap the injected affine nuisance).
Items are ([0,1] CHW tensor, label); the injected nuisance is applied later, in SiteShiftedDataset,
and normalization is applied AFTER the nuisance (so the nuisance lives in pixel space, like a real
batch effect, and the backbone still sees a correctly-normalized input).

Two datasets:
  cifar100   -> 32x32, 100 classes (legacy from-scratch CNN/ViT track).
  imagenette -> 10-class ImageNet subset at higher resolution (the upcycled-ViT substrate):
                few classes => a STRONG label-shortcut is possible (log2(10) bits, vs ~2 bits at
                100 classes), and 224-res => 196 content-rich tokens for per-patch routing.

`img_size` resizes to the backbone's expected resolution (224 for pretrained ViT-S/16; 32 for the
from-scratch CIFAR track). Every returned dataset is guaranteed to expose `.targets` (a list of int
labels) so the site-assignment code can build the label<->site confound without loading pixels.
"""
import os

import numpy as np
import torchvision
import torchvision.transforms as T

# fast.ai 10-class ImageNet subsets (same layout: <dir>/{train,val}/<wnid>/*.JPEG).
#   imagenette = easy/distinct classes (pretrained ViT ~99% -> too easy, no headroom)
#   imagewoof  = fine-grained dog breeds (pretrained ViT ~88-93% -> headroom for the batch to bite)
FASTAI = {
    "imagenette": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz", "imagenette2-320"),
    "imagewoof":  ("https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz",  "imagewoof2-320"),
}

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default per-dataset normalization. The pretrained ViT expects ImageNet stats; CIFAR uses its own.
NORMALIZE = T.Normalize(CIFAR100_MEAN, CIFAR100_STD)        # back-compat alias (cifar100 default)


def get_normalize(name: str):
    if name in FASTAI:                                  # ImageNet subsets -> ImageNet stats (pretrained ViT)
        return T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return T.Normalize(CIFAR100_MEAN, CIFAR100_STD)


def _ensure_targets(ds):
    """Guarantee ds.targets (list[int]) without loading any image pixels.

    CIFAR exposes .targets already. torchvision Imagenette exposes ._samples = [(path, label), ...]
    (older builds: ._labels or .imgs). We attach a plain list so assign_sites can read labels cheaply."""
    if hasattr(ds, "targets") and ds.targets is not None:
        ds.targets = list(ds.targets)
        return ds
    for attr in ("_samples", "imgs", "samples"):
        s = getattr(ds, attr, None)
        if s is not None:
            ds.targets = [int(p[1]) for p in s]
            return ds
    lbl = getattr(ds, "_labels", None)
    if lbl is not None:
        ds.targets = [int(x) for x in lbl]
        return ds
    # last resort: materialize labels (slow; only if the torchvision build hides the table)
    ds.targets = [int(ds[i][1]) for i in range(len(ds))]
    return ds


def _cifar100(train, root, img_size):
    if train:
        tf = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    else:
        tf = []
    if img_size and img_size != 32:
        tf.append(T.Resize(img_size))                  # e.g. upcycled ViT @224 over CIFAR (debug)
    tf.append(T.ToTensor())                            # -> [0,1]; NO Normalize, NO ColorJitter
    ds = torchvision.datasets.CIFAR100(root=root, train=train, download=True, transform=T.Compose(tf))
    return _ensure_targets(ds)


def _fastai_subset(name, train, root, img_size):
    """imagenette / imagewoof via the fast.ai tarball + ImageFolder (version-independent).
    Layout after extract: <root>/<dir>/{train,val}/<wnid>/*.JPEG  -> ImageFolder gives .targets."""
    img_size = img_size or 224
    if train:
        tf = [T.RandomResizedCrop(img_size, scale=(0.35, 1.0)), T.RandomHorizontalFlip()]
    else:
        tf = [T.Resize(int(img_size * 1.14)), T.CenterCrop(img_size)]
    tf.append(T.Lambda(lambda im: im.convert("RGB")))  # a few images are grayscale
    tf.append(T.ToTensor())                            # -> [0,1] CHW; NO Normalize, NO ColorJitter
    tf = T.Compose(tf)
    split = "train" if train else "val"

    url, dirname = FASTAI[name]
    base = os.path.join(root, dirname)
    if not os.path.isdir(os.path.join(base, split)):
        from torchvision.datasets.utils import download_and_extract_archive
        download_and_extract_archive(url, download_root=root)
    ds = torchvision.datasets.ImageFolder(os.path.join(base, split), transform=tf)
    return _ensure_targets(ds)


def get_base_dataset(name: str, train: bool, root: str, img_size: int = None):
    if name == "cifar100":
        return _cifar100(train, root, img_size)
    if name in FASTAI:
        return _fastai_subset(name, train, root, img_size)
    raise ValueError(f"unsupported dataset: {name}")
