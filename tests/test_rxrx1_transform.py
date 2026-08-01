import numpy as np
import pytest
from PIL import Image

from moe_shift.data.rxrx1 import _cell_dino_cp5, _rxrx1_transform


def _nonconstant_rgb(size=32):
    grid = np.arange(size * size, dtype=np.uint16).reshape(size, size)
    image = np.stack([
        grid % 256,
        (3 * grid + 17) % 256,
        (7 * grid + 29) % 256,
    ], axis=-1).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def test_wilds_style_resizes_and_standardizes_each_channel():
    out = _rxrx1_transform(24, train=False, style="wilds")(_nonconstant_rgb())
    assert tuple(out.shape) == (3, 24, 24)
    assert np.allclose(out.mean(dim=(1, 2)).numpy(), 0.0, atol=2e-5)
    assert np.allclose(out.std(dim=(1, 2)).numpy(), 1.0, atol=2e-5)


def test_wilds_style_training_transform_is_seed_reproducible():
    import torch

    transform = _rxrx1_transform(24, train=True, style="wilds")
    torch.manual_seed(7)
    first = transform(_nonconstant_rgb())
    torch.manual_seed(7)
    second = transform(_nonconstant_rgb())
    assert torch.equal(first, second)


def test_unknown_rxrx1_transform_style_fails_loudly():
    with pytest.raises(ValueError, match="Unknown RxRx1 transform style"):
        _rxrx1_transform(24, train=False, style="not-a-style")


def test_cell_dino_layout_maps_homologous_channels_and_zero_fills_missing_stains():
    import torch

    x = torch.stack([torch.full((3, 3), 1.0), torch.full((3, 3), 2.0),
                     torch.full((3, 3), 3.0)])
    out = _cell_dino_cp5(x)
    assert tuple(out.shape) == (5, 3, 3)
    assert torch.equal(out[0], x[0])                  # nuclei -> DNA
    assert torch.equal(out[1], x[1])                  # ER -> ER
    assert torch.count_nonzero(out[2]) == 0           # missing RNA
    assert torch.equal(out[3], x[2])                  # actin -> AGP
    assert torch.count_nonzero(out[4]) == 0           # missing mitochondria


def test_cell_dino_wilds_transform_outputs_five_standardized_slots():
    out = _rxrx1_transform(
        32, train=False, style="wilds", channel_layout="cell_dino_cp5")(_nonconstant_rgb())
    assert tuple(out.shape) == (5, 32, 32)
    assert np.allclose(out[[0, 1, 3]].mean(dim=(1, 2)).numpy(), 0.0, atol=2e-5)
    assert np.allclose(out[[0, 1, 3]].std(dim=(1, 2)).numpy(), 1.0, atol=2e-5)
    assert np.count_nonzero(out[[2, 4]].numpy()) == 0


def test_cell_dino_layout_requires_wilds_normalization():
    with pytest.raises(ValueError, match="require rxrx1_transform=wilds"):
        _rxrx1_transform(32, train=False, style="imagenet", channel_layout="cell_dino_cp5")
