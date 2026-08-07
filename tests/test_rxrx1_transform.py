import numpy as np
import pytest
from PIL import Image

from moe_shift.data.rxrx1 import (
    _RawSiteView,
    _cell_dino_cp5,
    _native_channel_paths,
    _rxrx1_native6_to_cell_dino_cp5,
    _rxrx1_raw_transform,
    _rxrx1_transform,
)


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


def test_native_six_channel_mapping_matches_cell_painting_stains():
    import torch

    x = torch.stack([torch.full((2, 2), float(i)) for i in range(1, 7)])
    out = _rxrx1_native6_to_cell_dino_cp5(x)
    assert tuple(out.shape) == (5, 2, 2)
    assert torch.equal(out[0], x[0])              # Hoechst -> DNA
    assert torch.equal(out[1], x[1])              # ConA -> ER
    assert torch.equal(out[2], x[3])              # Syto14 -> RNA
    assert torch.equal(out[3], 0.5 * (x[2] + x[5]))  # actin + Golgi -> AGP
    assert torch.equal(out[4], x[4])              # MitoTracker -> Mito


def test_native_channel_paths_replace_composite_stem(tmp_path):
    paths = _native_channel_paths(tmp_path, "images/HEPG2-01/Plate1/B02_s1.png")
    assert paths[0] == tmp_path / "images/HEPG2-01/Plate1/B02_s1_w1.png"
    assert paths[-1] == tmp_path / "images/HEPG2-01/Plate1/B02_s1_w6.png"


def test_raw_site_view_reads_joint_six_channel_sample(tmp_path):
    import torch

    rel = "images/HEPG2-01/Plate1/B02_s1.png"
    for channel, path in enumerate(_native_channel_paths(tmp_path, rel), start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        grid = (np.arange(12 * 12).reshape(12, 12) + channel * 11) % 256
        Image.fromarray(grid.astype(np.uint8), mode="L").save(path)

    class FakeDataset:
        _input_array = np.asarray([rel])
        metadata_array = torch.tensor([[7, 1]])          # [experiment, cell_type]
        y_array = torch.tensor([3])

    class FakeSubset:
        dataset = FakeDataset()
        indices = np.asarray([0])

        def __len__(self):
            return 1

    view = _RawSiteView(
        FakeSubset(), exp_col=0, remap={7: 2}, raw_root=tmp_path,
        transform=_rxrx1_raw_transform(8, train=False, channel_layout="native6"),
        cell_col=1,
    )
    x, y, site, env, cell = view[0]
    assert tuple(x.shape) == (6, 8, 8)
    assert np.allclose(x.mean(dim=(1, 2)).numpy(), 0.0, atol=2e-5)
    assert np.allclose(x.std(dim=(1, 2)).numpy(), 1.0, atol=2e-5)
    assert (y, site, env, cell) == (3, 2, 7, 1)


def test_raw_site_view_emits_sentinel_cell_type_when_column_absent(tmp_path):
    """cell_type is optional: a WILDS build without the field must stay runnable.

    Every non-oracle arm ignores the 5th element, so a -1 sentinel is correct here. The oracle arm
    fails loudly later, in run_ccas.batch_group_ids, where the requirement is explicit.
    """
    import torch

    rel = "images/HEPG2-01/Plate1/B02_s1.png"
    for channel, path in enumerate(_native_channel_paths(tmp_path, rel), start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        grid = (np.arange(12 * 12).reshape(12, 12) + channel * 11) % 256
        Image.fromarray(grid.astype(np.uint8), mode="L").save(path)

    class FakeDataset:
        _input_array = np.asarray([rel])
        metadata_array = torch.tensor([[7]])
        y_array = torch.tensor([3])

    class FakeSubset:
        dataset = FakeDataset()
        indices = np.asarray([0])

        def __len__(self):
            return 1

    view = _RawSiteView(
        FakeSubset(), exp_col=0, remap={7: 2}, raw_root=tmp_path,
        transform=_rxrx1_raw_transform(8, train=False, channel_layout="native6"),
    )
    assert view[0][4] == -1
