import torch
import torch.nn as nn

from moe_shift.capacity import MoEFFN
from moe_shift.capacity.model import CCASModel


class _Mlp(nn.Module):
    def __init__(self, d=32, h=64):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(d, h), nn.GELU(), nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _Mlp()

    def forward(self, x):
        return x + self.mlp(x)


class _Chunk(nn.ModuleList):
    def forward(self, x):
        for block in self:
            x = block(x)
        return x


class _FakeCellDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_features = 32
        actual = [_Block() for _ in range(8)]
        self.blocks = nn.ModuleList([
            _Chunk(actual[:2]),
            _Chunk([nn.Identity(), nn.Identity(), *actual[2:4]]),
            _Chunk([*[nn.Identity() for _ in range(4)], *actual[4:6]]),
            _Chunk([*[nn.Identity() for _ in range(6)], *actual[6:8]]),
        ])

    def forward_features(self, x):
        tokens = x.mean(dim=(2, 3)).mean(dim=1, keepdim=True).unsqueeze(-1).expand(-1, 5, 32)
        for chunk in self.blocks:
            tokens = chunk(tokens)
        return {
            "x_norm_clstoken": tokens[:, 0],
            "x_norm_patchtokens": tokens[:, 1:],
        }


class _FakeChannelAdaptiveDino(_FakeCellDino):
    def get_intermediate_layers(self, x, n=1):
        assert n == 1
        batch, channels, height, width = x.shape
        raw = self.forward_features(x.reshape(batch * channels, 1, height, width))
        patches = raw["x_norm_patchtokens"].reshape(batch, channels, -1, self.num_features)
        cls = raw["x_norm_clstoken"].reshape(batch, channels * self.num_features)
        return ((patches, cls),)


def test_cell_dino_adapter_flattens_chunks_and_extracts_cls(monkeypatch, tmp_path):
    checkpoint = tmp_path / "cell_dino.pth"
    checkpoint.write_bytes(b"fake checkpoint for provenance test")

    def fake_load(repo, model, **kwargs):
        assert repo == str(tmp_path.resolve())
        assert model == "cell_dino_cp_vits8"
        assert kwargs["pretrained_path"] == str(checkpoint.resolve())
        assert kwargs["in_channels"] == 5
        return _FakeCellDino()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    model = CCASModel(
        num_classes=11, variant="moe", backbone_source="cell_dino",
        hub_repo_dir=tmp_path, checkpoint_path=checkpoint, img=128,
        placement="middle", n_experts=4,
    )
    assert len(model.blocks) == 8
    assert isinstance(model.blocks[model.capacity.block_index].mlp, MoEFFN)
    assert model.backbone_provenance["checkpoint_filename"] == checkpoint.name
    assert len(model.backbone_provenance["checkpoint_sha256"]) == 64
    out = model(torch.randn(2, 5, 16, 16))
    assert tuple(out.shape) == (2, 11)


def test_frozen_backbone_leaves_only_classifier_trainable(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _FakeCellDino())
    model = CCASModel(
        num_classes=11, variant="original", backbone_source="cell_dino",
        hub_repo_dir=tmp_path, pretrained=False, img=128, freeze_backbone=True,
    )
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())


def test_partial_adaptation_unfreezes_only_last_blocks_and_norm(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _FakeCellDino())
    model = CCASModel(
        num_classes=11, variant="original", backbone_source="cell_dino",
        hub_repo_dir=tmp_path, pretrained=False, img=128, unfreeze_last_n_blocks=2,
    )
    assert all(not p.requires_grad for block in model.blocks[:-2] for p in block.parameters())
    assert all(p.requires_grad for block in model.blocks[-2:] for p in block.parameters())
    assert all(p.requires_grad for p in model.fc.parameters())
    assert model.backbone_provenance["adaptation"] == "last_2_blocks_plus_norm"


def test_cell_dino_official_feature_pool_concatenates_cls_and_patch_mean(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: _FakeCellDino())
    model = CCASModel(
        num_classes=11, variant="original", backbone_source="cell_dino",
        hub_repo_dir=tmp_path, pretrained=False, img=128,
        feature_pool="cls_patch_mean",
    )
    x = torch.randn(2, 5, 16, 16)
    raw = model.backbone.forward_features(x)
    expected = torch.cat(
        (raw["x_norm_clstoken"], raw["x_norm_patchtokens"].mean(dim=1)), dim=-1
    )

    assert model.dim == 64
    assert model.fc.in_features == 64
    assert model.backbone_provenance["feature_pool"] == "cls_patch_mean"
    torch.testing.assert_close(model.forward_features(x), expected)


def test_channel_adaptive_dino_loads_native_six_channel_backbone(monkeypatch, tmp_path):
    checkpoint = tmp_path / "channel_adaptive_dino.pth"
    checkpoint.write_bytes(b"fake native-channel checkpoint")

    def fake_load(repo, model, **kwargs):
        assert repo == str(tmp_path.resolve())
        assert model == "channel_adaptive_dino_vitl16"
        assert kwargs["pretrained_path"] == str(checkpoint.resolve())
        assert kwargs["in_channels"] == 6
        return _FakeChannelAdaptiveDino()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    model = CCASModel(
        num_classes=11, variant="original", backbone_source="channel_adaptive_dino",
        hub_repo_dir=tmp_path, checkpoint_path=checkpoint,
        hub_model="channel_adaptive_dino_vitl16", input_channels=6, img=224,
        feature_pool="cls_patch_mean",
    )
    assert model.backbone_provenance["source"] == "channel_adaptive_dino"
    assert model.backbone_provenance["input_channels"] == 6
    assert model.dim == 2 * 6 * 32
    assert model.fc.in_features == 2 * 6 * 32
    assert tuple(model(torch.randn(2, 6, 16, 16)).shape) == (2, 11)


def test_channel_adaptive_dino_uses_official_bag_of_channels_features(monkeypatch, tmp_path):
    backbone = _FakeChannelAdaptiveDino()
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: backbone)
    model = CCASModel(
        num_classes=11, variant="original", backbone_source="channel_adaptive_dino",
        hub_repo_dir=tmp_path, pretrained=False, hub_model="channel_adaptive_dino_vitl16",
        input_channels=6, img=224, feature_pool="cls_patch_mean",
    )
    x = torch.randn(2, 6, 16, 16)
    patches, cls = backbone.get_intermediate_layers(x, n=1)[-1]
    expected = torch.cat((cls, patches.mean(dim=-2).reshape(2, -1)), dim=-1)
    torch.testing.assert_close(model.forward_features(x), expected)
