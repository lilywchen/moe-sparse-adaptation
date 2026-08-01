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
        return {"x_norm_clstoken": tokens[:, 0]}


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
