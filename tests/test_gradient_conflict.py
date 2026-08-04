import torch
import torch.nn as nn

from moe_shift.audit.gradient_conflict import (
    balanced_environment_draws,
    count_sketch,
    environment_index_groups,
    pairwise_cosine_metrics,
    profile_gradient_conflict,
)


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(4)
        self.x = torch.randn(12, 4, generator=generator)
        self.y = (self.x[:, 0] > 0).long()
        self.environment_ids = torch.tensor([0] * 6 + [1] * 6)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        environment = int(self.environment_ids[index])
        return self.x[index], self.y[index], environment, environment


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))

    def forward(self, x):
        return x + self.mlp(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock(), TinyBlock()])
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.fc(x)


def test_balanced_draws_are_reproducible_and_equal_size():
    groups = environment_index_groups(TinyDataset())
    first = balanced_environment_draws(groups, 3, 2, seed=7)
    second = balanced_environment_draws(groups, 3, 2, seed=7)
    assert first == second
    assert all(len(indices) == 3 for row in first for indices in row.values())


def test_count_sketch_and_cosine_metrics_detect_opposition():
    vector = torch.tensor([1.0, -2.0, 3.0])
    sketch = count_sketch([vector], sketch_size=32, seed=2)
    assert torch.equal(sketch, count_sketch([vector], sketch_size=32, seed=2))
    metrics = pairwise_cosine_metrics(torch.stack([sketch, -sketch]))
    assert metrics["mean_cosine"] < -0.999
    assert metrics["conflict_rate"] == 1.0


def test_profile_covers_every_ffn_and_restores_mode():
    torch.manual_seed(3)
    model = TinyModel().train()
    report = profile_gradient_conflict(
        model, TinyDataset(), "cpu", samples_per_environment=2, rounds=2,
        sketch_size=64, seed=9)
    assert model.training
    assert [layer["block_index"] for layer in report["layers"]] == [0, 1, 2]
    assert sorted(report["conflict_ranking"]) == [0, 1, 2]
    assert report["n_environments"] == 2
