"""Small from-scratch models for the RxRx1 HUVEC systematic study."""
from __future__ import annotations

import copy

import torch
from torch import nn

from .resnet import resnet18
from .vit import Block, Mlp, PatchMoE, ViT
from moe_shift.capacity.ffn import SharedResidualMoEFFN


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


class StudyViT(ViT):
    """DeiT-Tiny-sized six-channel ViT with optional late sparse FFNs."""

    def __init__(self, num_classes=1108, image_size=224, experts=1, top_k=1,
                 moe_layers=0, matched_hidden=None, dim=192, depth=12, heads=3,
                 residual_moe=None):
        super().__init__(
            num_classes=num_classes, img=image_size, patch=16, dim=int(dim),
            depth=int(depth), heads=int(heads),
            mlp_ratio=4.0, drop=0.0, instance_norm=False, input_channels=6,
        )
        self.moe_block_indices = []
        if residual_moe is not None and int(moe_layers) > 0:
            config = dict(residual_moe)
            for index in range(len(self.blocks) - int(moe_layers), len(self.blocks)):
                self.blocks[index].mlp = SharedResidualMoEFFN(
                    self.blocks[index].mlp, **config)
                self.moe_block_indices.append(index)
        elif matched_hidden is not None and int(moe_layers) > 0:
            for index in range(len(self.blocks) - int(moe_layers), len(self.blocks)):
                self.blocks[index].mlp = Mlp(self.dim, int(matched_hidden), 0.0)
        elif int(experts) > 1 and int(moe_layers) > 0:
            for index in range(len(self.blocks) - int(moe_layers), len(self.blocks)):
                routed = PatchMoE(
                    self.blocks[index].mlp, self.dim, N=int(experts), k=int(top_k),
                    granularity="patch", router="cosine", router_temp=0.7,
                )
                self.blocks[index].mlp = routed
                self.moe_block_indices.append(index)

    @property
    def moe_blocks(self):
        return tuple(self.blocks[index].mlp for index in self.moe_block_indices)

    def routing_aux_loss(self, balance_weight=0.01, zloss_weight=1e-4):
        if not self.moe_blocks:
            return self.fc.weight.new_zeros(())
        losses = []
        for block in self.moe_blocks:
            if isinstance(block, SharedResidualMoEFFN):
                losses.append(block.aux_loss(float(balance_weight), float(zloss_weight)))
            else:
                losses.append(block.aux_loss(float(zloss_weight), float(balance_weight)))
        return torch.stack(losses).mean()


def _nearest_matched_hidden(num_classes=1108, image_size=224, experts=4, moe_layers=2):
    target = parameter_count(StudyViT(
        num_classes, image_size, experts=experts, top_k=1, moe_layers=moe_layers))
    low, high = 768, 768 * int(experts) + 1024
    best = None
    while low <= high:
        hidden = (low + high) // 2
        model = StudyViT(num_classes, image_size, moe_layers=moe_layers,
                         matched_hidden=hidden)
        count = parameter_count(model)
        candidate = (abs(count - target), hidden, count, target)
        best = candidate if best is None or candidate < best else best
        if count < target:
            low = hidden + 1
        else:
            high = hidden - 1
    return best[1], {"dense_params": best[2], "moe_params": best[3],
                     "absolute_delta": best[0]}


def build_study_model(kind, num_classes=1108, image_size=224):
    if kind == "resnet18":
        model = resnet18(
            num_classes, stem="imagenet", norm="groupnorm", input_norm=False,
            input_channels=6)
        return model, {"kind": kind, "total_params": parameter_count(model)}
    if kind in ("vit_tiny", "mae_vit_tiny"):
        model = StudyViT(num_classes, image_size)
        return model, {"kind": kind, "total_params": parameter_count(model)}
    if kind == "vit_micro":
        model = StudyViT(
            num_classes, image_size, dim=128, depth=6, heads=4)
        return model, {
            "kind": kind, "total_params": parameter_count(model),
            "dim": 128, "depth": 6, "heads": 4, "patch_size": 16,
        }
    if kind in ("vit_tiny_moe", "mae_vit_tiny_moe"):
        model = StudyViT(num_classes, image_size, experts=4, top_k=1, moe_layers=2)
        return model, {
            "kind": kind, "total_params": parameter_count(model), "experts": 4,
            "top_k": 1, "moe_layers": 2,
        }
    if kind == "vit_tiny_dense_matched":
        hidden, audit = _nearest_matched_hidden(num_classes, image_size, 4, 2)
        model = StudyViT(num_classes, image_size, moe_layers=2, matched_hidden=hidden)
        return model, {
            "kind": kind, "total_params": parameter_count(model),
            "matched_hidden": hidden, **audit,
        }
    residual_kinds = {
        "vit_tiny_residual_token": dict(routing_unit="token", balance="global"),
        "vit_tiny_residual_image": dict(routing_unit="image", balance="global"),
        "vit_tiny_residual_within": dict(
            routing_unit="token", balance="within_environment"),
        "vit_tiny_residual_frozen": dict(
            routing_unit="token", balance="global", router_frozen=True),
    }
    if kind in residual_kinds:
        config = dict(
            n_experts=3, top_k=1, geometry="cosine", temperature=0.7,
            routing_estimator="full_st", **residual_kinds[kind])
        model = StudyViT(
            num_classes, image_size, moe_layers=2, residual_moe=config)
        return model, {
            "kind": kind, "total_params": parameter_count(model),
            "moe_layers": 2, "shared_dense_path": True,
            "residual_experts": 3, "top_k": 1, **config,
        }
    raise ValueError(f"unknown HUVEC study model: {kind!r}")


class MaskedAutoencoder(nn.Module):
    """A small visible-token MAE using the exact downstream dense or MoE encoder."""

    def __init__(self, encoder: StudyViT, mask_ratio=0.75, decoder_dim=128,
                 decoder_depth=2, decoder_heads=4):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = float(mask_ratio)
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must lie strictly between zero and one")
        self.decoder_embed = nn.Linear(encoder.dim, int(decoder_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(decoder_dim)))
        n_patches = (encoder.image_size // encoder.patch_size) ** 2
        self.decoder_pos = nn.Parameter(torch.zeros(1, n_patches + 1, int(decoder_dim)))
        self.decoder_blocks = nn.ModuleList([
            Block(int(decoder_dim), int(decoder_heads), mlp_ratio=4.0, drop=0.0)
            for _ in range(int(decoder_depth))
        ])
        self.decoder_norm = nn.LayerNorm(int(decoder_dim))
        patch_values = encoder.patch_size ** 2 * encoder.input_channels
        self.decoder_pred = nn.Linear(int(decoder_dim), patch_values)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos, std=0.02)

    def patchify(self, images):
        p = self.encoder.patch_size
        batch, channels, height, width = images.shape
        if height != width or height % p:
            raise ValueError("MAE input must be square and divisible by patch size")
        grid = height // p
        patches = images.reshape(batch, channels, grid, p, grid, p)
        return patches.permute(0, 2, 4, 3, 5, 1).reshape(batch, grid * grid, p * p * channels)

    def forward(self, images):
        tokens = self.encoder.patch_embed(images).flatten(2).transpose(1, 2)
        batch, n_patches, dim = tokens.shape
        keep_count = max(1, int(n_patches * (1.0 - self.mask_ratio)))
        noise = torch.rand(batch, n_patches, device=images.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep_count]
        positions = self.encoder.pos[:, 1:].expand(batch, -1, -1)
        visible = torch.gather(
            tokens + positions, 1, ids_keep.unsqueeze(-1).expand(-1, -1, dim))
        cls = self.encoder.cls.expand(batch, -1, -1) + self.encoder.pos[:, :1]
        encoded = torch.cat((cls, visible), dim=1)
        for block in self.encoder.blocks:
            encoded = block(encoded)
        encoded = self.encoder.norm(encoded)

        decoded = self.decoder_embed(encoded)
        missing = self.mask_token.expand(batch, n_patches - keep_count, -1)
        patch_tokens = torch.cat((decoded[:, 1:], missing), dim=1)
        patch_tokens = torch.gather(
            patch_tokens, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]))
        decoded = torch.cat((decoded[:, :1], patch_tokens), dim=1) + self.decoder_pos
        for block in self.decoder_blocks:
            decoded = block(decoded)
        prediction = self.decoder_pred(self.decoder_norm(decoded)[:, 1:])
        target = self.patchify(images)
        target = (target - target.mean(-1, keepdim=True)) / (
            target.var(-1, keepdim=True).add(1e-6).sqrt())
        mask = torch.ones(batch, n_patches, device=images.device)
        mask[:, :keep_count] = 0
        mask = torch.gather(mask, 1, ids_restore)
        loss = ((prediction - target) ** 2).mean(-1)
        reconstruction = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        auxiliary = self.encoder.routing_aux_loss() if self.encoder.moe_blocks else reconstruction.new_zeros(())
        return reconstruction, auxiliary


def clone_encoder_state(mae: MaskedAutoencoder):
    return copy.deepcopy(mae.encoder.state_dict())
