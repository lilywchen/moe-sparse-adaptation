"""Build the CCAS model with exactly one FFN block converted.

Variants (all function-preserving at init):
  original    - untouched pretrained FFN (lower total budget P0; accuracy-compute reference)
  dense_wide  - the same capacity as the MoE, allocated as SHARED width (fixed budget P*)
  moe         - learned top-1 router over E experts             (fixed budget P*)
  moe_frozen  - identical architecture, router frozen at init   (fixed budget P*)
"""
import hashlib
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

from .surgery import convert_block


def _git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _actual_blocks(backbone):
    """Flatten DINOv2 ``BlockChunk`` containers without re-registering modules.

    Cell-DINO checkpoints are stored with four chunk containers whose leading entries are
    ``Identity`` placeholders.  Surgery and layer-wise decay must operate on the 12 actual blocks,
    while the backbone itself retains its original chunked forward/state-dict structure.
    """
    out = []
    for item in backbone.blocks:
        if hasattr(item, "mlp"):
            out.append(item)
        else:
            out.extend(block for block in item if hasattr(block, "mlp"))
    if not out:
        raise TypeError("backbone exposes no transformer blocks with an .mlp")
    return out


class CCASModel(nn.Module):
    def __init__(self, num_classes, variant="moe", placement="middle", n_experts=8, top_k=1,
                 routing_unit="token", geometry="cosine", balance="global", temperature=0.07,
                 timm_name="vit_small_patch14_dinov2", img=224, pretrained=True,
                 drop_path=0.2, sym_break_wide=0.1, sym_break_moe=0.0,
                 backbone_source="timm", hub_repo_dir=None, checkpoint_path=None,
                 hub_model="cell_dino_cp_vits8", input_channels=5,
                 feature_pool="cls",
                 expected_hub_repo_commit=None, freeze_backbone=False, **_ignore):
        super().__init__()
        self.backbone_source = str(backbone_source)
        self.feature_pool = str(feature_pool)
        if self.feature_pool not in ("cls", "cls_patch_mean"):
            raise ValueError(f"unknown feature_pool: {self.feature_pool!r}")
        if self.backbone_source == "timm":
            import timm
            self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0,
                                              img_size=img, drop_path_rate=drop_path)
            self.backbone_provenance = {
                "source": "timm", "model": timm_name, "pretrained": bool(pretrained),
            }
        elif self.backbone_source in ("cell_dino", "channel_adaptive_dino"):
            if self.backbone_source == "cell_dino" and int(img) != 128:
                raise ValueError("Cell-DINO CP ViT-S/8 competence protocol requires img_size=128")
            if self.backbone_source == "channel_adaptive_dino" and int(img) != 224:
                raise ValueError("Channel-Adaptive DINO ViT-L/16 protocol requires img_size=224")
            if not hub_repo_dir:
                raise ValueError(f"model.hub_repo_dir is required for {self.backbone_source}")
            repo = Path(hub_repo_dir).expanduser().resolve()
            if not repo.is_dir():
                raise FileNotFoundError(f"DINO hub repo not found: {repo}")
            repo_sha = _git_sha(repo)
            if expected_hub_repo_commit and repo_sha != str(expected_hub_repo_commit):
                raise RuntimeError(
                    f"DINO repo commit mismatch: expected {expected_hub_repo_commit}, got {repo_sha}")
            checkpoint = None
            if pretrained:
                if not checkpoint_path:
                    raise ValueError(
                        f"model.checkpoint_path is required when {self.backbone_source} is pretrained")
                checkpoint = Path(checkpoint_path).expanduser().resolve()
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"DINO checkpoint not found: {checkpoint}")
            self.backbone = torch.hub.load(
                str(repo), hub_model, source="local", pretrained=bool(pretrained),
                pretrained_path=(str(checkpoint) if checkpoint is not None else None),
                in_channels=int(input_channels), drop_path_rate=float(drop_path),
            )
            self.backbone_provenance = {
                "source": self.backbone_source, "model": hub_model, "pretrained": bool(pretrained),
                "hub_repo_commit": repo_sha,
                "checkpoint_filename": (checkpoint.name if checkpoint is not None else None),
                "checkpoint_sha256": (_sha256(checkpoint) if checkpoint is not None else None),
                "input_channels": int(input_channels),
                "feature_pool": self.feature_pool,
            }
        else:
            raise ValueError(f"unknown model.backbone_source: {self.backbone_source!r}")

        self.dim = self.backbone.num_features * (2 if self.feature_pool == "cls_patch_mean" else 1)
        # Plain list on purpose: blocks remain registered only under the backbone, preserving the
        # official checkpoint namespace. convert_block only needs mutable references.
        self.blocks = _actual_blocks(self.backbone)
        self.variant = variant
        # The classifier must exist before capacity accounting so ``total_params`` really means
        # the complete trainable model rather than backbone-only parameters.
        self.fc = nn.Linear(self.dim, num_classes)

        _, self.capacity = convert_block(
            self, variant, placement=placement, n_experts=n_experts, top_k=top_k,
            routing_unit=routing_unit, geometry=geometry, balance=balance,
            temperature=temperature, sym_break_wide=sym_break_wide, sym_break_moe=sym_break_moe)

        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            if variant != "original":
                raise ValueError("freeze_backbone is a competence diagnostic for variant=original only")
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    # -- environment ids are used ONLY by the within-batch balancing loss, never at inference --
    def set_env(self, env):
        if self._moe_block is not None:
            self._moe_block.set_env(env)

    def forward_features(self, x):
        f = self.backbone.forward_features(x)
        if isinstance(f, dict):
            cls = f["x_norm_clstoken"]
            if self.feature_pool == "cls":
                return cls
            if "x_norm_patchtokens" not in f:
                raise KeyError("cls_patch_mean pooling requires x_norm_patchtokens")
            return torch.cat((cls, f["x_norm_patchtokens"].mean(dim=1)), dim=-1)
        if self.feature_pool != "cls":
            raise TypeError("cls_patch_mean pooling requires dictionary backbone features")
        return self.backbone.forward_head(f, pre_logits=True)

    def forward(self, x):
        return self.fc(self.forward_features(x))

    def aux_loss(self, balance_w, zloss_w=0.0):
        if self._moe_block is None:
            return torch.zeros((), device=self.fc.weight.device)
        return self._moe_block.aux_loss(balance_w, zloss_w)

    @property
    def moe_block(self):
        return self._moe_block


def build_ccas(cfg):
    m = cfg["model"]
    return CCASModel(num_classes=m["num_classes"], variant=m["variant"],
                     placement=m.get("placement", "middle"), n_experts=m.get("n_experts", 8),
                     top_k=m.get("top_k", 1), routing_unit=m.get("routing_unit", "token"),
                     geometry=m.get("geometry", "cosine"), balance=m.get("balance", "global"),
                     temperature=m.get("temperature", 0.07),
                     timm_name=m.get("timm_name", "vit_small_patch14_dinov2"),
                     img=cfg.get("img_size", 224), pretrained=m.get("pretrained", True),
                     drop_path=m.get("drop_path", 0.2),
                     sym_break_wide=m.get("sym_break_wide", 0.1),
                     sym_break_moe=m.get("sym_break_moe", 0.0),
                     backbone_source=m.get("backbone_source", "timm"),
                     hub_repo_dir=m.get("hub_repo_dir"),
                     checkpoint_path=m.get("checkpoint_path"),
                     hub_model=m.get("hub_model", "cell_dino_cp_vits8"),
                     input_channels=m.get("input_channels", 5),
                     feature_pool=m.get("feature_pool", "cls"),
                     expected_hub_repo_commit=m.get("expected_hub_repo_commit"),
                     freeze_backbone=m.get("freeze_backbone", False))
