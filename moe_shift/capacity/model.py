"""Build the CCAS model: pretrained DINOv2 ViT-S/14 with exactly one FFN block converted.

Variants (all function-preserving at init):
  original    - untouched pretrained FFN (lower total budget P0; accuracy-compute reference)
  dense_wide  - the same capacity as the MoE, allocated as SHARED width (fixed budget P*)
  moe         - learned top-1 router over E experts             (fixed budget P*)
  moe_frozen  - identical architecture, router frozen at init   (fixed budget P*)
"""
import torch
import torch.nn as nn

from .surgery import convert_block


class CCASModel(nn.Module):
    def __init__(self, num_classes, variant="moe", placement="middle", n_experts=8, top_k=1,
                 routing_unit="token", geometry="cosine", balance="global", temperature=0.07,
                 timm_name="vit_small_patch14_dinov2", img=224, pretrained=True,
                 drop_path=0.2, sym_break_wide=0.1, sym_break_moe=0.0, **_ignore):
        super().__init__()
        import timm
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0,
                                          img_size=img, drop_path_rate=drop_path)
        self.dim = self.backbone.num_features
        self.blocks = self.backbone.blocks               # convert_block looks for .blocks
        self.variant = variant

        _, self.capacity = convert_block(
            self, variant, placement=placement, n_experts=n_experts, top_k=top_k,
            routing_unit=routing_unit, geometry=geometry, balance=balance,
            temperature=temperature, sym_break_wide=sym_break_wide, sym_break_moe=sym_break_moe)

        self.fc = nn.Linear(self.dim, num_classes)

    # -- environment ids are used ONLY by the within-batch balancing loss, never at inference --
    def set_env(self, env):
        if self._moe_block is not None:
            self._moe_block.set_env(env)

    def forward_features(self, x):
        f = self.backbone.forward_features(x)
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
                     sym_break_moe=m.get("sym_break_moe", 0.0))
