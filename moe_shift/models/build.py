"""Assemble dense ResNet-18 / ViT-S or their MoE variants."""
import torch

from .moe import MoEBlock
from .oracle import SharedRoutedBlock
from .spatial_moe import SpatialMoEBlock
from .resnet import resnet18, resnet50
from .vit import vit_small, vit_pretrained, PatchMoE


def _truthy(v):
    return v.lower() in ("1", "true", "yes") if isinstance(v, str) else bool(v)


def _inorm(m):
    """Robustly read the model.instance_norm flag (handles bool or string overrides)."""
    v = m.get("instance_norm", False)
    return v.lower() in ("1", "true", "yes") if isinstance(v, str) else bool(v)


def build_vit(cfg):
    """ViT-S backbone; optionally replace the last `moe_layers` blocks' MLP with a PatchMoE
    (per-patch routing if model.moe.spatial else sample-level, capacity-matched). The MoE
    REPLACES the MLP (no added sublayer), and experts are upcycled from that MLP.

    vit.pretrained=true  -> load a timm ImageNet-pretrained ViT-S/16 (the GMoE/upcycling substrate,
                            196 tokens) and upcycle experts from its pretrained MLP.
    vit.pretrained=false -> the from-scratch custom ViT (legacy CIFAR track)."""
    m = cfg["model"]
    v = m.get("vit", {})
    if _truthy(v.get("pretrained", False)):
        model = vit_pretrained(m["num_classes"], timm_name=v.get("timm_name", "vit_small_patch16_224"),
                               img=v.get("img", 224), pretrained=True, drop=v.get("drop", 0.0),
                               drop_path=v.get("drop_path", 0.0), instance_norm=_inorm(m))
    else:
        model = vit_small(m["num_classes"], img=v.get("img", 32), patch=v.get("patch", 4),
                          dim=v.get("dim", 192), depth=v.get("depth", 9), heads=v.get("heads", 6),
                          mlp_ratio=v.get("mlp_ratio", 4.0), drop=v.get("drop", 0.1),
                          instance_norm=_inorm(m))

    moe_blocks, spatial = [], False
    if m["moe"]["enabled"]:
        gran = "patch" if _truthy(m["moe"].get("spatial", True)) else "sample"
        n_moe = int(m["moe"].get("moe_layers", 1))          # last n blocks -> MoE (GMoE uses 2)
        for blk in list(model.blocks)[-n_moe:]:
            pm = PatchMoE(blk.mlp, model.dim, N=m["moe"]["N"], k=m["moe"]["k"], granularity=gran,
                          router=m["moe"].get("router", "cosine"),
                          router_temp=m["moe"].get("router_temp", 0.07))
            blk.mlp, blk.is_moe = pm, True
            moe_blocks.append(pm)
        spatial = (gran == "patch")

    object.__setattr__(model, "moe_block", moe_blocks[-1] if moe_blocks else None)   # last MoE = audit subject
    object.__setattr__(model, "moe_spatial", spatial)       # patch -> per-token audit; sample -> per-image
    object.__setattr__(model, "_moe_blocks", moe_blocks)

    def aux_losses(losses_cfg):
        if not moe_blocks:
            return torch.zeros((), device=next(model.parameters()).device)
        return sum(b.aux_loss(losses_cfg["zloss_w"], losses_cfg["balance_w"]) for b in moe_blocks)
    object.__setattr__(model, "aux_losses", aux_losses)
    return model


def build_model(cfg):
    m = cfg["model"]
    arch = m.get("arch", "resnet18")
    if arch in ("vit", "convnext"):          # both use the timm-backbone wrapper + MLP upcycling
        return build_vit(cfg)
    if arch == "resnet50":                         # RxRx1 / WILDS-standard backbone (pretrained, BN)
        model = resnet50(m["num_classes"], pretrained=_truthy(m.get("pretrained", True)),
                         input_norm=_inorm(m), norm=m.get("norm", "batchnorm"))
    else:
        model = resnet18(m["num_classes"], stem=m["stem"], norm=m["norm"], input_norm=_inorm(m))

    moe_block = None
    spatial = False
    if m["moe"]["enabled"]:
        # placement = which stage's LAST block becomes the MoE. Token count = that stage's H*W:
        #   layer4 -> 4x4=16, layer3 -> 8x8=64, layer2 -> 16x16=256. Channels auto-adapt (512/256/128).
        placement = m["moe"].get("placement", "layer4")
        layer = getattr(model, placement)
        last_idx = len(layer) - 1
        base = layer[last_idx]
        spatial = _truthy(m["moe"].get("spatial", False))
        if spatial:                                  # GMoE-style per-token routing (default cosine)
            moe_block = SpatialMoEBlock(base, N=m["moe"]["N"], k=m["moe"]["k"],
                                        router=m["moe"].get("router", "cosine"),
                                        router_temp=m["moe"].get("router_temp", 0.07))
        else:
            moe_block = MoEBlock(base, N=m["moe"]["N"], k=m["moe"]["k"],
                                 router=m["moe"].get("router", "linear"),
                                 router_temp=m["moe"].get("router_temp", 0.07))
        layer[last_idx] = moe_block

    # attach without re-registering as a submodule (moe_block already lives in layer4)
    object.__setattr__(model, "moe_block", moe_block)
    object.__setattr__(model, "moe_spatial", spatial)   # tells the runner to use the per-token audit

    def aux_losses(losses_cfg):
        if moe_block is None:
            return torch.zeros((), device=next(model.parameters()).device)
        return moe_block.aux_loss(losses_cfg["zloss_w"], losses_cfg["balance_w"])
    object.__setattr__(model, "aux_losses", aux_losses)

    return model


def build_oracle_model(cfg):
    """ResNet-18 whose last layer4 block is a SharedRoutedBlock (oracle disentangling).
    One routed expert per SEEN site (= K-1); routing is by the true site label, set
    per-batch by the runner via model.set_sites(). model.set_shared_only(True) switches
    to shared-path-only inference (used for the held-out/unseen site)."""
    m = cfg["model"]
    model = resnet18(m["num_classes"], stem=m["stem"], norm=m["norm"], input_norm=_inorm(m))

    last_idx = len(model.layer4) - 1
    base = model.layer4[last_idx]
    n_routed = cfg["sites"]["K"] - 1                  # seen sites {0..K-2}; site K-1 is held out
    p_drop = cfg["model"].get("oracle", {}).get("p_drop", 0.5)
    block = SharedRoutedBlock(base, n_routed=n_routed, p_drop=p_drop)
    model.layer4[last_idx] = block

    object.__setattr__(model, "oracle_block", block)
    object.__setattr__(model, "set_sites", lambda sites: block.set_sites(sites))
    object.__setattr__(model, "set_shared_only", lambda flag: setattr(block, "shared_only", flag))
    return model
