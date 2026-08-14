"""ViT-S backbone + GMoE-style per-patch MoE — the clean substrate for the routing study.

Why ViT here (vs the ResNet): in a CNN, routing granularity is tangled with channels and depth
(layer4/3/2 = 512/256/128 ch). A ViT decouples them: every block sees the SAME token count and
width, and 'granularity' is a pure choice of WHERE the router decides:

    PatchMoE(granularity='patch')   -> one routing decision PER TOKEN  (GMoE: content attributes)
    PatchMoE(granularity='sample')  -> one routing decision PER IMAGE  (capacity-matched control)

Both use N experts, so 'patch - sample' isolates routing GRANULARITY at matched capacity — the
honest routing effect, with no extra-sublayer confound (the MoE REPLACES the block's MLP, it does
not add one; so unlike the CNN SpatialMoEBlock there is no LN+FFN capacity to subtract, and the
N=1 case is just the dense block). Experts are UPCYCLED from the block's own MLP (GMoE-style), so
at init every expert is identical and the MoE block == the dense block; experts diverge as routing
specializes.

Interface contract (matches run_experiment.py + audit/*):
  model.forward_features(x) -> [B, dim] (cls token, post-norm)        model.fc : Linear(dim,classes)
  model.moe_block           -> the LAST MoE block, for the audit       model.moe_spatial : bool
  model.aux_losses(losses)  -> summed z+balance over all MoE blocks
  moe_block.top1()          -> per-TOKEN ids [B*T] (patch) or per-IMAGE ids [B] (sample)
  moe_block.tokens_per_image, .experts, .N, .k                         (audit + param accounting)
"""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


def input_instance_norm(x):
    """Per-sample, per-channel standardization of an input batch [B,C,H,W].

    The global-correction baseline: removes a per-image (hence per-site) affine
    location-scale exactly when that shift is spatially HOMOGENEOUS (shift.type='affine').
    It CANNOT cancel a content-region-dependent shift ('affine_spatial'), because different
    regions need different corrections — which is the gap MoE's conditional routing fills."""
    mu = x.mean(dim=(2, 3), keepdim=True)
    sd = x.std(dim=(2, 3), keepdim=True) + 1e-5
    return (x - mu) / sd


class Mlp(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, heads, drop=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):                                   # x: [B, T, C]
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # [B, H, T, dh]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = self.drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class PatchMoE(nn.Module):
    """Replaces a block's MLP with N routed copies of it (cosine router, top-k).

    granularity='patch'  : route every token independently (GMoE).
    granularity='sample' : one decision per image (pooled token) -> all its tokens share experts.
    N=1 reduces to the original MLP (== dense block); experts are deep-copies of `mlp` (upcycle)."""
    def __init__(self, mlp: nn.Module, dim: int, N=8, k=2, granularity="patch",
                 router="cosine", router_temp=0.07, routing_estimator="selected_st"):
        super().__init__()
        self.N, self.k = N, max(1, min(k, N))
        self.granularity = granularity
        if routing_estimator not in ("selected_st", "legacy_renorm"):
            raise ValueError("PatchMoE routing_estimator must be selected_st|legacy_renorm")
        self.routing_estimator = str(routing_estimator)
        self.experts = nn.ModuleList([copy.deepcopy(mlp) for _ in range(N)])   # upcycle from the MLP
        self.router_type = router
        if router == "linear":
            self.router = nn.Linear(dim, N)
        elif router == "cosine":
            self.proj = nn.Linear(dim, dim, bias=False)
            self.codebook = nn.Parameter(torch.randn(N, dim) * 0.1)
            self.log_temp = nn.Parameter(torch.tensor(math.log(router_temp)))
        else:
            raise ValueError(f"unknown router: {router}")
        self.tokens_per_image = None
        self.last = None

    def _route(self, t):                                    # t: [M, C] -> [M, N]
        if self.router_type == "linear":
            return self.router(t).clamp(-20, 20)
        z = F.normalize(self.proj(t), dim=-1)
        e = F.normalize(self.codebook, dim=-1)
        return ((z @ e.t()) / self.log_temp.exp().clamp(min=1e-2)).clamp(-20, 20)

    def _apply_experts(self, tokens, idx, w):
        """tokens:[M,C]  idx:[M,k]  w:[M,k] -> [M,C] (masked dispatch, disjoint within a slot)."""
        out = torch.zeros_like(tokens)
        for slot in range(self.k):
            contrib = torch.zeros_like(tokens)
            ids, ws = idx[:, slot], w[:, slot]
            for e in range(self.N):
                m = ids == e
                if m.any():
                    contrib[m] = self.experts[e](tokens[m]).to(contrib.dtype)
            out = out + ws.unsqueeze(-1) * contrib
        return out

    def forward(self, x):                      # [B,T,C] (ViT) or [B,H,W,C] (ConvNeXt channels-last MLP)
        orig = x.shape
        if x.ndim == 4:                        # ConvNeXt block.mlp sees [B,H,W,C] -> flatten spatial to tokens
            x = x.reshape(orig[0], orig[1] * orig[2], orig[3])
        B, T, C = x.shape
        self.tokens_per_image = T
        if self.granularity == "sample":
            logits = self._route(x.mean(dim=1))             # [B, N] one decision per image
        else:
            logits = self._route(x.reshape(B * T, C))       # [B*T, N] per token
        gates = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(gates, self.k, dim=-1)
        if self.k == 1 and self.routing_estimator == "selected_st":
            # Forward value is exactly one (hard sparse dispatch), while the backward pass keeps
            # the selected softmax probability's task gradient. Renormalizing a single value by
            # itself would make the router invisible to the classification objective.
            topv = topv + (torch.ones_like(topv) - topv).detach()
        else:
            topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        self.last = {"logits": logits, "gates": gates, "idx": topi,
                     "routing_estimator": self.routing_estimator}

        if self.granularity == "sample":
            # expand per-image choice to all tokens, then dispatch on the flat token set
            idx = topi.unsqueeze(1).expand(B, T, self.k).reshape(B * T, self.k)
            w = topv.unsqueeze(1).expand(B, T, self.k).reshape(B * T, self.k)
            out = self._apply_experts(x.reshape(B * T, C), idx, w)
        else:
            out = self._apply_experts(x.reshape(B * T, C), topi, topv)
        out = out.reshape(B, T, C)
        return out.reshape(orig) if len(orig) == 4 else out      # back to [B,H,W,C] for ConvNeXt

    def aux_loss(self, zloss_w, balance_w):
        logits, gates, idx = self.last["logits"], self.last["gates"], self.last["idx"]
        M = logits.shape[0]
        z = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        dispatch = torch.zeros(self.N, device=logits.device)
        for slot in range(self.k):
            dispatch.scatter_add_(0, idx[:, slot], torch.ones(M, device=logits.device))
        frac = dispatch / (M * self.k)
        balance = self.N * (frac * gates.mean(dim=0)).sum()
        return zloss_w * z + balance_w * balance

    def top1(self):
        """per-TOKEN ids [B*T] (patch) or per-IMAGE ids [B] (sample)."""
        return self.last["idx"][:, 0].detach().cpu()


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)     # may be wrapped: see make_moe
        self.is_moe = False

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))                     # mlp is PatchMoE if upcycled
        return x


class ViT(nn.Module):
    def __init__(self, num_classes, img=32, patch=4, dim=192, depth=9, heads=6,
                 mlp_ratio=4.0, drop=0.1, instance_norm=False, input_channels=3, **_ignore):
        super().__init__()
        self.input_channels = int(input_channels)
        self.patch_size = int(patch)
        self.image_size = int(img)
        self.patch_embed = nn.Conv2d(self.input_channels, dim, patch, patch)
        n_patches = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)
        self.dim = dim
        self.input_in = bool(instance_norm)

    def forward_features(self, x):
        if self.input_in:
            x = input_instance_norm(x)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, n_patches, dim]
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = self.drop(torch.cat([cls, x], dim=1) + self.pos)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]                            # cls token

    def forward(self, x):
        return self.fc(self.forward_features(x))


def vit_small(num_classes, **kw):
    return ViT(num_classes, **kw)


class TimmViT(nn.Module):
    """Pretrained-ViT backbone (timm) wrapped to the repo's interface contract.

    This is the GMoE / sparse-upcycling substrate: a dense ImageNet-pretrained ViT-S/16
    (196 content-rich tokens, not 16) whose last `moe_layers` block MLPs are later replaced
    by `PatchMoE` (build_vit) — experts upcycled from the pretrained MLP, so at init the MoE
    block == the dense block and specialization is learned during fine-tuning. We do NOT train
    a ViT from scratch (that needs JFT-scale data); we fine-tune a pretrained one.

    Contract exposed (so run_experiment.py + audit/* are unchanged):
      .blocks  -> the backbone's transformer blocks (ModuleList-like; build_vit swaps blk.mlp)
      .dim     -> embed dim (384 for ViT-S)
      .fc      -> Linear(dim, num_classes) classifier head we own (DANN reads .fc.in_features)
      forward_features(x) -> [B, dim] pooled (cls) pre-logit features
      forward(x)          -> [B, num_classes]
    """
    def __init__(self, num_classes, timm_name="vit_small_patch16_224", img=224,
                 pretrained=True, drop=0.0, drop_path=0.0, instance_norm=False, **_ignore):
        super().__init__()
        import timm
        # num_classes=0 -> head is Identity and forward_head pools to [B, dim] (global_pool='token').
        # drop_path = stochastic depth: the standard ViT regularizer (layer-level, no pixel change ->
        # batch-effect-safe), needed because RxRx1 + geometry-only aug lets a ViT memorize the train set.
        try:                                               # ViT/Swin accept img_size; ConvNeXt does not
            self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0,
                                               img_size=img, drop_rate=drop, drop_path_rate=drop_path)
        except TypeError:
            self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0,
                                               drop_rate=drop, drop_path_rate=drop_path)
        self.dim = self.backbone.num_features
        # Flat block list for MoE upcycling, backbone-agnostic: plain ViT/DINOv2 (.blocks), Swin
        # (.layers[i].blocks), ConvNeXt (.stages[i].blocks). Dense runs don't touch it; build_vit
        # upcycles the last n. ConvNeXt/Swin block.mlp sees [B,H,W,C] -> PatchMoE handles 4D.
        self.blocks = getattr(self.backbone, "blocks", None)
        if self.blocks is None:
            for cont in ("layers", "stages"):
                if hasattr(self.backbone, cont):
                    blks = []
                    for stage in getattr(self.backbone, cont):
                        blks += list(getattr(stage, "blocks", []))
                    self.blocks = nn.ModuleList(blks) if blks else None
                    break
        self.fc = nn.Linear(self.dim, num_classes)
        self.input_in = bool(instance_norm)

    def forward_features(self, x):
        if self.input_in:
            x = input_instance_norm(x)
        feats = self.backbone.forward_features(x)                 # [B, T, dim] tokens
        return self.backbone.forward_head(feats, pre_logits=True) # [B, dim] pooled pre-logits

    def forward(self, x):
        return self.fc(self.forward_features(x))


def vit_pretrained(num_classes, **kw):
    return TimmViT(num_classes, **kw)
