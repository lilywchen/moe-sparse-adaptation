"""Published-strength six-channel RxRx1 classifier backbones.

The wrappers deliberately expose a small, shared contract: ``forward_features``
returns the 1024-dimensional profiling embedding and ``forward`` returns the
1,139-way perturbation/control logits.  This keeps architecture comparisons on
the same head and data recipe.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _six_channel_conv(conv: nn.Conv2d, pretrained: bool) -> nn.Conv2d:
    replacement = nn.Conv2d(
        6, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
        dilation=conv.dilation, groups=conv.groups, bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    if pretrained:
        # Preserve the pretrained spatial filters without privileging an RGB
        # ordering that has no meaning for Cell Painting stains.  Replicating
        # the channel-mean and scaling by 3/6 preserves the expected response
        # magnitude of the original three-channel convolution.
        with torch.no_grad():
            mean = conv.weight.mean(dim=1, keepdim=True)
            replacement.weight.copy_(mean.repeat(1, 6, 1, 1) * 0.5)
            if conv.bias is not None:
                replacement.bias.copy_(conv.bias)
    return replacement


class ProfilingHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 1139,
                 embedding_dim: int = 1024):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DenseNet161RxRx1(nn.Module):
    """DenseNet-161 + the two-layer 1024-d head reported for RxRx1."""

    def __init__(self, num_classes: int = 1139, pretrained: bool = False,
                 memory_efficient: bool = False):
        super().__init__()
        from torchvision.models import DenseNet161_Weights, densenet161

        weights = DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None
        network = densenet161(weights=weights, memory_efficient=memory_efficient)
        network.features.conv0 = _six_channel_conv(network.features.conv0, pretrained)
        self.features = network.features
        self.head = ProfilingHead(network.classifier.in_features, num_classes)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.features(x), inplace=True)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.features(self.forward_backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.classifier(self.forward_features(x))


class ResNet50RxRx1(nn.Module):
    """ResNet-50 control with the same profiling head as DenseNet-161."""

    def __init__(self, num_classes: int = 1139, pretrained: bool = False):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        network = resnet50(weights=weights)
        network.conv1 = _six_channel_conv(network.conv1, pretrained)
        self.stem = nn.Sequential(
            network.conv1, network.bn1, network.relu, network.maxpool)
        self.layer1, self.layer2 = network.layer1, network.layer2
        self.layer3, self.layer4 = network.layer3, network.layer4
        self.head = ProfilingHead(network.fc.in_features, num_classes)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.features(self.forward_backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.classifier(self.forward_features(x))


class TimmViTRxRx1(nn.Module):
    """ViT-S/16 challenger with an otherwise matched 1024-dimensional head."""

    def __init__(self, num_classes: int = 1139, image_size: int = 512,
                 pretrained: bool = False, name: str = "vit_small_patch16_224"):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            name, pretrained=pretrained, in_chans=6, img_size=image_size,
            num_classes=0, global_pool="token")
        dim = int(getattr(self.backbone, "num_features"))
        self.head = ProfilingHead(dim, num_classes)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.features(self.forward_backbone(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.classifier(self.forward_features(x))


def build_rxrx1_calibration_model(name: str, num_classes: int = 1139,
                                  image_size: int = 512,
                                  pretrained: bool = False,
                                  memory_efficient: bool = False) -> nn.Module:
    if name == "densenet161":
        return DenseNet161RxRx1(num_classes, pretrained, memory_efficient)
    if name == "resnet50":
        return ResNet50RxRx1(num_classes, pretrained)
    if name in ("vit_small", "vit_small_patch16"):
        return TimmViTRxRx1(num_classes, image_size, pretrained)
    raise ValueError(f"unknown RxRx1 calibration model: {name!r}")
