"""ResNet-18 with configurable stem + norm, exposing forward_features and .layer4
(whose final block build.py swaps for an MoEBlock)."""
import torch.nn.functional as F
from torch import nn


def norm_layer(norm: str, channels: int) -> nn.Module:
    if norm == "groupnorm":
        return nn.GroupNorm(min(32, channels), channels)
    if norm in ("batchnorm", "adabn"):       # adabn behaves as BN at train time
        return nn.BatchNorm2d(channels)
    raise ValueError(f"unknown norm: {norm}")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1, norm="groupnorm"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.n1 = norm_layer(norm, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.n2 = norm_layer(norm, out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                norm_layer(norm, out_c),
            )

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes, stem="cifar", norm="groupnorm", input_norm=False,
                 input_channels=3):
        super().__init__()
        # InstanceNorm baseline: per-image, per-channel standardization of the INPUT cancels
        # any per-image affine nuisance for free (the trivial floor for homogeneous batch effects).
        input_channels = int(input_channels)
        self.input_norm = nn.InstanceNorm2d(input_channels, affine=False) if input_norm else None
        if stem == "cifar":
            self.stem = nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, 1, 1, bias=False),
                norm_layer(norm, 64), nn.ReLU(inplace=True))
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False),
                norm_layer(norm, 64), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))
        self.in_c = 64
        self.layer1 = self._make(64, 2, 1, norm)
        self.layer2 = self._make(128, 2, 2, norm)
        self.layer3 = self._make(256, 2, 2, norm)
        self.layer4 = self._make(512, 2, 2, norm)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make(self, out_c, blocks, stride, norm):
        layers, strides = [], [stride] + [1] * (blocks - 1)
        for s in strides:
            layers.append(BasicBlock(self.in_c, out_c, s, norm))
            self.in_c = out_c
        return nn.Sequential(*layers)

    def forward_features(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)

    def forward(self, x):
        return self.fc(self.forward_features(x))


def resnet18(num_classes, stem="cifar", norm="groupnorm", input_norm=False,
             input_channels=3):
    return ResNet18(num_classes, stem, norm, input_norm, input_channels)


# ---------------------------------------------------------------------------
# ResNet-50 — the RxRx1 / WILDS-standard backbone (ImageNet-pretrained, BatchNorm).
# ---------------------------------------------------------------------------
def _bn_to_gn(module):
    """Replace every BatchNorm2d with GroupNorm(min(32,C), C). Only for norm='groupnorm';
    note this DISCARDS the pretrained BN affines, so pair it with pretrained=False (or accept
    a warm conv init). BN is itself a batch-coupling mechanism, so GroupNorm is the relevant
    arm for a batch-effect study — but the WILDS baseline is pretrained BN, so that is the default."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(min(32, child.num_features), child.num_features))
        else:
            _bn_to_gn(child)


class ResNet50Backbone(nn.Module):
    """ImageNet-pretrained torchvision ResNet-50 wrapped to the repo contract:
    forward_features(x) -> [B, 2048], .fc, and .layer1..layer4 (build.py swaps the last
    layer4 Bottleneck for an MoE/SpatialMoE block; both wrap a block by deepcopy / reading
    .conv1.in_channels, which is 2048 for the last layer4 block)."""
    def __init__(self, num_classes, pretrained=True, input_norm=False, norm="batchnorm"):
        super().__init__()
        import torchvision
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        self.input_norm = nn.InstanceNorm2d(3, affine=False) if input_norm else None
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(net.fc.in_features, num_classes)     # 2048 -> num_classes
        if norm == "groupnorm":
            _bn_to_gn(self)

    def forward_features(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.pool(x).flatten(1)

    def forward(self, x):
        return self.fc(self.forward_features(x))


def resnet50(num_classes, pretrained=True, input_norm=False, norm="batchnorm"):
    return ResNet50Backbone(num_classes, pretrained, input_norm, norm)
