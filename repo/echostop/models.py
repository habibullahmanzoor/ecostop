"""Vision architectures for the pilot (plan §16): ResNet-9 (fast-CIFAR
variant) and a CIFAR-scale ResNet-18. Both take `in_channels` so the same
code handles CIFAR-10/SVHN (3-channel) and Fashion-MNIST/MNIST-family
(1-channel) without resizing inputs, and both end in adaptive average
pooling so they're robust to the 28x28 vs 32x32 spatial-size difference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


def _conv_bn(in_ch: int, out_ch: int, pool: bool = False) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(_conv_bn(ch, ch), _conv_bn(ch, ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNet9(nn.Module):
    """DAWNBench-style fast CIFAR ResNet (plan §16 — the cheap, high-repetition
    architecture). Adaptive final pool so 28x28 (Fashion-MNIST) and 32x32
    (CIFAR-10/SVHN) both work without resizing.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.prep = _conv_bn(in_channels, 64)
        self.layer1 = _conv_bn(64, 128, pool=True)
        self.res1 = Residual(128)
        self.layer2 = _conv_bn(128, 256, pool=True)
        self.layer3 = _conv_bn(256, 512, pool=True)
        self.res3 = Residual(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.res3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def resnet18_small(in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    """torchvision ResNet-18 with the ImageNet stem (7x7 stride-2 conv +
    maxpool) replaced by a 3x3 stride-1 conv and no initial maxpool — the
    standard adaptation for CIFAR-scale (32x32) inputs, since the ImageNet
    stem over-downsamples small images before the residual stages ever run.
    """
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model


def build_model(name: str, in_channels: int, num_classes: int) -> nn.Module:
    if name == "resnet9":
        return ResNet9(in_channels=in_channels, num_classes=num_classes)
    if name == "resnet18":
        return resnet18_small(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"unknown model name: {name!r}")


class TabularMLP(nn.Module):
    """Small MLP for the tabular calibration-corpus jobs (plan §16)."""

    def __init__(self, in_features: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
