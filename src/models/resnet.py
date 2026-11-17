"""ResNet-18 for 1-channel log-mel spectrograms.

torchvision's ResNet-18 expects colour images (3 channels: red, green, blue). Our input has
one channel. With ImageNet weights we replace the first convolution by a 1-channel one whose
filters are the AVERAGE of the pretrained red, green and blue filters, so the pretrained
edge and texture detectors are kept. The last layer is replaced by a 7-class output.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(n_classes: int = 7, in_channels: int = 1, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    old = model.conv1
    new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)
    if pretrained:
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
    model.conv1 = new
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model
