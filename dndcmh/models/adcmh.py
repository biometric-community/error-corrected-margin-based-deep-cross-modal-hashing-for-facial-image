"""ADCMH: Image CNN + Attribute MLP → continuous hash codes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageCNN(nn.Module):
    def __init__(self, hash_bits: int = 63, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, hash_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        return torch.tanh(self.head(f))


class AttributeMLP(nn.Module):
    def __init__(self, attr_dim: int = 40, hash_bits: int = 63) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(attr_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, hash_bits),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(a))


class ADCMH(nn.Module):
    """Paired image/attribute encoders for deep cross-modal hashing."""

    def __init__(
        self,
        hash_bits: int = 63,
        attr_dim: int = 40,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.image_cnn = ImageCNN(hash_bits=hash_bits, pretrained=pretrained)
        self.attr_mlp = AttributeMLP(attr_dim=attr_dim, hash_bits=hash_bits)
        self.hash_bits = hash_bits

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_cnn(x)

    def encode_attr(self, a: torch.Tensor) -> torch.Tensor:
        return self.attr_mlp(a)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        return self.encode_image(x), self.encode_attr(a)

    @staticmethod
    def binarize(z: torch.Tensor) -> torch.Tensor:
        return torch.sign(z).clamp(min=-1.0)
