"""Image-CNN hash encoder (paper Sec. 4: VGG-19 → fch; here ResNet-18 → tanh)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ImageCNN(nn.Module):
    """Map an RGB face crop to a continuous hash code in [-1, 1]^c.

    Paper uses VGG-19 (fc7 replaced by fch with tanh). This rebuild uses
    ResNet-18 for speed/memory (see DEVIATIONS.md D1).
    """

    def __init__(self, hash_bits: int = 63, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.hash_layer = nn.Linear(feat_dim, hash_bits)
        nn.init.normal_(self.hash_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.hash_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return torch.tanh(self.hash_layer(feats))
