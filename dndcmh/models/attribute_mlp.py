"""Attribute-MLP hash encoder (paper Sec. 4: 512-ReLU → 512-ReLU → c-tanh)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttributeMLP(nn.Module):
    """Map a binary/continuous attribute vector to a continuous hash in [-1, 1]^c."""

    def __init__(self, attr_dim: int = 40, hash_bits: int = 63, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(attr_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hash_bits),
            nn.Tanh(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        return self.net(attrs)
