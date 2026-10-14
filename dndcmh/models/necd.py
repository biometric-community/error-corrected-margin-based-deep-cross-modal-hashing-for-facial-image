"""NECD: Neural Error-Correcting Decoder.

Paper uses belief-propagation on a Tanner graph for a designed (N,K) ECC.
Here we use a residual MLP soft decoder that maps noisy ±1 codes toward
shared codewords within Hamming radius e (documented in DEVIATIONS.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NECD(nn.Module):
    def __init__(self, hash_bits: int = 63, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hash_bits, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hash_bits),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # residual denoise in continuous space, then tanh toward ±1
        return torch.tanh(z + self.net(z))

    @staticmethod
    def flip_bits(code: torch.Tensor, e: int, rng=None) -> torch.Tensor:
        """Corrupt ±1 codes with up to e random bit flips (training noise)."""
        b, n = code.shape
        out = code.clone()
        for i in range(b):
            k = int(torch.randint(1, e + 1, (1,), generator=rng).item()) if e > 0 else 0
            if k == 0:
                continue
            idx = torch.randperm(n, generator=rng)[:k]
            out[i, idx] = -out[i, idx]
        return out
