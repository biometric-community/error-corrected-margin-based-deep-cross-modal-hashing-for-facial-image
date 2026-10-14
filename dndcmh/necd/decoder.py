"""Neural Error-Correcting Decoder (NECD).

Paper Sec. 3.2.2 / Appendix: Tanner-graph neural belief-propagation for an
(N,K) ECC with capability e ≥ m. This rebuild uses a simplified soft-decode
MLP that maps noisy ±1 codes toward clean codewords (see DEVIATIONS.md D2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDecodeMLP(nn.Module):
    """Feed-forward soft decoder: noisy code → soft codeword in (-1,1)^N."""

    def __init__(self, hash_bits: int = 63, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hash_bits, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hash_bits),
            nn.Tanh(),
        )

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        return self.net(noisy)


class NECD(nn.Module):
    """NECD wrapper with Hamming-ball noise training utilities."""

    def __init__(self, hash_bits: int = 63, capability_e: int = 6, hidden: int = 256):
        super().__init__()
        self.hash_bits = hash_bits
        self.capability_e = capability_e
        self.decoder = SoftDecodeMLP(hash_bits=hash_bits, hidden=hidden)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(codes)

    def decode_binary(self, codes: torch.Tensor) -> torch.Tensor:
        soft = self.forward(codes.float())
        return soft.sign().clamp(min=-1.0)

    @staticmethod
    def flip_bits(codes: torch.Tensor, max_flips: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Flip up to max_flips bits of ±1 codes (paper: noise within radius e)."""
        b, n = codes.shape
        out = codes.clone()
        for i in range(b):
            k = int(torch.randint(0, max_flips + 1, (1,), generator=generator).item())
            if k == 0:
                continue
            idx = torch.randperm(n, generator=generator)[:k]
            out[i, idx] = -out[i, idx]
        return out

    def training_step(
        self,
        clean: torch.Tensor,
        max_flips: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Train decoder to map noisy codes back to clean ±1 targets (MSE on soft)."""
        e = self.capability_e if max_flips is None else max_flips
        # Prefer real ADCMH codes; also support zero-codeword protocol from paper
        target = clean.sign().clamp(min=-1.0)
        noisy = self.flip_bits(target, e)
        pred = self.forward(noisy)
        loss = F.mse_loss(pred, target)
        # Optional hard accuracy
        with torch.no_grad():
            acc = (pred.sign().clamp(min=-1.0) == target).float().mean()
        return loss, {"necd_loss": float(loss.detach()), "necd_bit_acc": float(acc)}
