"""Margin-based Distance Logistic Loss (DLL) + quantization / balance terms."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _pairwise_hamming_soft(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Soft Hamming distance on tanh codes in [0, hash_bits]."""
    # (1 - u·v)/2 per dim summed → approx Hamming for ±1
    return 0.5 * (u.shape[-1] - (u * v).sum(dim=-1))


def dll_loss(img_z: torch.Tensor, attr_z: torch.Tensor, margin_m: float) -> torch.Tensor:
    """
    Matched pairs (same index) should have soft-Hamming < m;
    mismatched pairs (circular shift) should exceed m.
    Logistic on (d - m).
    """
    b = img_z.size(0)
    d_pos = _pairwise_hamming_soft(img_z, attr_z)
    # hard negatives: roll attributes
    attr_neg = torch.roll(attr_z, shifts=1, dims=0)
    d_neg = _pairwise_hamming_soft(img_z, attr_neg)
    # P(match) ~ sigmoid(-(d - m)); maximize for pos, minimize for neg
    loss_pos = F.softplus(d_pos - margin_m)
    loss_neg = F.softplus(margin_m - d_neg)
    return (loss_pos + loss_neg).mean()


def quantization_loss(z: torch.Tensor) -> torch.Tensor:
    return (z.abs() - 1.0).pow(2).mean()


def balance_loss(z: torch.Tensor) -> torch.Tensor:
    """Encourage zero-mean bits (bit balance)."""
    return z.mean(dim=0).pow(2).mean()


def adcmh_loss(
    img_z: torch.Tensor,
    attr_z: torch.Tensor,
    margin_m: float,
    theta: float = 1.0,
    lambda_reg: float = 0.1,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    dll = dll_loss(img_z, attr_z, margin_m)
    q = quantization_loss(img_z) + quantization_loss(attr_z)
    bal = balance_loss(img_z) + balance_loss(attr_z)
    # cross-modal alignment of continuous codes
    align = F.mse_loss(img_z, attr_z)
    total = dll + theta * q + lambda_reg * bal + gamma * align
    return total, {
        "dll": float(dll.detach()),
        "quant": float(q.detach()),
        "balance": float(bal.detach()),
        "align": float(align.detach()),
        "total": float(total.detach()),
    }
