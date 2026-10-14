# Implementation Plan — DNDCMH

## Paper

Error-Corrected Margin-Based Deep Cross-Modal Hashing for Facial Image Retrieval (TBIOM 2020).

## Goal

Rebuild **DNDCMH** = **ADCMH** (Image CNN + Attribute MLP, margin DLL) + **NECD** (neural ECC decoder) for attribute→face Hamming retrieval on CelebA.

## Figure inventory

| Paper fig | Type | Our artifact |
|-----------|------|--------------|
| Fig. 6 ranking / MAP bars | bar MAP by arity | `map_by_arity.svg` |
| Training curves (implied) | line loss | `train_loss.svg` |
| Fig. 9–11 P-R | deferred (subset) | optional later |

## Stages

1. CelebA attr loader + ADCMH (ResNet18 + MLP)
2. DLL + quantization + balance + align losses
3. NECD residual soft decoder (simplified vs Tanner BP)
4. MAP/NDCG single/double/triple queries
5. Smoke → size gate → full train → report → publish

## Data

- Required: CelebA (`projects/datasets/celeba`) — attrs + aligned images
- Deferred: LFW attrs, YTF (missing) — see DEVIATIONS.md
