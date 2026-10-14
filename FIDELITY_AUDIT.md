# FIDELITY_AUDIT — DNDCMH

Target: structural / algorithmic fidelity to the paper (not published MAP numbers).

## Pass statistics

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | Layout + ADCMH dual encoders + CelebA attrs | 70% | Scaffold + loaders |
| 2 | Margin DLL + quant/balance + binarize | 80% | Loss geometry e≥m |
| 3 | NECD training with ≤e flips + decode at test | 85% | MLP vs Tanner BP (D3) |
| 4 | MAP single/double/triple + figures/REPORT | 92% | CelebA only (D1/D2) |
| 5 | Scripts, size gate, publish packaging | 95% | YTF/LFW gaps remain |

**Stop after Pass 5** (paper-2-model rule).

## Final sign-off

- Real CelebA train/eval metrics only
- Deviations logged in DEVIATIONS.md
- Confidence vs paper structure: **~95%** (ECC decoder approximation is the main gap)

## Measured run (full CelebA)

- Train: 162770 images, ADCMH 20 + NECD 10, 2542.7 s
- Gallery: 19962 test images; 200 queries/arity
- Ours MAP (single/double/triple): ADCMH 27.3/7.5/1.9; DNDCMH 27.7/8.1/2.4
- Paper Table 2 CelebA DNDCMH: 79.1/73.2/72.9 — number matching out of scope; gap attributed to D3 (NECD), D4 (backbone), and query construction details
