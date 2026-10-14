# Error-Corrected Margin-Based Deep Cross-Modal Hashing for Facial Image Retrieval

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch rebuild of **DNDCMH** (ADCMH + NECD) for attribute→face Hamming retrieval (TBIOM 2020).

**Repository:** https://github.com/biometric-community/error-corrected-margin-based-deep-cross-modal-hashing-for-facial-image

Monorepo path: `projects/papers/error-corrected-margin-based-deep-cross-modal-hashing-for-facial-image/`

## Setup

```bash
bash scripts/setup_env.sh
```

Uses the shared monorepo `.venv` (no per-project venv).

## Data

Point configs at CelebA:

- `../../datasets/celeba/extracted/celeba` (`img_align_celeba/`, `list_attr_celeba.txt`, `list_eval_partition.txt`)

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh          # size-gated (<5 GiB)
bash scripts/eval.sh configs/default.yaml
bash scripts/report.sh configs/default.yaml
```

See **[REPORT.md](./REPORT.md)** for metrics and figures.

## Predict

```bash
bash scripts/predict.sh configs/default.yaml -- --image /path/to/face.jpg --use-necd
```

## Layout

- `dndcmh/` — models, losses, metrics, train/eval/predict/report
- `configs/` — `default.yaml` (full), `smoke.yaml`
- `scripts/` — shell entrypoints
- `DEVIATIONS.md`, `FIDELITY_AUDIT.md`, `IMPLEMENTATION_PLAN.md`

## Citation

Taherkhani et al., “Error-Corrected Margin-Based Deep Cross-Modal Hashing for Facial Image Retrieval,” *IEEE TBIOM*, 2020. DOI: [10.1109/TBIOM.2020.2983467](https://doi.org/10.1109/TBIOM.2020.2983467)
