#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/check_dataset_size.sh ../../datasets/celeba || {
  code=$?
  if [[ $code -eq 3 ]]; then
    echo "Size gate: skip full train (>=5 GiB)"
    exit 3
  fi
  exit "$code"
}
bash scripts/train.sh configs/default.yaml
bash scripts/eval.sh configs/default.yaml
bash scripts/report.sh configs/default.yaml
echo "Full train + eval + report done"
