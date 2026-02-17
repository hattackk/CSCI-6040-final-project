#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-data/advbench/harmful_behaviors.csv}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
MAX_EXAMPLES="${MAX_EXAMPLES:-25}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found at ${DATASET_PATH}"
  echo "Set DATASET_PATH or place file at data/advbench/harmful_behaviors.csv"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"

python -m fitd_repro \
  --dataset-path "${DATASET_PATH}" \
  --backend hf \
  --model "${MODEL_NAME}" \
  --attack standard \
  --defense none \
  --max-examples "${MAX_EXAMPLES}" \
  --output-dir "results/${STAMP}_phase2_standard"

python -m fitd_repro \
  --dataset-path "${DATASET_PATH}" \
  --backend hf \
  --model "${MODEL_NAME}" \
  --attack fitd \
  --defense none \
  --max-examples "${MAX_EXAMPLES}" \
  --output-dir "results/${STAMP}_phase2_fitd"

python -m fitd_repro \
  --dataset-path "${DATASET_PATH}" \
  --backend hf \
  --model "${MODEL_NAME}" \
  --attack fitd \
  --defense vigilant \
  --max-examples "${MAX_EXAMPLES}" \
  --output-dir "results/${STAMP}_phase2_fitd_vigilant"

echo "Phase 2 runs complete. Outputs are under results/${STAMP}_*"
