#!/usr/bin/env bash
# Swap the model served by the vLLM container.
#
# Usage:
#   scripts/vllm_switch_model.sh <hf-model-id>
#
# The target must be accessible to HF_TOKEN (see .env) and fit in the
# configured VRAM share (VLLM_GPU_MEM_UTIL). Weights are cached in
# ~/.cache/huggingface via the compose volume mount, so repeat swaps to
# a previously-loaded model skip the download step.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <hf-model-id>"
  echo "Example: $0 Qwen/Qwen2-7B-Instruct"
  exit 1
fi

NEW_MODEL="$1"

docker compose stop vllm >/dev/null 2>&1 || true
docker compose rm -f vllm >/dev/null 2>&1 || true

VLLM_MODEL="${NEW_MODEL}" docker compose up -d vllm

echo "vLLM restarting with model: ${NEW_MODEL}"
echo "Tail logs with: docker compose logs -f vllm"
echo "Wait for 'Application startup complete' before running experiments."
