#!/usr/bin/env bash
# Serve YannQi/R-4B through vLLM's OpenAI-compatible server.
# Extra flags pass straight through, e.g. `scripts/serve_r4b.sh --tensor-parallel-size 2`.
# --language-model-only skips the vision tower (~0.8GB), which System-One does not use yet.
set -euo pipefail
export HF_HOME="${HF_HOME:-$(dirname "$0")/../data/cache}"
exec vllm serve YannQi/R-4B \
  --served-model-name r4b \
  --trust-remote-code \
  --language-model-only \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.93}" \
  --port "${VLLM_PORT:-8000}" \
  "$@"
