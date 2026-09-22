#!/usr/bin/env bash
# Serve an R-4B GGUF through llama.cpp's llama-server (text only; the mmproj vision file is not needed).
# R4B_GGUF_FILE picks the quant (default Q4_K_M). Extra flags pass straight through to llama-server.
set -euo pipefail
export HF_HOME="${HF_HOME:-$(dirname "$0")/../data/cache}"
MODEL=$(hf download infil00p/R-4B-GGUF "${R4B_GGUF_FILE:-R-4B-Q4_K_M.gguf}" --quiet)
exec llama-server -m "$MODEL" \
  --port "${LLAMA_PORT:-8081}" \
  --n-gpu-layers "${LLAMA_GPU_LAYERS:-99}" \
  --ctx-size "${LLAMA_CTX:-32768}" \
  --parallel "${LLAMA_PARALLEL:-4}" \
  "$@"
