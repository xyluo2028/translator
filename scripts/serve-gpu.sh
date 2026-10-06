#!/usr/bin/env bash
set -euo pipefail
TRANSLATOR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRANSLATOR_OLLAMA="$TRANSLATOR_ROOT/.local/ollama/bin/ollama"
if [[ ! -x "$TRANSLATOR_OLLAMA" ]]; then
    echo "Project-local Ollama is missing. See the RTX 4080 setup in README.md." >&2
    exit 1
fi
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS="$TRANSLATOR_ROOT/.local/ollama-models"
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_CONTEXT_LENGTH=2048
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
exec "$TRANSLATOR_OLLAMA" serve
