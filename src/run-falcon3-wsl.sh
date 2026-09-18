#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-$ROOT/src/thirdparty/Falcon3-1B-Instruct-f16.gguf}"
LLAMA_CLI="${LLAMA_CLI:-$ROOT/targets/llama.cpp/build/bin/llama-cli}"
DATASET="${DATASET:-$ROOT/dataset/generated/natural-language_50000_micro_test.json}"
RESULTS="${RESULTS:-$ROOT/results/cpu/ubuntu-wsl/llama.cpp}"
HARDWARE="${HARDWARE:-Intel 13900K}"

test -f "$MODEL" || { echo "model not found: $MODEL" >&2; exit 1; }
test -x "$LLAMA_CLI" || { echo "llama-cli not found or not executable: $LLAMA_CLI" >&2; exit 1; }
test -f "$DATASET" || { echo "dataset not found: $DATASET" >&2; exit 1; }

make -C "$ROOT/src" -j"$(nproc)" collect
mkdir -p "$RESULTS"

cd "$ROOT/src"
./collect "$HARDWARE" "Falcon3-1B-Instruct" cpu \
  "$LLAMA_CLI" "$MODEL" gguf "$DATASET" "$RESULTS" \
  0 new 1 64 default F16
