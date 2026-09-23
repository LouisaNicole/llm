#!/bin/bash

HW="Intel 13900K"
LLM=Llama-2-7b-chat
#LLM=Llama-3.1-8b-instruct
CHECKPOINT_FREQ=50
MAX_OUT_TOKENS=250

LLAMA_CPP_DIR=../targets/llama.cpp/llama-cli
LLM_WEIGHT_DIR=/media/ain/Win11/prj/models #../../models
DATASET_PATHNAME="./results/llama.cpp/Intel 13900K/${LLM}_natural-language_166666_test.json"
#DATASET_PATHNAME=./generated/synth_25000.json
PYTHON3=python
DEBUG=n

GDB=
if [[ "x$DEBUG" == "xy" ]]; then
  GDB="gdb -ex r --args "
fi
set -e

case $LLM in
  Llama-*)
    GGUF_FILE="$LLM_WEIGHT_DIR/meta/$LLM/ggml-$LLM.gguf"
    ;;
  *)
    echo unknown model $LLM
    exit 1
    ;;
esac

case "$HW" in
  "Intel 1240P")
    ROUNDS=10000
    TARGETS=32000
    EXTRA_FLAGS=
    ;;
  
  "Intel 13900K")
    ROUNDS=1000 #10000
    TARGETS=32000
    EXTRA_FLAGS=
    ;;
    
  "Intel Xeon Platinum 8255C")
    ROUNDS=10000
    TARGETS=256
    EXTRA_FLAGS=--no-setpriotiry # the cloud platform does not allow set scheduler 
    ;;

  *)
    echo Unknown target $HW
    exit 1
    ;;
esac

RESULTS_DIR=./results/llama.cpp/$HW/cache_trace.csv

make -j collect-cache-trace
$GDB ./collect-cache-trace "Intel 13900K" \
  $LLM \
  "$LLAMA_CPP_DIR" \
  "$GGUF_FILE" \
  "$DATASET_PATHNAME" \
  "$RESULTS_DIR" \
  $MAX_OUT_TOKENS
