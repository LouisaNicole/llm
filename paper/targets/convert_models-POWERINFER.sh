#!/bin/bash

prog=./llama.cpp

#MODEL_BASE=Llama-2-7b-chat
#MODEL_VERD=meta

#MODEL_BASE=Llama-3.1-8b-instruct
#MODEL_VERD=meta

#MODEL_BASE=Llama-3.2-3b-instruction
#MODEL_VERD=meta

#MODEL_BASE=Phi-3.5-mini-instruct
#MODEL_VERD=microsoft

#MODEL_BASE=gemma-2-9b-it
#MODEL_VERD=google

#MODEL_BASE=Mistral-7b-instruct
#MODEL_VERD=mistral

#MODEL_BASE=Nemotron-mini-4b-instruct
#MODEL_VERD=nvidia

#MODEL_BASE=vicuna-13b-v1.5
#MODEL_VERD=lmsys

#MODEL_BASE=Falcon3-10B-Instruct
#MODEL_VERD=tiiuae

#MODEL_BASE=Llama-3.2-1B-Instruct
#MODEL_VERD=meta

MODEL_BASE=Bamboo-DPO-v0_1
MODEL_VERD=PowerInfer

QUANT=Q4_K_M
prog=./PowerInfer

MODEL=$MODEL_VERD/$MODEL_BASE

MODEL_DIR=/media/ain/新加卷/hf-models
MODEL_OUTDIR=/media/ain/Win11/prj/models
OUTFILE=$MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE.gguf

mkdir -p $MODEL_OUTDIR/$MODEL

set -e

if true; then
python3 ./PowerInfer/convert-hf-to-gguf.py \
    --outfile $OUTFILE \
    --outtype f16 \
    $MODEL_DIR/$MODEL
fi

echo Out "$OUTFILE"
