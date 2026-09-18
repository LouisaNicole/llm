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

MODEL_BASE=Mistral-7b-instruct
MODEL_VERD=mistral

#MODEL_BASE=Nemotron-mini-4b-instruct
#MODEL_VERD=nvidia

#MODEL_BASE=vicuna-13b-v1.5
#MODEL_VERD=lmsys

#MODEL_BASE=Falcon3-10B-Instruct
#MODEL_VERD=tiiuae

#MODEL_BASE=Llama-3.2-1B-Instruct
#MODEL_VERD=meta

#MODEL_BASE=Falcon3-1B-Instruct
#MODEL_VERD=tiiuae

#MODEL_BASE=DeepSeek-R1-Distill-Qwen-7B
#MODEL_VERD=deepseek-ai


QUANT=Q4_K_M
prog=./PowerInfer

MODEL=$MODEL_VERD/$MODEL_BASE

MODEL_DIR=/media/ain/新加卷1/hf-models
MODEL_OUTDIR=/home/ain/prj/models
OUTFILE=$MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE.gguf

mkdir -p $MODEL_OUTDIR/$MODEL

set -e

if false; then
python3 ./llama.cpp/convert_hf_to_gguf.py \
    --outfile $MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE-f16.gguf \
    --outtype f16 \
    $MODEL_DIR/$MODEL
fi

if false; then
./llama.cpp/llama-quantize --token-embedding-type F16 \
    $MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE-f16.gguf \
    $OUTFILE Q4_K_M
fi

if false; then
./llama.cpp/llama-quantize --token-embedding-type BF16 \
    $MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE-f16.gguf \
    $MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE-embdBF16.gguf Q4_K_M

./llama.cpp/llama-quantize --token-embedding-type F32 \
    $MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE-f16.gguf \
    $MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE-embdF32.gguf Q4_K_M


./llama.cpp/llama-quantize --token-embedding-type Q8_0 \
    $MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE-f16.gguf \
    $MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE-embdQ8_0.gguf Q4_K_M


fi

./llama.cpp/llama-quantize --token-embedding-type Q6_K \
    $MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE-f16.gguf \
    $MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE-embdQ6_K.gguf Q4_K_M


./llama.cpp/llama-quantize --token-embedding-type Q4_K \
    $MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE-f16.gguf \
    $MODEL_OUTDIR/$MODEL/ggml-$MODEL_BASE-embdQ4_K.gguf Q4_K_M

echo Out "$OUTFILE"
