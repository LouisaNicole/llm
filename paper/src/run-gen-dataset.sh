#!/bin/bash

N_MAX_PROMPT_TOKENS_PER_DATASET=50000 # to get 500,000 tokens in rawtrain

GENERATED_DIR=../dataset/generated
mkdir -p $GENERATED_DIR
REF_TOKENIZER_MODEL=../thirdparty_models/meta/Llama-3.1-8b-instruct/ggml-Llama-3.1-8b-instruct.gguf

set -e
make -j gen-dataset

gen() {
	# @params COLLECTION
	OUT_PATHNAME=${GENERATED_DIR}/$1_${N_MAX_PROMPT_TOKENS_PER_DATASET}

	./gen-dataset \
		../dataset ${OUT_PATHNAME}.json \
		$N_MAX_PROMPT_TOKENS_PER_DATASET \
		$REF_TOKENIZER_MODEL \
		123 \
		$1 \
		2>&1 | tee ${OUT_PATHNAME}.log
}

gen natural-language
