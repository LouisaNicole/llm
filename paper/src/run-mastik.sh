#!/bin/bash
set -e
make exp_mastik
./exp_mastik --llm llama-2-7b-chat  --hw "Intel 13900K" --rounds 1000 --targets 32768