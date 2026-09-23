#!/bin/bash
set -e
if true; then
cd ./llama.cpp
make -j llama-cli GGML_CUDA=1
cd ..
fi

if false; then
#cd ./PowerInfer/build
#make clean
#cd ..
cd ./PowerInfer
cmake -S . -B build -DLLAMA_CUBLAS=ON
cmake --build build --config Release
fi
