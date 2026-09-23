#!/bin/bash
set -e
# trace-common.c may have error dependency, so we clean the build every time
cd ./llama.cpp-trace-timeline
make clean
make -j llama-cli GGML_CUDA=1
cd ..

cd ./PowerInfer-trace-timeline/build
make clean
cd ..
cmake -S . -B build -DLLAMA_CUBLAS=ON
cmake --build build --config Release
