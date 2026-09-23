#!/bin/bash

# Usage: run-collect.sh <victim machine> <victim operating system> <parts>

HW=$1
OPERATIONSYS=$2
PARTS=$3

# Check the args
case $HW in
    "Intel 14900K"|"Intel 13900K"|"Intel 12700KF")
      echo Specified machine $HW
    ;;
    *)
      echo Unknown machine $HW
    ;;
esac

case $OPERATIONSYS in
  'ubuntu22_04'|'debian12'|"windows11")
    echo Specified operating system $OPERATIONSYS
  ;;
  *)
    echo Unknown operating system $HW
  ;;
esac
echo Specified parts: $PARTS

CHECKPOINT_FREQ=50
MAX_OUT_TOKENS=250

LLM_WEIGHT_DIR=../thirdparty_models
LLM_HF_WEIGHT_DIR=../thirdparty_models
DATASET_DIR=../dataset/generated
DEBUG=n

GDB=
if [[ "x$DEBUG" == "xy" ]]; then
  GDB="gdb -ex r --args "
fi


colelct_cache_traces() {
  local VICTIM_FRAMEWORK=$1
  local DEVICE=$2
  local LLM=$3
  local DATASET_ID=$4
  local DATASET_PATHNAME=$DATASET_DIR/$DATASET_ID
  local INCREMENTAL=$5
  local SCAN_PREFILL_BS=$6
  local EMBD_QUANT=$7
  if [ -n "$EMBD_QUANT" ]; then
    echo "******* Embd: $EMBD_QUANT *******"
  else
    EMBD_QUANT=F16
  fi
  
  echo === $LLM : $DATASET_PATHNAME ===
  
  case $LLM in
    gemma-*)
      LLM_VENDOR=google
      ;;
    Llama-*)
      LLM_VENDOR=meta
      ;;
    Phi-*)
      LLM_VENDOR=microsoft
      ;;
    Mistral-*)
      LLM_VENDOR=mistral
      ;;
    Nemotron-*)
      LLM_VENDOR=nvidia
      ;;
    vicuna-*)
      LLM_VENDOR=lmsys
      ;;
    Falcon*)
      LLM_VENDOR=tiiuae
      ;;
    *)
      echo unknown model $LLM
      exit 1
      ;;
  esac

  SKIP_L1_HIT=

  if [ "$EMBD_QUANT" == "Q8_0" ]; then
    SKIP_L1_HIT=y
  fi
  
  case $VICTIM_FRAMEWORK in
    llama.cpp)
      PROG_DIR=../targets/llama.cpp/llama-cli
      if [ "$EMBD_QUANT" != "F16" ]; then
        MODEL_FILE="$LLM_WEIGHT_DIR/$LLM_VENDOR/$LLM/ggml-$LLM-embd$EMBD_QUANT.gguf"
      else
        MODEL_FILE="$LLM_WEIGHT_DIR/$LLM_VENDOR/$LLM/ggml-$LLM.gguf"
      fi
      MODEL_TYPE=gguf
    ;;
    transformers)
      PROG_DIR='../targets/transformers/test-transformers.py'
      MODEL_FILE="$LLM_HF_WEIGHT_DIR/$LLM_VENDOR/$LLM"
      # format:tokenizer(gguf):tokenizer(safetensors)
      case $LLM in
        Llama-3.2-1b*)
          SAFETENSORS=model.safetensors
          ;;
        Phi-3.5-mini*)
          SAFETENSORS=model-00001-of-00002.safetensors
          ;;
        Falcon3-1B*)
          SAFETENSORS=model.safetensors
        ;;
        *)
          echo unknown model $LLM for safetensors
          exit 1
          ;;
      esac
      MODEL_TYPE=safetensors:$LLM_WEIGHT_DIR/$LLM_VENDOR/$LLM/ggml-$LLM.gguf:$SAFETENSORS

      if [ $DEVICE == 'gpu' ]; then
        ngl="--n-gpu-layers=33"
      else
        ngl=""
      fi

      source ../targets/transformers/venv/bin/activate
      # start daemon
      echo "../targets/transformers/daemon.py \
            -m \"$MODEL_FILE\" \
            -f ".tmp.txt" \
            -s 123 \
            --max-out-tokens $MAX_OUT_TOKENS \
            --ctx-size 4096 \
            --repeat-penalty 1.01 \
            $ngl"
      ../targets/transformers/daemon.py \
            -m "$MODEL_FILE" \
            -f ".tmp.txt" \
            -s 123 \
            --max-out-tokens $MAX_OUT_TOKENS \
            --ctx-size 4096 \
            --repeat-penalty 1.01 \
            $ngl &
      daemon_pid=$!
      cleanup() {
        kill $daemon_pid
      }
      sleep 2 # wait for the creation of the pipes
      trap cleanup EXIT
      python3 ../targets/transformers/hello.py
    ;;
    bitnet)
      PROG_DIR=../targets/BitNet/test.py
      MODEL_FILE="$LLM_WEIGHT_DIR/$LLM_VENDOR/$LLM/ggml-model-i2_s.gguf"
      MODEL_TYPE=gguf
    ;;
    gpt4all)
      PROG_DIR=../targets/GPT4All/test.py
      MODEL_FILE="$LLM_WEIGHT_DIR/$LLM_VENDOR/$LLM/ggml-$LLM.gguf"
      MODEL_TYPE=gguf
      source ../targets/GPT4All/venv/bin/activate
      http_proxy=
      https_proxy=
      LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../targets/GPT4All/gpt4all/lib
    ;;
    lmstudio)
      PROG_DIR=../targets/lmstudio/test.py
      MODEL_FILE="$LLM_WEIGHT_DIR/$LLM_VENDOR/$LLM/ggml-$LLM.gguf"
      MODEL_TYPE=gguf
      source ../targets/lmstudio/venv/bin/activate
    ;;
    ollama)
      PROG_DIR=../targets/Ollama/test.py
      MODEL_FILE="/home/tuser/.ollama/models/blobs/sha256-4404b10c5a784b1f47c2a915c1f6f5f6678d2ab1a0efddd271f4c2e8e367affb" # manually set it
      MODEL_TYPE=gguf
      source ../targets/Ollama/venv/bin/activate
    ;;
    local-ai)
      PROG_DIR=../targets/LocalAI/test.py
      MODEL_FILE="$LLM_WEIGHT_DIR/$LLM_VENDOR/$LLM/ggml-$LLM.gguf"
      MODEL_TYPE=gguf
      sudo local-ai --models-path=../targets/LocalAI/models &
      server_pid=$!
      cleanup() {
        kill $server_pid
      }
      sleep 10 # wait for the creation of the server
      trap cleanup EXIT
      source ../targets/lmstudio/venv/bin/activate # shared python venv
    ;;
    llamafile)
      PROG_DIR=../targets/llamafile/test.py
      MODEL_FILE="$LLM_WEIGHT_DIR/$LLM_VENDOR/$LLM/ggml-$LLM.gguf"
      MODEL_TYPE=gguf
      gpu_arg=
      if [ $DEVICE == "gpu" ]; then
        gpu_arg="-ngl 9999"
      fi
      echo $gpu_arg -m $MODEL_FILE --server --v2 --listen 0.0.0.0:8888
      llamafile $gpu_arg -m $MODEL_FILE --server --v2 --listen 0.0.0.0:8888 &
      server_pid=$!
      cleanup() {
        kill $server_pid
      }
      sleep 10 # wait for the creation of the server
      trap cleanup EXIT
      source ../targets/lmstudio/venv/bin/activate # shared python venv
    ;;
    powerinfer)
      PROG_DIR=../targets/PowerInfer/build/bin/main
      MODEL_FILE="$LLM_WEIGHT_DIR/$LLM_VENDOR/$LLM/ggml-$LLM.gguf"
      MODEL_TYPE=gguf
      SKIP_L1_HIT=y
    ;;
    ipex-llm)
      PROG_DIR=../targets/ipex-llm/test.py
      MODEL_FILE="$LLM_WEIGHT_DIR/$LLM_VENDOR/$LLM/ggml-$LLM.gguf"
      MODEL_TYPE=gguf
      source /opt/intel/oneapi/setvars.sh --force
      export ONEAPI_DEVICE_SELECTOR=ext_oneapi_cuda:gpu
      LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../targets/ipex-llm/llama-cpp/
      source ../targets/lmstudio/venv/bin/activate # shared python venv

    ;;
    koboldcpp)
      PROG_DIR=../targets/koboldcpp/test.py
      MODEL_FILE="$LLM_WEIGHT_DIR/$LLM_VENDOR/$LLM/ggml-$LLM.gguf"
      MODEL_TYPE=gguf
      source ../targets/lmstudio/venv/bin/activate # shared python venv
    ;;
    *)
      echo unknown victim framework $VICTIM_FRAMEWORK
      exit 1
      ;;
  esac

  RESULTS_DIR=../results/$DEVICE/$OPERATIONSYS/$VICTIM_FRAMEWORK

  echo Using $GGUF_FILE

  if [[ "x$SCAN_PREFILL_BS" == "xy" ]]; then
	  prefill_batch_sizes=(128 64 32)
  else
	  prefill_batch_sizes=(default)
  fi
  for pbs in "${prefill_batch_sizes[@]}"
  do
	  $GDB./collect "$HW" \
		$LLM \
    $DEVICE \
		"$PROG_DIR" \
		"$MODEL_FILE" \
    "$MODEL_TYPE" \
		"$DATASET_PATHNAME" \
		"$RESULTS_DIR" \
		2400 \
		$INCREMENTAL \
		$CHECKPOINT_FREQ \
		$MAX_OUT_TOKENS \
		$pbs \
    $EMBD_QUANT \
    $SKIP_L1_HIT
  done

}

set -e
make -j collect

#######

run_llamacpp_micro() {
  DS_COLLECTION_ID=natural-language_50000_micro
  colelct_cache_traces "llama.cpp" $1 "Mistral-7b-instruct" "${DS_COLLECTION_ID}_test.json" new n
}

run_llamacpp_micro_quant() {
  DS_COLLECTION_ID=natural-language_50000_micro
  embd_quants=(Q8_0 BF16 F32)
  for quant in "${embd_quants[@]}"
  do
    colelct_cache_traces "llama.cpp" $1 "Mistral-7b-instruct" "${DS_COLLECTION_ID}_test.json" new n $quant
  done
}

run_bitnet_micro() {
  DS_COLLECTION_ID=natural-language_50000_micro
  colelct_cache_traces "bitnet" $1 "Falcon3-7B-Instruct-1.58bit" "${DS_COLLECTION_ID}_test.json" new n  
}

run_gpt4all_micro() {
  DS_COLLECTION_ID=natural-language_50000_micro
  colelct_cache_traces "gpt4all" $1 "Phi-3.5-mini-instruct" "${DS_COLLECTION_ID}_test.json" new n  
}

run_lmstudio_micro() {
  DS_COLLECTION_ID=natural-language_50000_micro
  colelct_cache_traces "lmstudio" $1 "Mistral-7b-instruct" "${DS_COLLECTION_ID}_test.json" new n  
}

run_ollama_micro() {
  DS_COLLECTION_ID=natural-language_50000_micro
  colelct_cache_traces "ollama" $1 "Mistral-7b-instruct" "${DS_COLLECTION_ID}_test.json" new n  
}

run_powerinfer_micro() {
  DS_COLLECTION_ID=natural-language_50000_micro
  colelct_cache_traces "powerinfer" $1 "Mistral-7b-instruct" "${DS_COLLECTION_ID}_test.json" new n  
}

run_transformers_micro() {
  DS_COLLECTION_ID=natural-language_50000_micro
  colelct_cache_traces "transformers" $1 "Falcon3-1B-Instruct" "${DS_COLLECTION_ID}_test.json" new n  
}

run_local_ai_micro() {
  DS_COLLECTION_ID=natural-language_50000_micro
  colelct_cache_traces "local-ai" $1 "Mistral-7b-instruct" "${DS_COLLECTION_ID}_test.json" new n  
}

run_llamafile_micro() {
  DS_COLLECTION_ID=natural-language_50000_micro
  colelct_cache_traces "llamafile" $1 "Mistral-7b-instruct" "${DS_COLLECTION_ID}_test.json" new n  
}

run_ipex_llm_micro() {
  DS_COLLECTION_ID=natural-language_50000_micro
  colelct_cache_traces "ipex-llm" $1 "Mistral-7b-instruct" "${DS_COLLECTION_ID}_test.json" new n  
}

run_koboldcpp_micro() {
  DS_COLLECTION_ID=natural-language_50000_micro
  colelct_cache_traces "koboldcpp" $1 "Mistral-7b-instruct" "${DS_COLLECTION_ID}_test.json" new n  
}


wait_keypress() {
  read -n 1 -s -r -p "Please press any key to continue if you have completed the above steps..."
}

######

if [[ "$PARTS" == *"framework_eval"* ]]; then
  # for RQ4
  run_llamacpp_micro cpu
  run_llamacpp_micro gpu

  run_bitnet_micro cpu
  run_bitnet_micro gpu

  run_gpt4all_micro cpu
  run_gpt4all_micro gpu

  run_powerinfer_micro cpu
  run_powerinfer_micro gpu

  run_transformers_micro cpu
  run_transformers_micro gpu

  run_local_ai_micro cpu
  run_local_ai_micro gpu

  run_llamafile_micro cpu
  run_llamafile_micro gpu 

  run_llamacpp_micro_quant cpu
  run_llamacpp_micro_quant gpu

  ####

  echo ! you should manaually run LM Studio service and set the device in LM Studio GUI
  wait_keypress
  run_lmstudio_micro cpu
  run_lmstudio_micro gpu

  echo ! you should manaually run:
  echo     export OLLAMA_LLM_LIBRARY=cpu
  echo     ollama serve
  wait_keypress
  run_ollama_micro cpu

  echo ! manaually run:
  echo      pkill ollama
  echo      export OLLAMA_LLM_LIBRARY=
  echo      ollama serve
  wait_keypress
  run_ollama_micro gpu 

  echo ! manually configure and run the GUI
  wait_keypress
  run_koboldcpp_micro cpu
  run_koboldcpp_micro gpu

  echo ! manually run the server
  wait_keypress
  run_ipex_llm_micro cpu
  run_ipex_llm_micro gpu
fi

######

if [[ "$PARTS" == *"hardware_os_eval"* ]]; then
  # for RQ5
  run_llamacpp_micro cpu
  run_llamacpp_micro gpu
fi

######

if [[ "$PARTS" == *"embd_quant_eval"* ]]; then
  # for Appendix
  run_llamacpp_micro_quant cpu
  run_llamacpp_micro_quant gpu
fi

######

if [[ "$PARTS" == *"main_eval"* ]]; then
  # for RQ1~RQ3
  DS_COLLECTION_ID=natural-language_50000
  VICTIM_FRAMEWORK='llama.cpp'
  DEVICE='gpu'

  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Falcon3-10B-Instruct" "${DS_COLLECTION_ID}_test.json" new n
  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Falcon3-10B-Instruct" "${DS_COLLECTION_ID}_test.json" new y
  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Falcon3-10B-Instruct" "${DS_COLLECTION_ID}_val.json" new n

  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Llama-3.1-8b-instruct" "${DS_COLLECTION_ID}_test.json" new n
  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Llama-3.1-8b-instruct" "${DS_COLLECTION_ID}_train.json" new n
  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Llama-3.1-8b-instruct" "${DS_COLLECTION_ID}_test.json" new y
  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Llama-3.1-8b-instruct" "${DS_COLLECTION_ID}_val.json" new n

  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "gemma-2-9b-it" "${DS_COLLECTION_ID}_test.json" new n
  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "gemma-2-9b-it" "${DS_COLLECTION_ID}_test.json" new y
  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "gemma-2-9b-it" "${DS_COLLECTION_ID}_val.json" new n

  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Mistral-7b-instruct" "${DS_COLLECTION_ID}_test.json" new n
  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Mistral-7b-instruct" "${DS_COLLECTION_ID}_test.json" new y
  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Mistral-7b-instruct" "${DS_COLLECTION_ID}_val.json" new n

  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Phi-3.5-mini-instruct" "${DS_COLLECTION_ID}_train.json" new n
  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Phi-3.5-mini-instruct" "${DS_COLLECTION_ID}_val.json" new n
  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Phi-3.5-mini-instruct" "${DS_COLLECTION_ID}_test.json" new n
  colelct_cache_traces $VICTIM_FRAMEWORK $DEVICE "Phi-3.5-mini-instruct" "${DS_COLLECTION_ID}_test.json" new y
fi
