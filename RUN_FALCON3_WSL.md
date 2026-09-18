# Falcon3-1B-Instruct F16 WSL 小样本实验流程

当前流程用于课程版复现：

1. 跑 microbench，验证 cache timing 基础信号。
2. 用 Falcon3-1B-Instruct F16 跑小样本 victim inference。
3. 采集 cache trace，观察 token embedding 访问泄露现象。

这不是完整论文训练流程，不需要学生从零训练几天。

## 0. 进入项目

```bash
cd /mnt/e/luyimin/research/what_you_say
conda activate ikwys
```

确认关键文件存在：

```bash
ls src/thirdparty/Falcon3-1B-Instruct-f16.gguf
ls dataset/generated/natural-language_50000_micro_test.json
ls targets/llama.cpp/build/bin/llama-cli
```

## 1. 如果 llama-cli 不存在，先编译

```bash
cmake -S targets/llama.cpp -B targets/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF

cmake --build targets/llama.cpp/build --target llama-cli -j"$(nproc)"
```

## 2. 先测试 Falcon3 模型能不能推理

```bash
targets/llama.cpp/build/bin/llama-cli \
  -m src/thirdparty/Falcon3-1B-Instruct-f16.gguf \
  -p "<|user|>\nWhat is a CPU cache?\n<|assistant|>\n" \
  -n 32 \
  -c 512 \
  -t 4
```

如果能输出回答，说明模型和 llama.cpp 正常。

## 3. 跑 microbench

```bash
cd /mnt/e/luyimin/research/what_you_say/src
make exp_mastik
./exp_mastik --llm llama-2-7b-chat --hw "Intel 13900K" --rounds 1000 --targets 32768
```

## 4. 跑小样本 cache trace collection

推荐一键脚本：

```bash
cd /mnt/e/luyimin/research/what_you_say
bash src/run-falcon3-wsl.sh
```

等价手动命令：

```bash
cd /mnt/e/luyimin/research/what_you_say/src
make -j"$(nproc)" collect

mkdir -p ../results/cpu/ubuntu-wsl/llama.cpp

./collect "Intel 13900K" "Falcon3-1B-Instruct" cpu \
  ../targets/llama.cpp/build/bin/llama-cli \
  thirdparty/Falcon3-1B-Instruct-f16.gguf \
  gguf \
  ../dataset/generated/natural-language_50000_micro_test.json \
  ../results/cpu/ubuntu-wsl/llama.cpp \
  0 new 1 64 default F16
```

## 5. 查看结果

```bash
find /mnt/e/luyimin/research/what_you_say/results -type f | sort
```

常见输出路径类似：

```text
results/cpu/ubuntu-wsl/llama.cpp/Intel 13900K/Falcon3-1B-Instruct_natural-language_50000_micro_test.json
```

## 6. 如果要继续做恢复分析

`models.zip` 是攻击模型 checkpoint，用于跳过从零微调。它不能替代 Falcon3 victim LLM。

课堂建议：

1. 学生本机完成 microbench 和小样本 trace collection。
2. 教师用预训练 checkpoint 演示文本恢复。
3. 不要求学生本机完整训练 attack model。
