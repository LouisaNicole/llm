#!/usr/bin/env python3
import argparse
import json
from transformers import AutoTokenizer
from openai import OpenAI
import subprocess

def load_tokenizer(tokenizer_path):
    return AutoTokenizer.from_pretrained(tokenizer_path,
                                        add_prefix_space =False # need to avoiding wronly remove the leading space for certain tokenizers
                                        )

def tokens2text(tokenzier, tokens):
    return tokenzier.decode(tokens, skip_special_tokens=True).encode()

def text2tokens(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference')

    parser.add_argument('-m')
    parser.add_argument('-f')
    parser.add_argument('-s',type=int)
    parser.add_argument('--max-out-tokens', type=int)
    parser.add_argument('--ctx-size', type=int)
    parser.add_argument('--repeat-penalty',type=float)
    parser.add_argument('--n-gpu-layers',type=int, default=0)
    args = parser.parse_args()

    if args.n_gpu_layers:
        device = 'gpu'
    else:
        device = 'cpu'

    llm_vendor = 'mistral'
    llm = 'Mistral-7b-instruct'
    encoder = load_tokenizer(f'/media/ain/新加卷1/tokenizers/{llm_vendor}/{llm}')
    
    with open(args.f, 'r') as fp:
        prompt = fp.read()

    cmdline = ["../targets/ipex-llm/llama-cpp/llama-cli" if device == 'cpu' else '../targets/llama.cpp/llama-cli',
               "-m", args.m,
                "-f", args.f,
                '-s', str(args.s),
                '--ctx-size', str(args.ctx_size),
                '--repeat-penalty', str(args.repeat_penalty),
                '--no-display-prompt']
    if device=='gpu':
        cmdline.extend(['--n-gpu-layers', str(args.n_gpu_layers)])
    
    process = subprocess.Popen(
        cmdline,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output, error = process.communicate()

    #print(output)

    with open('groundtruth.json', 'w') as fp:
        fp.write(json.dumps({'i': encoder.encode(prompt), 'o': encoder.encode(output)}))