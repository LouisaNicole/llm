#!/usr/bin/env python3
from gpt4all import GPT4All
import argparse
import json
from transformers import AutoTokenizer

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

    print('init')
    model = GPT4All(args.m, n_ctx=args.ctx_size,
                    device='cuda' if args.n_gpu_layers else 'cpu',
                    ngl=args.n_gpu_layers)

    with open(args.f, 'r') as fp:
        prompt = fp.read()

    print('runn')
    output = model.generate(prompt, max_tokens=args.max_out_tokens)
    print(output)

    llm = 'Phi-3.5-mini-instruct'
    assert args.m[-len(llm+'.gguf'):] == llm + '.gguf'
    encoder = load_tokenizer(f'/media/ain/新加卷/tokenizers/microsoft/{llm}')
    with open('groundtruth.json', 'w') as fp:
        fp.write(json.dumps({'i': encoder.encode(prompt), 'o': encoder.encode(output)}))