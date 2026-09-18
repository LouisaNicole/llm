#!/usr/bin/env python3
import argparse
import json
from transformers import AutoTokenizer
from openai import OpenAI

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

    llm_vendor = 'mistral'
    llm = 'Mistral-7b-instruct'
    encoder = load_tokenizer(f'/media/ain/新加卷/tokenizers/{llm_vendor}/{llm}')
    
    with open(args.f, 'r') as fp:
        prompt = fp.read()

    # Point to the local server
    client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="lm-studio")

    completion = client.chat.completions.create(
        model=f"{llm}:latest",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    output = completion.choices[0].message.content
    print(output)

    with open('groundtruth.json', 'w') as fp:
        fp.write(json.dumps({'i': encoder.encode(prompt), 'o': encoder.encode(output)}))