#!/usr/bin/env python3
import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

import socket
import os
import sys
import time

PIPE_IN = ".test-transformers.in"
PIPE_OUT = ".test-transformers.out"

def start_server():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m')
    parser.add_argument('-f')
    parser.add_argument('-s',type=int)
    parser.add_argument('--max-out-tokens', type=int)
    parser.add_argument('--ctx-size', type=int)
    parser.add_argument('--repeat-penalty',type=float)
    parser.add_argument('--n-gpu-layers',type=int)
    args = parser.parse_args()

    if os.path.exists(PIPE_IN):
        os.remove(PIPE_IN)
    os.mkfifo(PIPE_IN)
    if os.path.exists(PIPE_OUT):
        os.remove(PIPE_OUT)
    os.mkfifo(PIPE_OUT)

    if args.n_gpu_layers is None:
        device = 'cpu'
    else:
        device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(args.m)
    model = AutoModelForCausalLM.from_pretrained(args.m, device_map=device,
                                                torch_dtype=torch.bfloat16)
    try:
        model = model.to_bettertransformer()
    except Exception as e:
        print(e)
    print("B: Server started, waiting for connections...") 
    
    while True:
        with open(PIPE_IN, 'rb') as fifo:
            buf = fifo.read()
        
        if buf !=b'hello':
            cmdlines = json.loads(buf)
            client_args = parser.parse_args(cmdlines[1:])

            with open(client_args.f, 'r') as fp:
                prompt = fp.read()
            input_args = tokenizer(prompt, return_tensors="pt").to(device)

            print('prompt', prompt)
            streamer = TextStreamer(tokenizer, skip_prompt=True)

            out = model.generate(**input_args, streamer=streamer,
                                        pad_token_id=tokenizer.eos_token_id, 
                                        #temperature=0,
                                        top_p=0.8,
                                        max_new_tokens=client_args.max_out_tokens,
                                        repetition_penalty=client_args.repeat_penalty)
            
            #tokenizer = AutoTokenizer.from_pretrained("your-model-name")
            #input_tokens = tokenizer(prompt)['input_ids']
            
            in_tokens = input_args['input_ids'][0].cpu().numpy().tolist()
            json.dump({'i': in_tokens,
                       'o': out[0][len(in_tokens):].cpu().numpy().tolist()},
                       open('groundtruth.json', 'w'))
            
        with open(PIPE_OUT, 'wb') as fifo:
            fifo.write(b'OK')
            

if __name__ == "__main__":
    start_server()