#!/usr/bin/env python3

import os
import sys
import signal
import platform
import argparse
import subprocess

def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_inference():
    build_dir = "../targets/BitNet/" + "build"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")
    command = [
        f'{main_path}',
        '-m', args.m,
        '-f', str(args.f),
        '-s', str(args.s),
        '--ctx-size', str(args.ctx_size), 
        '--repeat-penalty', str(args.repeat_penalty),
    ]
    if args.n_gpu_layers:
        command.extend(['--n-gpu-layers', str(args.n_gpu_layers)])
    run_command(command)

def signal_handler(sig, frame):
    print("Ctrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser(description='Run inference')

    parser.add_argument('-m')
    parser.add_argument('-f')
    parser.add_argument('-s',type=int)
    parser.add_argument('--max-out-tokens', type=int)
    parser.add_argument('--ctx-size', type=int)
    parser.add_argument('--repeat-penalty',type=float)
    parser.add_argument('--n-gpu-layers',type=int)
    args = parser.parse_args()

    run_inference()