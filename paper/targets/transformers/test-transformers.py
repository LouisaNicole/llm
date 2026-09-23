#!/usr/bin/env python3
import socket
import json
import sys

PIPE_IN = ".test-transformers.in"
PIPE_OUT = ".test-transformers.out"

def main():
    with open(PIPE_IN, 'wb') as fifo:
        fifo.write(json.dumps(sys.argv).encode())
    print('cmd sent')

    # wait for it
    with open(PIPE_OUT, 'rb') as fifo:
        result = fifo.read()
    assert result == b'OK'



if '__main__' == __name__:
    main()


