
#!/usr/bin/env python3
import os
PIPE_IN = ".test-transformers.in"
PIPE_OUT = ".test-transformers.out"

def main():
    with open(PIPE_IN, 'wb') as fifo:
        fifo.write('hello'.encode())
    print('cmd hello')
    # wait for it
    with open(PIPE_OUT, 'rb') as fifo:
        result = fifo.read()
    assert result == b'OK', result

if '__main__' == __name__:
    main()