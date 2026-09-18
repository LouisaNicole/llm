import json

def jsonl_read(fn):
    with open(fn, 'r') as fp:
        return [json.loads(line) for line in fp.readlines()]

def jsonl_write(fn, array):
    assert type(array) == list
    with open(fn, 'w') as fp:
        for d in array:
            fp.write(json.dumps(d) + '\n')
