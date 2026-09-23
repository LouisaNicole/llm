from experiment_common import *
import random

parser = argparse.ArgumentParser()
parser.add_argument('--ds-collection-id', type=str, default='natural-language_50000')
parser.add_argument('--num-samples', type=int, default=20)
args = parser.parse_args()

gen_ds = json.load(open(prompting_ds_fn(args.ds_collection_id),'r'))

s = []
for d in gen_ds['datasets']:
    for v in d['samples']:
        s.append((d['name'], v))
sampled_ds = random.sample(s, k=args.num_samples)

ret_test = {}
for ds_name, sample in sampled_ds:
    if ds_name not in ret_test:
        ret_test[ds_name] = {'name': ds_name,
                        'samples': [sample]}
    else:
        ret_test[ds_name]['samples'].append(sample)

ret_test = {'datasets': [d for _,d in ret_test.items()]}

GEN_DS_TEST_FN=prompting_ds_test_fn(args.ds_collection_id + '_micro')
json.dump(ret_test, open(GEN_DS_TEST_FN, 'w'))
