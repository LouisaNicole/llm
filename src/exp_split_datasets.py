import os
os.environ['NOT_INIT_OPENAI'] = '1'
from experiment_common import *

parser = argparse.ArgumentParser()
parser.add_argument('--ds-collection-id', type=str, default='natural-language_50000', help="")
args = parser.parse_args()


GEN_DS_TRAIN_FN=prompting_ds_train_fn(args.ds_collection_id)
GEN_DS_VAL_FN=prompting_ds_val_fn(args.ds_collection_id)
GEN_DS_TEST_FN=prompting_ds_test_fn(args.ds_collection_id)

def split_train_valid_test(gen_ds, f_train,f_val,f_test):
    ret_train = {'datasets': []}
    ret_test = {'datasets': []}
    ret_val = {'datasets': []}
    for ds in gen_ds['datasets']:
        # trainlist : 60%
        # tv = vallist + testlist : 40%
        trainlist, tv = train_test_split(ds['samples'], test_size=0.4, random_state=42)
        # vallist : 20%
        # testlist : 20%
        vallist, testlist = train_test_split(tv, test_size=0.5, random_state=42)

        ret_train['datasets'].append({'name': ds['name'],
                                        'samples':trainlist})
        ret_val['datasets'].append({ 'name': ds['name'],
                                   'samples':vallist,})
        ret_test['datasets'].append({'name': ds['name'],
                                    'samples':testlist})
    
    json.dump(ret_train, open(f_train, 'w'))
    json.dump(ret_val, open(f_val, 'w'))
    json.dump(ret_test, open(f_test, 'w'))

gen_ds = json.load(open(prompting_ds_fn(args.ds_collection_id),'r'))

split_train_valid_test(gen_ds,
    GEN_DS_TRAIN_FN,
    GEN_DS_VAL_FN,
    GEN_DS_TEST_FN)