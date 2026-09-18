from constants import *
from hyperparam import get_map_new_model_name
from dynamicavg import DynamicAvg
import argparse
from tqdm import tqdm
import numpy as np
from myjsonl import *
import math
from tab_performance import get_metric, get_test_restored_file
from tab_common import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--train-ds-collection-id', type=str, default='natural-language_50000')
    parser.add_argument('--eval-ds-collection-id', type=str, default='natural-language_50000_micro')
    parser.add_argument('--hardware', type=str, default='Intel 13900K')
    args = parser.parse_args()

    atks = ['rr', 'pr']
    unit = '%'
    
    devices = ['cpu', 'gpu']
    operationsys = 'ubuntu22_04'

    frameworks = [
    'lmstudio', 
    'transformers',
    'ollama', 
    'llama.cpp',
    'gpt4all',
    'local-ai',
    'bitnet',
    'powerinfer',
    'ipex-llm', 
    'koboldcpp', 
    ]

    victim_llms = [
    'Mistral-7b-instruct', 
    'Falcon3-1B-Instruct',
    'Mistral-7b-instruct', 
    'Mistral-7b-instruct',
    'Phi-3.5-mini-instruct',
    'Mistral-7b-instruct', 
    'Falcon3-7B-Instruct-1.58bit',
    'Mistral-7b-instruct',
    'Mistral-7b-instruct', 
    'Mistral-7b-instruct', 
    ]
    

    gpu_failed_frm = ['transformers']

    founda_model='Llama-3.1-8B-Instruct'
    #founda_model='gpt-4o-mini-2024-07-18'
    
    use_valset = False
    prefill_bs = None # 32
    map_new_model_name = get_map_new_model_name(founda_model=founda_model,
                            train_ds_collection_id=args.train_ds_collection_id,
                            at=0,
                            hr=1,
                            hf=1,
                            rr_hl=1,
                            pr_hl=1,
                            ht=1,
                            batch_size=2,
                            rank=256,
                            alpha=256,
                            scale=10,
                            rr_lr = 8e-5,
                            pr_lr = 2e-4,
                            sigma = 0.08,
                            rr_prob = 0.2, # 0.2 0.4 0.8
                            pr_prob = 0.2,
                            rr_epoch = 3,
                            pr_epoch = 2
                        )

    def mkarray(x,y=None,z=None,w=None):
        ret = [None] * x
        if y:
            for i in range(x):
                ret[i] = [None] * y
                if z:
                    for j in range(y):
                        ret[i][j] = [None] *z
                        if w:
                            for k in range(z):
                                ret[i][j][k] = [None]*w
        return ret
    val = mkarray(len(frameworks), len(devices), len(atks))

    for frm_i, frm in enumerate(tqdm(frameworks)):
        llm_vendor,llm = VICTIM_LLM[get_llm_index(victim_llms[frm_i])]
        for ia, atk in enumerate(atks):
            for id, device in enumerate(devices):
                n_asr = 0
                n_atk = 0
                
                if device == 'gpu' and frm in gpu_failed_frm:
                    n_samples = len(jsonl_read(get_test_restored_file('cpu', operationsys, frm, args.hardware, llm,
                                                                      map_new_model_name[atk], args.eval_ds_collection_id, False, None)))
                    data = [0]*n_samples
                else:
                    data, n_samples = get_metric(map_new_model_name[atk],
                                                    device, operationsys, frm, args.hardware,
                                                    args.eval_ds_collection_id, llm, None, 'CosSim')
                v = np.mean(data)
                val[frm_i][id][ia] = (n_samples,v)


    # stats
    weighted_avg = mkarray(len(devices), len(atks))

    RFP_SCALE=10
    def reserve_fp(v): # for comparing integers instead of floats
        return round(v*100*RFP_SCALE) # round to 0.1% for unit=%, and 0.001 for unit=None
    
    for id, device in enumerate(devices):
        for ia, atk in enumerate(atks): 
            total_v = 0
            total_n_samples = 0
            for frm_i, _ in enumerate(frameworks):
                n_samples, v = val[frm_i][id][ia]
                if not math.isnan(v): # this happends when we have empty results
                    rfp_v = reserve_fp(v) 
                    total_v += v * n_samples
                    total_n_samples += n_samples
            weighted_avg[id][ia] = total_v / total_n_samples

    def disp_val(val, unit):
        if math.isnan(val):
            return '-', 0
        else:
            rfp_v = reserve_fp(val)
            if unit == '%':
                return f'{rfp_v/RFP_SCALE:.1f}', rfp_v
            elif unit is None:
                return f'{val:.3f}', rfp_v
            else:
                assert False

    # generate the table
    for frm_i, frm in enumerate(frameworks):
        print(f'{frameworks[frm_i]}: ', end='')

        for id, device in enumerate(devices):
            if id ==1:
                print(' ', end='')
            for ia, atk in enumerate(atks):
                print(f'& ', end='')
                v, rfp_v = disp_val(val[frm_i][id][ia][1], unit)
                print(v, end='')
            print('\t', end='')
        print(r'\\')

