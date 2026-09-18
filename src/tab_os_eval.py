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
    args = parser.parse_args()

    atks = ['rr', 'pr']
    unit = '%'
    
    hardware = 'Intel 13900K'
    devices = ['cpu', 'gpu']
    framework = 'llama.cpp'
    victim_llm = 'Mistral-7b-instruct'
    oss = ['ubuntu22_04', 'windows11', 'debian12']
    os_disp = ['Ubuntu 22.04', 'Windows 11', 'Debian 12']
    metrics = ['CosSim', 'ASR']

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
    val = mkarray(len(oss), len(devices), len(atks), len(metrics))

    llm_vendor,llm = VICTIM_LLM[get_llm_index(victim_llm)]

    for os_i, operationsys in enumerate(tqdm(oss)):
        for id, device in enumerate(devices):
            for ia, atk in enumerate(atks):
                for im, metric in enumerate(metrics):
                    n_asr = 0
                    n_atk = 0
                    
                    data, n_samples = get_metric(map_new_model_name[atk],
                                                device, operationsys, framework, hardware,
                                                args.eval_ds_collection_id, llm, None, metric)
                    v = np.mean(data)
                    val[os_i][id][ia][im] = (n_samples,v)
                
    RFP_SCALE=10
    def reserve_fp(v): # for comparing integers instead of floats
        return round(v*100*RFP_SCALE) # round to 0.1% for unit=%, and 0.001 for unit=None

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

    markdown = True
    if markdown:
        sep = '|'
        startch = '|'
        midch = ''
        endch = '|'
    else:
        sep = '&'
        startch = '\\midrule\n'
        midch = '& '
        endch = r'\\'

    # generate the table
    for os_i, operationsys in enumerate(oss):
        if operationsys != 'ubuntu22_04':
            if markdown:
                bf0, bf1 = '**', '**'
            else:
                bf0, bf1 = '\\textbf{', '}'
        else:
            bf0, bf1 = '', ''
        print(f'{startch}{bf0}{os_disp[os_i]}{bf1} ', end='')
        
        for ia, atk in enumerate(atks):
            if ia ==1:
                print(f' {midch}', end='')
            for metric_i, metric in enumerate(metrics):
                print(f'{sep} ', end='')
                v, rfp_v = disp_val(val[os_i][1][ia][metric_i][1], unit)
                print(bf0 + v + bf1, end='')
            print(f'\t', end='')

        print(f'{endch}')