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
    parser.add_argument('--report-format', choices=['markdown', 'latex'], default='latex')
    args = parser.parse_args()

    atks = ['rr', 'pr']
    unit = '%'
    
    hardware = 'Intel 13900K'
    device = 'gpu'
    metrics = ['CosSim', 'ASR'] #['LDA', 'R1','RL', 'CosSim', 'ASR']
    operationsys = 'ubuntu22_04'
    framework = 'llama.cpp'
    victim_llm = 'Mistral-7b-instruct'
    embd_quants = ['F16', 'Q8_0', 'BF16', 'F32']

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
    val = mkarray(len(embd_quants), len(metrics), len(atks))

    llm_vendor,llm = VICTIM_LLM[get_llm_index(victim_llm)]

    for quant_i, embd_quant in enumerate(tqdm(embd_quants)):
        for ia, atk in enumerate(atks):
            for metric_i, metric in enumerate(metrics):
                n_asr = 0
                n_atk = 0
                
                data, n_samples = get_metric(map_new_model_name[atk],
                                            device, operationsys, framework, hardware,
                                            args.eval_ds_collection_id, llm, None, metric, embd_quant=embd_quant)
                v = np.mean(data)
                val[quant_i][metric_i][ia] = (n_samples,v)

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

    if args.report_format == 'markdown':
        sep = '|'
        startch = '|'
        midch = ''
        endch = '|'
    elif args.report_format == 'latex':
        sep = '&'
        startch = '\\midrule\n'
        midch = '& '
        endch = r'\\'

    # generate the table
    for quant_i, embd_quant in enumerate(embd_quants):
        print(f'{startch}{embd_quant} ', end='')

        for ia, atk in enumerate(atks):
            if ia ==1:
                print(f' {midch}', end='')
            for metric_i, metric in enumerate(metrics):
                print(f'{sep} ', end='')
                v, rfp_v = disp_val(val[quant_i][metric_i][ia][1], unit)
                if embd_quant != 'F16':
                    if args.report_format == 'markdown':
                        bf0, bf1 = '**', '**'
                    elif args.report_format == 'latex':
                        bf0, bf1 = '\\textbf{', '}'
                    print(v, end='')
                else:
                    print(v, end='')
            print(f'\t', end='')
        print(f'{endch}')


