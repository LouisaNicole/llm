from constants import *
from hyperparam import get_map_new_model_name
from dynamicavg import DynamicAvg
from experiment_common import load_tokenizer, text2tokens, create_directories
from tab_common import *
import argparse
from tqdm import tqdm
import numpy as np
from myjsonl import *
import math

use_cache = 0

def get_test_restored_file(device, operationsys, framework, hardware, llm, new_model_name, eval_ds_collection_id, val=False, prefill_bs=None, embd_quant='F16'):
    ATK_PREFIX=f'{get_probed_res_prefix(device, operationsys, framework, hardware)}/{llm}/{new_model_name}'

    def skip_sample(record, dataset_k, subds):
        if dataset_k is not None and record['ds_name'] != ds_name:
            return True
        if subds is not None and record['ds_sub'] not in subds:
            return True
        return False
    
    return f'{ATK_PREFIX}/analyzed_{"eval_val" if val else "test"}_results_{eval_ds_collection_id}{get_suffix(prefill_bs, embd_quant)}.jsonl'

def get_metric(new_model_name,
               device, operationsys, framework, hardware,
               eval_ds_collection_id, llm, ds_name, metric, subds=None, val=False, prefill_bs=None, embd_quant='F16'):
    
    def skip_sample(record, dataset_k, subds):
        if dataset_k is not None and record['ds_name'] != ds_name:
            return True
        if subds is not None and record['ds_sub'] not in subds:
            return True
        return False
    
    test_restored_file = get_test_restored_file(device, operationsys, framework, hardware, llm, new_model_name, eval_ds_collection_id, val, prefill_bs, embd_quant)

    if metric == 'ASR':
        n_succ = 0
        n_fail = 0
        for record in jsonl_read(test_restored_file):
            if skip_sample(record, ds_name, subds):
                continue
            if record['cos_sim'] > ASR_COS_THRESHOLD:
                n_succ +=1
            else:
                n_fail += 1
        return [n_succ / (n_succ + n_fail + 1e-20)], n_succ + n_fail

    elif metric == 'N_tokens':
        tokenizer = load_tokenizer(llm)
        ret = []
        for record in jsonl_read(test_restored_file):
            if skip_sample(record, ds_name, subds):
                continue
            ret.append(len(text2tokens(tokenizer, record['t']['messages'][2]['content'])))
        return ret, len(ret)
    
    elif metric == 'TokenAcc':
        tokenizer = load_tokenizer(llm)
        ret = []
        for record in jsonl_read(test_restored_file):
            if skip_sample(record, ds_name, subds):
                continue
            gt = text2tokens(tokenizer, record['t']['messages'][2]['content'])
            pred = text2tokens(tokenizer, record['pred'])
            TP = len(set(gt) & set(pred))
            FP = len(set(pred) - set(gt))
            TN = 0
            FN = len(set(gt) - set(pred))
            ret.append((TP+TN)/(TP+FP+TN+FN))
        return ret, len(ret)

    else:
        metric_to_jsonfield = {
            'LDA': 'LD',
            'R1': 'r1',
            'R2': 'r2',
            'RL': 'rL',
            'Exact': 'exact',
            'CosSim': 'cos_sim'
        }
        ret = []
        for record in jsonl_read(test_restored_file):
            if skip_sample(record, ds_name, subds):
                continue
            v = record[metric_to_jsonfield[metric]]
            ret.append(v)
        return ret, len(ret)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--train-ds-collection-id', type=str, default='natural-language_50000')
    parser.add_argument('--eval-ds-collection-id', type=str, default='natural-language_50000')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--operationsys', type=str, default='ubuntu22_04')
    parser.add_argument('--framework', type=str, default='llama.cpp')
    parser.add_argument('--hardware', type=str, default='Intel 13900K')
    args = parser.parse_args()

    atks = ['rr', 'pr']
    METRIC_NAME = ['N_tokens', 'R1', 'RL', 'LDA', 'CosSim', 'ASR'] 
    UNITS = ['int', '%', '%', '%', '%', '%']

    map_llm = {
        'Falcon3-10B-Instruct': r'''\multirow{4}*{\begin{tabular}{@{}c@{}}TII \\Falcon3-10B\\~\cite{falcon3}\end{tabular}}''',
        'Falcon3-7B-Instruct-1.58bit': r'''\multirow{4}*{\begin{tabular}{@{}c@{}}TII \\Falcon3-7B (1.58bit)\\~\cite{falcon3}\end{tabular}}''',
        'Falcon3-1B-Instruct':r'''\multirow{4}*{\begin{tabular}{@{}c@{}}TII \\Falcon3-1B \\~\cite{falcon3}\end{tabular}}''',
        'gemma-2-9b-it': r'''\multirow{4}*{\begin{tabular}{@{}c@{}}Google \\Gemma2-9B\\~\cite{gemma}\end{tabular}}''',
        'Llama-3.1-8b-instruct': r'''\multirow{4}*{\begin{tabular}{@{}c@{}}Meta \\Llama-3.1-8B\\~\cite{llama}\end{tabular}}''',
        'Llama-2-7b-chat': r'''\multirow{4}*{\begin{tabular}{@{}c@{}}Meta \\Llama-2-7B\\~\cite{llama}\end{tabular}}''',
        'Mistral-7b-instruct': r'''\multirow{4}*{\begin{tabular}{@{}c@{}} Mistral-7B\\~\cite{mistral}\end{tabular}}''',
        'Phi-3.5-mini-instruct': r'''\multirow{4}*{\begin{tabular}{@{}c@{}}Microsoft \\Phi-3.5-mini-3B\\~\cite{phi35}\end{tabular}}''',
    }

    founda_model='Llama-3.1-8B-Instruct'
    founda_model='gpt-4o-mini-2024-07-18'
    
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
    target_llms = [0,3,2,4,1]

    def mkarray(x,y,z=None,w=None):
        ret = [None] * x
        for i in range(x):
            ret[i] = [None] * y
            if z:
                for j in range(y):
                    ret[i][j] = [None] *z
                    if w:
                        for k in range(z):
                            ret[i][j][k] = [None]*w
        return ret

    create_directories('outputs/')
    cache_file_prefix = f'outputs/tab_performance_'
    if not use_cache or not os.path.exists(f'{cache_file_prefix}vs.npy'):
        val = mkarray(len(atks), len(METRIC_NAME), len(target_llms), len(map_ds_name.items()))

        for llm_ii, llm_i in enumerate(tqdm(target_llms)):
            llm_vendor,llm = VICTIM_LLM[llm_i]
            for ds_i, (ds_name, _) in enumerate(map_ds_name.items()):
                for ia, atk in enumerate(atks):
                    for im, m in enumerate(METRIC_NAME):
                        n_asr = 0
                        n_atk = 0
                        
                        data, n_samples = get_metric(map_new_model_name[atk],
                                                    args.device, args.operationsys, args.framework, args.hardware,
                                                    args.eval_ds_collection_id, llm, ds_name, m)
                        v = np.mean(data)
                        val[ia][im][llm_ii][ds_i] = (n_samples,v)

        # stats
        max_val = mkarray(len(atks), len(METRIC_NAME))
        min_val = mkarray(len(atks), len(METRIC_NAME))
        weighted_avg = mkarray(len(atks), len(METRIC_NAME))


        RFP_SCALE=10
        def reserve_fp(v): # for comparing integers instead of floats
            return round(v*100*RFP_SCALE) # round to 0.1% for unit=%, and 0.001 for unit=None
        
        for ia, atk in enumerate(atks):
            for im, m in enumerate(METRIC_NAME):
                min_val[ia][im] = float('inf')
                max_val[ia][im] = float('-inf')
                total_v = 0
                total_n_samples = 0
                for llm_ii, _ in enumerate(target_llms):
                    for ds_i, (ds_name, _) in enumerate(map_ds_name.items()):
                        n_samples, v = val[ia][im][llm_ii][ds_i]
                        if not math.isnan(v): # this happends for empty results
                            rfp_v = reserve_fp(v) 
                            min_val[ia][im] = min(min_val[ia][im], rfp_v)
                            max_val[ia][im] = max(max_val[ia][im], rfp_v)
                            total_v += v * n_samples
                            total_n_samples += n_samples
                weighted_avg[ia][im] = total_v / total_n_samples

        val = np.array(val)
        np.save(f'{cache_file_prefix}vs.npy', val)

        min_val = np.array(min_val)
        np.save(f'{cache_file_prefix}min_val.npy', min_val)

        max_val = np.array(max_val)
        np.save(f'{cache_file_prefix}max_val.npy', max_val)

        weighted_avg = np.array(weighted_avg)
        np.save(f'{cache_file_prefix}weighted_avg.npy', weighted_avg)
    else:
        print(f'!!! Warning: using cached results: {cache_file_prefix}* !!!')
        val = np.load(f'{cache_file_prefix}vs.npy')
        min_val = np.load(f'{cache_file_prefix}min_val.npy')
        max_val = np.load(f'{cache_file_prefix}max_val.npy')
        weighted_avg = np.load(f'{cache_file_prefix}weighted_avg.npy')
        
    def disp_val(val, unit):
        if math.isnan(val):
            return '-', 0
        else:
            rfp_v = reserve_fp(val)
            if UNITS[im] == '%':
                return f'{rfp_v/RFP_SCALE:.1f}', rfp_v
            elif UNITS[im] == 'int':
                return str(int(val)), rfp_v
            elif UNITS[im] is None:
                return f'{val:.3f}', rfp_v
            else:
                assert False

    # generate the table
    for llm_ii, llm_i in enumerate(target_llms):
        llm_vendor,llm = VICTIM_LLM[llm_i]

        print('%'+llm)
        print(r'\midrule')
        print(f'{map_llm[llm]} ', end='')

        for ds_i, (ds_name, _) in enumerate(map_ds_name.items()):
            print(f'& {map_ds_name[ds_name]} ',end='')

            for ia, atk in enumerate(atks):
                if ia ==1:
                    print('& ', end='')
                for im, m in enumerate(METRIC_NAME):
                    print(f'& ', end='')
                    v, rfp_v = disp_val(val[ia][im][llm_ii][ds_i][1], UNITS[im])
                    mark_minmax = (m != 'N_tokens')
                    if mark_minmax and min_val[ia][im] == rfp_v and max_val[ia][im] == rfp_v:
                        print(r' \underline{\textbf{' + v + '}}', end='')
                    elif mark_minmax and min_val[ia][im] == rfp_v: 
                        print(r' \underline{' + v + '}', end='')
                    elif mark_minmax and max_val[ia][im] == rfp_v:
                        print(r' \textbf{' + v + '}', end='')
                    else:
                        print(v, end='')
                print('\t', end='')
            print(r'\\')

    print(r'\midrule')
    print(r'\multicolumn{2}{c}{Average} ', end='')
    for ia, atk in enumerate(atks):
        if ia ==1:
            print('& ', end='')
        for im, m in enumerate(METRIC_NAME):
            print(f'& {disp_val(weighted_avg[ia][im], UNITS[im])[0]}', end='')
        print('\t', end='')

    print(r'\\')

