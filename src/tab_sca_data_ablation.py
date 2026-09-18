from argparse import Namespace
from experiment_common import *
import numpy as np
from tab_performance import get_metric
from constants import REDUCED_LLMS
from hyperparam import get_map_new_model_name
import matplotlib.pyplot as plt

small_font_size = 8
plt.rcParams.update({
    'font.size': 10, 
    #'font.family': 'Nimbus Roman No9 L',#'serif',
    'axes.titlesize': 10,  # 设置标题字体大小
    'axes.labelsize': 10,  # 设置坐标轴标签字体大小
    'xtick.labelsize': 8,  # 设置 x 轴刻度字体大小
    'ytick.labelsize': 8,  # 设置 y 轴刻度字体大小
    'legend.fontsize': 8,  # 设置图例字体大小
})

use_cache = 1

def tab_ablation(train_ds_collection_id, eval_ds_collection_id):
    atk = 'pr'
    metrics = ['R1','RL', 'LDA', 'CosSim', 'ASR']
    metric_labels = ['LS', 'R1', 'RL', 'Cos', 'ASR']
    founda_model='Llama-3.1-8B-Instruct'#'gpt-4o-mini-2024-07-18'
    device = 'gpu'
    operationsys = 'ubuntu22_04'
    framework = 'llama.cpp'
    hardware = 'Intel 13900K'
    args = Namespace(founda_model=founda_model, 
                train_ds_collection_id=train_ds_collection_id,
                eval_ds_collection_id=eval_ds_collection_id,
                batch_size=2, rank=256, alpha=256,
                rr_lr = 8e-5,
                pr_lr = 2e-4,
                sigma = 0.08,
                rr_prob = 0.2,
                pr_prob = 0.2,
                scale = 10,
                rr_epoch = 3,
                pr_epoch = 2
            )
    
    create_directories('outputs/')
    cache_file_prefix = f'outputs/supply_ablation_study_{atk}_'
    if not use_cache or not os.path.exists(f'{cache_file_prefix}vs.npy'):
        pr_configs = [
            Namespace(name='The Proposed Attack', at=0, hr=1, hf=1, rr_hl=1, pr_hl=1, ht=1),
            Namespace(name='Not used generated text', at=0, hr=0, hf=1, rr_hl=1, pr_hl=1, ht=1),
            Namespace(name='Pure side-channel attack', at=0, hr=1, hf=1, rr_hl=0, pr_hl=0, ht=1),
            Namespace(name='Pure output-based attack', at=0, hr=1, hf=1, rr_hl=1, pr_hl=1, ht=1, hc=0),
        ]
        configs = pr_configs

        vs = [None] * len(configs)
        for i in range(len(configs)):
            vs[i] = [None] * len(metrics)
            for j in range(len(vs[i])):
                vs[i][j] = []

        config_names = []
        for ic, config in enumerate(tqdm(configs)):
            for im, metric in enumerate(metrics):
                assert framework == 'llama.cpp'
                this_args = copy.deepcopy(args)
                this_args.__dict__.update(config.__dict__)
                del this_args.__dict__['name']
                del this_args.__dict__['eval_ds_collection_id']
                map_new_model_name = get_map_new_model_name(**vars(this_args))
                new_model_name = map_new_model_name[atk]

                all_items = []
                for llm_i in REDUCED_LLMS:
                    _,llm = VICTIM_LLM[llm_i]
                    items, _ = get_metric(new_model_name,
                                        device, operationsys, framework, hardware,
                                        eval_ds_collection_id,
                                        llm, ds_name=None, metric=metric, subds=None, val=False, prefill_bs=None, embd_quant='F16')
                    all_items.extend(items)
                vs[ic][im].append(np.mean(all_items))
            config_names.append(config.name)
        assert len(vs) == len(config_names)
        vs = np.array(vs)
        np.save(f'{cache_file_prefix}vs.npy', vs)
        config_names = np.array(config_names)
        np.save(f'{cache_file_prefix}config_names.npy', config_names)
    else:
        print(f'!!! Warning: using cached results: {cache_file_prefix}* !!!')
        vs = np.load(f'{cache_file_prefix}vs.npy')
        config_names = np.load(f'{cache_file_prefix}config_names.npy')

    for ic in range(len(config_names)):
        print(config_names[ic], (vs[ic] )*100, '%')

    unit = '%'
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

    markdown = False
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
    for cfg_i, cfg in enumerate(config_names):
        if cfg_i > 1:
            if markdown:
                bf0, bf1 = '**', '**'
            else:
                bf0, bf1 = '\\textbf{', '}'
        else:
            bf0, bf1 = '', ''
        print(f'{startch}{bf0}{cfg}{bf1} ', end='')
        
        for metric_i, metric in enumerate(metrics):
            print(f'{sep} ', end='')
            assert len(vs[cfg_i][metric_i]) == 1
            v, rfp_v = disp_val(vs[cfg_i][metric_i][0], unit)
            print(v, end='')
            #dv, rfp_dv = disp_val(vs[cfg_i][metric_i][0] - vs[0][metric_i][0], unit)
            #print(' (' + dv + ')', end='')

        #print(f'\t', end='')

        print(f'{endch}')

def main(**args):
    tab_ablation(**args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--train-ds-collection-id', type=str, default='natural-language_50000')
    parser.add_argument('--eval-ds-collection-id', type=str, default='natural-language_50000')
    args = parser.parse_args()
    main(**vars(args))