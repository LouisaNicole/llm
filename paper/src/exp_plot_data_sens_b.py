from matplotlib.ticker import PercentFormatter
from experiment_common import *
from tab_performance import get_metric
from constants import REDUCED_LLMS
from hyperparam import get_map_new_model_name
import numpy as np
import matplotlib.pyplot as plt
import argparse

use_cache = 1

def main(train_ds_collection_id, eval_ds_collection_id):
    pbs = [32, 64, 128, 256]
    metrics = ['LDA', 'R1','RL', 'CosSim', 'ASR']
    labels = ['LS', 'R1', 'RL', 'Cos', 'ASR']
    markers = ['o', 'v', '^', 's', 'D']
    founda_model='gpt-4o-mini-2024-07-18'
    device = 'gpu'
    operationsys = 'ubuntu22_04'
    framework = 'llama.cpp'
    hardware = 'Intel 13900K'

    create_directories('outputs/')
    cache_file_prefix = 'outputs/data_sens_b_'
    if not use_cache or not os.path.exists(f'{cache_file_prefix}vs.npy'):
        vs = [None] * len(metrics)
        for i in range(len(metrics)):
            vs[i] = []

        for im, metric in enumerate(tqdm(metrics)):
            for prefill_bs in pbs:
                
                assert framework == 'llama.cpp'
                if prefill_bs == 256:
                    prefill_bs = None # default bs of llama.cpp
                
                map_new_model_name = get_map_new_model_name(founda_model=founda_model, 
                    train_ds_collection_id=train_ds_collection_id,
                    batch_size=2, rank=256, alpha=256,
                    at=0,
                    hr=1,
                    hf=1,
                    rr_hl=1,
                    pr_hl=1,
                    ht=1,
                    rr_lr = 8e-5,
                    pr_lr = 2e-4,
                    sigma = 0.08,
                    rr_prob = 0.2, # 0.2 0.4 0.8
                    pr_prob = 0.2,
                    scale = 10,
                    rr_epoch = 3,
                    pr_epoch = 2
                )
                new_model_name = map_new_model_name['pr']

                all_items = []
                for llm_i in REDUCED_LLMS:
                    _,llm = VICTIM_LLM[llm_i]
                    items, _ = get_metric(new_model_name,
                                        device, operationsys, framework, hardware,
                                        eval_ds_collection_id, llm, ds_name=None, metric=metric, subds=None, prefill_bs=prefill_bs, embd_quant='F16')
                    all_items.extend(items)
                vs[im].append(np.mean(all_items))
        vs = np.array(vs)
        np.save(f'{cache_file_prefix}vs.npy', vs)
    else:
        print(f'!!! Warning: using cached results: {cache_file_prefix}* !!!')
        vs = np.load(f'{cache_file_prefix}vs.npy')

    marksize=5
    linewidth=0.7
    color_maps = [MY_COLOR_MAPS1[4], MY_COLOR_MAPS1[0], MY_COLOR_MAPS1[3], MY_COLOR_MAPS1[2], MY_COLOR_MAPS1[1]]
    linestyles = ['-', '-', '-', '-', '-.']
    for im, metric in enumerate(metrics):
        plt.plot(pbs, vs[im], label=labels[im],
                 marker=markers[im],
                 markerfacecolor='none',
                 markeredgecolor=color_maps[im],
                 color=color_maps[im],
                 markersize=marksize, linewidth=linewidth)

    plt.ylabel('Metrics')

    plt.xlabel('Prefill Batch Size $b$')
    plt.xticks(pbs, labels=[f'{x:.0f}' for x in pbs])
    #plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if False:
        for l in plt.gca().get_xticklabels():
            l.set_visible(False)
        label_pos = [0.05, 0.15, 0.2, 0.5, 0.85]
        for i, label in enumerate(p):
            t= f'{label}'
            plt.gcf().text(label_pos[i], -0.07, t, ha='center', fontsize=plt.rcParams['xtick.labelsize'])

    plt.gcf().set_size_inches(3.2,1.8)
    plt.gca().grid(True, which="both", linestyle='--', linewidth=0.5)

    plt.legend(bbox_to_anchor=(1, 1))
            #labelspacing=0.3,
            #borderpad=0.1,handletextpad=0.1)
    #plt.tight_layout()
    plt.ylim([0.65,1])
    plt.locator_params(axis='y', nbins=8)

    # set percentage display
    plt.yticks(plt.gca().get_yticks(), labels=[f'{x:.2f}' for x in plt.gca().get_yticks()])

    plt.savefig(f'{FIGURE_OUT_PATH}/data_sens_b.pdf',bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--train-ds-collection-id', type=str, default='natural-language_50000')
    parser.add_argument('--eval-ds-collection-id', type=str, default='natural-language_50000')
    args = parser.parse_args()
    main(**vars(args))