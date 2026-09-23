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

def plot_ablation(atk, train_ds_collection_id, eval_ds_collection_id):
    metrics = ['LDA', 'R1','RL', 'CosSim', 'ASR']
    metric_labels = ['LS', 'R1', 'RL', 'Cos', 'ASR']
    device = 'gpu'
    operationsys = 'ubuntu22_04'
    framework = 'llama.cpp'
    hardware = 'Intel 13900K'
    args = Namespace(
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
    cache_file_prefix = f'outputs/baseline_analysis_{atk}_'
    if not use_cache or not os.path.exists(f'{cache_file_prefix}vs.npy'):
        configs = [
            Namespace(name='Llama3.1-8B', founda_model='Llama-3.1-8B-Instruct', at=0, hr=1, hf=1, rr_hl=1, pr_hl=1, ht=1),
            Namespace(name='gpt-4o-mini', founda_model='gpt-4o-mini-2024-07-18', at=0, hr=1, hf=1, rr_hl=1, pr_hl=1, ht=1),
        ]

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

    plt.clf()

    width = 0.3
    space = 0.5

    plt.figure(figsize=(10, 6))

    plt.rcParams['hatch.linewidth'] = 0.25

    x = np.arange(1, len(metrics)+1)
    color_map = copy.deepcopy(MY_COLOR_MAPS1)
    color_map.reverse()
    patterns = ['///', '---', '\\\\', '|']
    for ic in range(len(config_names)):
        plt.bar(x + ic * width - space * width, vs[ic,:].flatten(), width,
                label=config_names[ic], color=color_map[ic], edgecolor='grey',
                hatch=patterns[ic])

    plt.ylabel('Metric')
    plt.xticks(x, metric_labels)

    plt.gcf().set_size_inches(2,1.5)
    plt.gcf().legend(loc='lower center',
            bbox_to_anchor=(0.5, -0.01 ),
            ncol=2,
            columnspacing=0.1,
            labelspacing=0.2,borderpad=0.1,handletextpad=0.1
    )
    if atk == 'rr':
        plt.ylim([0.2,1])
    else:
        plt.ylim([0,1])
    plt.locator_params(axis='y', nbins=8)
    plt.subplots_adjust(bottom=0.35)

    plt.savefig(f'{FIGURE_OUT_PATH}/baseline_analysis_{atk}.pdf',bbox_inches='tight')
    #plt.tight_layout()
    #plt.show()

def main(**args):
    for atk in tqdm(['rr', 'pr']):
        plot_ablation(atk, **args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--train-ds-collection-id', type=str, default='natural-language_50000')
    parser.add_argument('--eval-ds-collection-id', type=str, default='natural-language_50000')
    args = parser.parse_args()
    main(**vars(args))