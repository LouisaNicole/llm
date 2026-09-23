import os
os.environ['NOT_INIT_OPENAI'] = '1'
from experiment_common import *
from tab_performance import get_metric
import numpy as np
import matplotlib.pyplot as plt
from hyperparam import get_map_new_model_name
from constants import REDUCED_LLMS

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

def plot_param(atk, train_ds_collection_id, eval_ds_collection_id, sen_metric):
    sigma_vals=[0.04, 0.08, 0.16, 0.32, 0.64, 1.28]
    prob_vals=[0.1, 0.2, 0.4, 0.6, 0.8]
    metrics = ['LDA', 'R1','RL', 'CosSim', 'ASR']
    labels = ['LS', 'R1', 'RL', 'Cos', 'ASR']
    markers = ['o', 'v', '^', 's', 'D']
    founda_model='Llama-3.1-8B-Instruct'#'gpt-4o-mini-2024-07-18'
    device = 'gpu'
    operationsys = 'ubuntu22_04'
    framework = 'llama.cpp'
    hardware = 'Intel 13900K'

    sen_vals = prob_vals if sen_metric == 'p' else sigma_vals

    create_directories('outputs/')
    cache_file_prefix = f'outputs/param_{sen_metric}_analysis_{atk}_'
    if not use_cache or not os.path.exists(f'{cache_file_prefix}vs.npy'):
        vs = [None] * len(metrics)
        for i in range(len(metrics)):
            vs[i] = []

        for im, metric in enumerate(tqdm(metrics)):
            for sen_param in sen_vals:
                
                assert framework == 'llama.cpp'
                
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
                    sigma = sen_param if sen_metric == 's' else 0.08,
                    rr_prob = sen_param if sen_metric == 'p' else 0.2,
                    pr_prob = sen_param if sen_metric == 'p' else 0.2,
                    scale = 10,
                    rr_epoch = 3,
                    pr_epoch = 2
                )
                new_model_name = map_new_model_name[atk]

                all_items = []
                for llm_i in REDUCED_LLMS:
                    _,llm = VICTIM_LLM[llm_i]
                    items, _ = get_metric(new_model_name,
                                        device, operationsys, framework, hardware,
                                        eval_ds_collection_id, llm, ds_name=None, metric=metric, subds=None, val=True, prefill_bs=None, embd_quant='F16')
                    all_items.extend(items)
                vs[im].append(np.mean(all_items))
        vs = np.array(vs)
        np.save(f'{cache_file_prefix}vs.npy', vs)
    else:
        print(f'!!! Warning: using cached results: {cache_file_prefix}* !!!')
        vs = np.load(f'{cache_file_prefix}vs.npy')

    plt.clf()

    marksize=4
    linewidth=0.7
    color_map = copy.deepcopy(MY_COLOR_MAPS1)
    #color_map.reverse()
    #color_map[0] = (69/255,144/255,247/255) # blue
    for im, metric in enumerate(metrics):
        plt.plot(sen_vals, vs[im], label=labels[im],
                    marker=markers[im],
                    markerfacecolor='none',
                     markeredgecolor=color_map[im],
                    color=color_map[im], markersize=marksize, linewidth=linewidth)

    if sen_metric == 'p':
        plt.xlim([0,1])
        plt.xticks(sen_vals, labels=[f'{x*100:.0f}' for x in sen_vals],
                   fontsize=small_font_size)
        lenged_pad=0
        xlabel_pad=0
    else:
        #plt.xlim([0,max(sen_vals)])
        plt.xticks(sen_vals, labels=[f'{x:.2f}' for x in sen_vals],
                   fontsize=small_font_size, rotation=45)
        lenged_pad = 0.15
        xlabel_pad = 0.2

    plt.xlabel('$' + ("p (\\%)" if sen_metric == "p" else r"\sigma") + '$', labelpad=0)
    plt.gca().xaxis.set_label_coords(0.5, -0.7 - xlabel_pad)

    #plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #plt.gca().xaxis.label.set_y(-2)

    if sen_metric == 's':
        label_pos = [0.08, 0.17, 0.25]
        for l in plt.gca().get_xticklabels()[:len(label_pos)]:
            l.set_visible(False)
        for i, label in enumerate(label_pos):
            t= f'{sigma_vals[i]:.2f}'
            
            plt.gcf().text(label_pos[i]-0.01, -0.27, s=t, ha='center', fontsize=small_font_size,rotation=45)
    else:
        plt.gca().get_xticklabels()[0].set_visible(False)
        plt.gcf().text(prob_vals[0]+0.05, -0.11, s=int(prob_vals[0]*100), ha='center', fontsize=small_font_size)


    plt.gcf().set_size_inches(1.9,0.8)
    plt.gca().grid(True, which="both", linestyle='--', linewidth=0.5)

    plt.ylabel('Metrics')
    if sen_metric == 'p':
        plt.ylim([0.5, 1])
        plt.locator_params(axis='y', nbins=6)
    else:
        plt.ylim([0.7, 1])
        plt.locator_params(axis='y', nbins=6)

    # set percentage display
    plt.yticks(plt.gca().get_yticks(), labels=[f'{x:.2f}' for x in plt.gca().get_yticks()])

    plt.gcf().legend(loc='lower center',
           bbox_to_anchor=(0.5, -0.45-lenged_pad),
           ncol=5,
           columnspacing=0.1,
           labelspacing=0.1,borderpad=0.1,handletextpad=0.1
)
    #plt.tight_layout()
    plt.savefig(f'{FIGURE_OUT_PATH}/param_{sen_metric}_analysis_{atk}.pdf',bbox_inches='tight')
    #plt.show()

def main(train_ds_collection_id, eval_ds_collection_id):
    for atk in tqdm(['rr', 'pr']):
        for metric in ['p', 's']:
            plot_param(atk, train_ds_collection_id, eval_ds_collection_id, metric)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--train-ds-collection-id', type=str, default='natural-language_50000')
    parser.add_argument('--eval-ds-collection-id', type=str, default='natural-language_50000')
    args = parser.parse_args()
    main(**vars(args))