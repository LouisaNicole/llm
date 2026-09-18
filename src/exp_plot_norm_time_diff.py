from experiment_common import *
import argparse

use_cache = 1

def plot_norm_time_diff(ds, tokenizer):
    record = ds['results'][0]
    timepoints, tokens, _ = get_tp_tk(record)

    N=100
    
    (_,_), (tp_resp, tk_resp) = split_prompt_response(timepoints, tokens)

    tp_resp = tp_resp[2:N]
    tk_resp = tk_resp[2:N]

    tp_indices, fp_indices = split_false_positives_tokens_indices(tk_resp, record['g']['o'])
    fn_timepoints = get_false_negatives(tp_resp, tk_resp, record['g']['o'])


    #N = 150
    tbe = estimate_f0(tp_resp, tk_resp, tokenizer) * (tp_resp[1:] - tp_resp[:-1])
    #tbe = tbe[:-N]

    fig,ax = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    ax[0].plot(range(len(tbe)), tbe)
    ax[0].set_ylabel("$T'_Dk$")
    ax[0].set_yticks([0,1,2])
    ax[0].grid(True, which="both", linestyle='--', linewidth=0.5)

    
    markersize=20
    first = True
    for i in fp_indices:
        if True or i-1 > -1 and i-1 < len(tk_resp)-N:
            ax[1].scatter(i-1, [1.1], color='r', s=markersize, marker='D',
                          facecolor='r',
                            edgecolor='#800000',
                          label='FP' if first else None)
            first = False
    #ax[1].set_ylabel('FP')
    for label in ax[1].get_yticklabels():
        label.set_visible(False)
        

    first = True
    print('fn_timepoints=', fn_timepoints, tp_resp, tk_resp, record['g']['o'])
    for t in fn_timepoints:
        if False and t > tp_resp[-N]:
            break
        i = find_closest_indice(tp_resp, t)
        print(i)
        ax[1].scatter(i, [0.5], color='b', s=markersize, marker='o',
                      facecolor='skyblue',
                        edgecolor='navy',
                    label='FN' if first else None)
        first = False
    
    print("len(tp_indices) = ", len(tp_indices))
    for t in tp_indices:
        #ax[1].scatter(t-1, [0.25], color='g', s=8, marker='o', label='TP' if first else None)
        pass

    ax[1].set_ylabel('Event')
    ax[1].set_xlabel('Time Step $k$')
    for label in ax[1].get_yticklabels():
        label.set_visible(False)
    ax[1].legend(labelspacing=0.1,borderpad=0.1,ncol=2,columnspacing=0.1,handletextpad=0.1)
    ax[1].set_ylim([0,1.5])
    ax[1].grid(True, axis='x', which="both", linestyle='--', linewidth=0.5)

    plt.xlim([0,len(tbe)])

    plt.gcf().set_size_inches(5,0.8)
    plt.savefig(f'{FIGURE_OUT_PATH}/norm_time_diff.pdf', bbox_inches='tight')
    plt.show()


#########################
# Algoritm Exploring
#########################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--device', type=str, help="", default='gpu')
    parser.add_argument('--operationsys', type=str, default='ubuntu22_04')
    parser.add_argument('--framework', type=str, help="", default='llama.cpp')
    parser.add_argument('--hardware', type=str, help="", default='Intel 13900K')
    parser.add_argument('--ds-collection-id', type=str, default='natural-language_50000')
    parser.add_argument('--victim-llm', type=int, default=1)
    parser.add_argument('--prefill-bs', type=int, default=None)
    parser.add_argument('--embd-quant', type=str, default='F16')
    args = parser.parse_args()

    llm_vendor,llm=VICTIM_LLM[1]

    # load collected cache traces
    ds = json.load(open(get_probed_res_prefix(args.device, args.operationsys, args.framework, args.hardware) + '/norm_time_diff.json', 'r'))

    # for attack result
    tokenizer = load_tokenizer(llm)

    plot_norm_time_diff(ds, tokenizer)
