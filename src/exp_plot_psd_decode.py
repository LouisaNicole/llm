from experiment_common import *
from matplotlib.gridspec import GridSpec
import argparse

def plot_timing_signal(timepoints, ax, value,label='', color='b', marker='o',markersize=None):
    if False:
        markerline, stemline, baseline, = ax.stem(timepoints-timepoints[0],[value]*len(timepoints),linefmt=color+'-',markerfmt=color+'o',basefmt=color+'.',
        label=label)
        plt.setp(stemline, linewidth = 1.25)
        plt.setp(markerline, markersize = 4)
    ax.scatter(timepoints-timepoints[0], [value]*len(timepoints), marker=marker, facecolors='none' if marker=='o' else color, edgecolors=color, label=label, s=markersize)

def figure_PSD_for_resp(ds, tokenizer, time_limit_ratio=0.2, tpmarkpeek=0.75, tpmarkwidth=10):
    sel_record = 452 # a random selected one for an example
    record = ds['results'][sel_record]
    timepoints, tokens, _ = get_tp_tk(record)
    p_start, p_end, r_start, r_end, _,_,_ = identify_prompt_resp(timepoints)
    tp_resp = timepoints[r_start:r_end]
    tk_resp = tokens[r_start:r_end]
    time_limit = (tp_resp[-1]-tp_resp[0]+1) * time_limit_ratio

    resp_groundtruth = record['g']['o']
    (tp_timepoints, tp_tokens), (fp_timepoints, fp_tokens) = split_false_positives(tp_resp, tk_resp, resp_groundtruth)

    fig = plt.gcf()
    #fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[0.6, 1])
    ax = [fig.add_subplot(gs[0, :]),
          fig.add_subplot(gs[1, 0]),
          fig.add_subplot(gs[1, 1])]

    fs, nfft, nperseg, fmax = get_sample_params(tp_timepoints)
    print(f'nperseg = {nperseg}')
    print(f'nfft = {nfft}')
    print(f'fmax = {fmax/1e3} KHz')
    print(f'fs = {fs/1e3} KHz')
    freq_res = 1/(timepoints[-1]-timepoints[0])
    print(f'Freq resolution = {freq_res} Hz')

    tmp=0.15
    plot_timing_signal(tp_timepoints, ax[0], tmp,label='True Positive', color='tab:blue', marker='x')
    plot_timing_signal(fp_timepoints, ax[0], 1-tmp,label='False Positive',color='tab:orange', marker='o')
    ax[0].set_ylim([0,1])
    ax[0].set_xlim([0,time_limit])
    #ax[0].set_ylabel('Cache Hit Event')
    ax[0].set_xlabel('Time of Cache Hit Events (s)', labelpad=0)
    ax[0].legend(labelspacing=0.1,borderpad=0.1,ncol=2,columnspacing=0,handletextpad=0.1)
    for label in ax[0].get_yticklabels():
        label.set_visible(False)

    #plt.show() # DEBUG

    unit = 'Hz'
    fs += 110
    nfft = round(fs * (tp_timepoints[-1] - tp_timepoints[0]))
    _, _,_, p1 = plot_PSD(tp_timepoints, ax[1], fs, nfft, nperseg, markpeek=tpmarkpeek, markwidth=tpmarkwidth, unit=unit, color='tab:blue')
    _, _,_, p2 = plot_PSD(fp_timepoints, ax[2], fs, nfft, nperseg, unit=unit, color='tab:orange')
    a = -12#min(np.min(p1), np.min(p2))
    b = max(np.max(p1), np.max(p2))+15
    xlim = [50,fs/2]
    ax[1].set_ylim((a,b))
    ax[1].set_xlim(xlim)
    ax[1].grid(True, which="both", linestyle='--', linewidth=0.5)
    #ax[1].tick_params(axis='y', labelsize=8)
    #plt.yticks(fontsize=8)

    ax[2].set_ylim((a,b))
    ax[2].set_xlim(xlim)
    for i in [1,2]:
        ax[i].set_xlabel('')
    ax[2].set_ylabel('')
    ax[2].grid(True, which="both", linestyle='--', linewidth=0.5)
    #ax[2].tick_params(axis='y', labelsize=8)

    ax[1].set_xlabel(f'True Positive Freq ({unit})',y=0.01)
    ax[2].set_xlabel(f'False Positive Freq ({unit})',y=0.01)
    #fig.supxlabel(f'Frequency ({unit})',y=0.01)#,fontsize=9)

    fig.subplots_adjust(hspace=0.8, wspace=0.20,
                        bottom=0.18)
    #fig.tight_layout()

    delta=0.1
    pos = ax[0].get_position()
    ax[0].set_position([pos.x0-delta,pos.y0, pos.width+delta, pos.height])

    fig.set_size_inches(4,2)

    plt.savefig(f'{FIGURE_OUT_PATH}/psd_example.pdf',bbox_inches='tight')
    plt.show()

#########################
# Algoritm Exploring
#########################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--device', type=str, help="", default='gpu')
    parser.add_argument('--operationsys', type=str, help="", default='ubuntu22_04')
    parser.add_argument('--framework', type=str, help="", default='llama.cpp')
    parser.add_argument('--hardware', type=str, help="", default='Intel 13900K')
    parser.add_argument('--ds-collection-id', type=str, default='natural-language_50000') 
    parser.add_argument('--victim-llm', type=int, default=1)
    parser.add_argument('--prefill-bs', type=int, default=None)
    parser.add_argument('--embd-quant', type=str, default='F16')
    args = parser.parse_args()

    llm_vendor,llm=VICTIM_LLM[args.victim_llm]

    # load collected cache traces
    ds = json.load(open(probed_ds_test_fn(get_probed_res_prefix(args.device, args.operationsys, args.framework, args.hardware),
                                          llm, args.ds_collection_id, args.prefill_bs, args.embd_quant), 'r'))

    # for attack result
    tokenizer = load_tokenizer(llm)
    figure_PSD_for_resp(ds, tokenizer)