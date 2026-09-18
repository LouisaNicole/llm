from experiment_common import *
from matplotlib.ticker import MultipleLocator

def plot_cache_trace(victim, tokenizer):
    prefix = get_probed_res_prefix('gpu', 'ubuntu22_04', 'llama.cpp', 'Intel 13900K')
    with open(f'{prefix}/cache_trace.csv', 'r') as fp:
        latency = list(map(int,fp.readlines()))

    with open(f'{prefix}/cache_trace.csv.vocab.csv', 'r') as fp:
        vocab = list(map(int,fp.readlines()))

    assert len(latency) % len(vocab) == 0
    num_tries = int(len(latency) / len(vocab))
    print('num_vocab = ', len(vocab))
    print('num_tries = ', num_tries)

    sort = []
    # sort by their original order in the ground truth
    if 0:
        in_gt = json.load(open('groundtruth.json', 'r'))['i']
        out_gt = json.load(open('groundtruth.json', 'r'))['o']
    else:
        # run-collect-latency first, then copy the ['g']['o'] of the groundtruth.json to the follows:
        in_gt = [1,518,25580,29962,518,25580,29962,6113,385,3686,388,1048,278,2834,310,20212,21504,29889,1987,19138,675,596,3686,388,964,263,26576,29889,922,862,403,278,3686,388,322,278,26576,411,29871,29953,263,2475,3873,15072,29901,334,2328,29930,13,797,916,3838,29892,596,2933,881,505,278,1494,883,29901,13,29961,404,388,29962,13,2328,1068,13,29961,1129,331,3816,29914,25580,3816,29914,25580,29962]
        out_gt = [29871,20212,21504,471,263,1565,15680,755,29892,263,767,310,1784,5969,1237,322,1209,1080,29889,19298,297,12115,297,29871,29896,29955,29900,29953,29892,540,471,278,29871,29896,29945,386,310,29871,29896,29955,4344,304,263,23794,280,2136,261,322,670,6532,29889,19454,278,3165,569,1812,2559,886,29892,21504,11492,304,4953,697,310,278,1556,7112,2556,13994,297,3082,4955,29889,13,13,2887,263,4123,767,29892,21504,471,623,29878,4173,287,304,670,9642,8099,5011,29892,1058,15205,263,14010,5381,297,18292,29889,21504,9098,11827,3654,304,367,263,2071,24455,9227,29892,6920,29892,322,9805,261,29892,322,540,4720,3897,263,18096,297,278,5381,29889,940,1304,670,14010,3965,304,9805,278,16636,29604,29892,607,3897,697,310,278,1556,17644,1303,322,3390,287,14578,21321,297,278,8104,583,29889,13,13,7675,29895,1915,29915,29879,2551,297,27256,5331,304,916,28602,1907,29889,940,471,11467,304,278,16636,13266,297,29871,29896,29955,29941,29896,322,9098,3897,263,11822,297,278,784,2592,29889,940,471,263,4549,22545,403,363,9793,322,9213,304,10127,278,3014,310,16636,29889,940,884,5318,263,1820,6297,297,278,18195,292,322,26188,310,278,3826,23838,310,28052,663,297,29871,29896,29955,29955,29953,29892,322,540,471,697,310,278,937,1804,414,310,278,3303,3900,20063,297,29871,29896,29955,29947,29955,29889]
    
    if True:
        for i,tk in enumerate(in_gt + out_gt):
            if tk not in sort:
                sort.append(tk)
    for v in vocab:
        if v not in set(sort):
            sort.append(v)
    vocab_rank = {}
    vocab_rank_rev = {}
    id = 0
    for i, v in enumerate(sort):
        if v not in vocab_rank:
            vocab_rank[v] = id
            id+=1
            vocab_rank_rev[vocab_rank[v]] = v
    assert id == len(vocab), f'{id} {len(vocab)}'

    timepoints = [i for i in range(num_tries)]
    cache_traces_y = np.zeros((len(vocab), len(timepoints)))
    for j in range(len(vocab)):
        for i in range(num_tries):
            cache_traces_y[vocab_rank[vocab[j]]][i] = latency[i*len(vocab) + j]
    
    hits = []
    for j in range(num_tries):
        x = np.argmin(cache_traces_y[:,j])
        if cache_traces_y[:,j][x] < 190:
            hits.append((j,x))

    print('text',tokens2text(tokenizer, list(map(lambda x: vocab_rank_rev[x[1]], hits))))

    

    s=0
    if s==0:
        colors = [(222,45,38),
                (252,146,114),
                (254,224,210),] # red
    elif s==1:
        colors = [(117,107,177),
                (188,189,220),
                (239,237,245)]
    elif s==2:
        colors = [(221,28,119),
                    (201,148,199),
                    (231,225,239)]
    elif s==3:
        colors= [(67,162,202),
                (168,221,181),
                (224,243,219)]
        
    elif  s==4:
        colors = [(28,144,153),
                (166,189,219),
                (236,226,240)]
    
    elif s==5:
        colors = [(197,27,138),
                (250,159,181),
                (253,224,221)]
    for i,c in enumerate(colors):
        colors[i] = (c[0]/255, c[1]/255, c[2]/255)
    colors = ['yellow', 'red', 'black']
    colors = [(197/255,27/255,138/255), 'white'] # deep red
    colors = [(222/255,45/255,38/255), 'white'] # deep orange
    #colors = [(28/255,144/255,153/255), 'white'] # cryan
    colors = ['blue', 'white']
    #colors = [(239/255,83/255,110/255), (236/255,180/255,71/255), (43/255,212/255,159/255)]
    cmap = LinearSegmentedColormap.from_list("my_cmap", colors)

    #cmap = 'viridis_r'
    #cmap = 'PuBu_r'#'GnBu_r'

    #for cmap in plt.colormaps():
    if True:
        print(cmap)
        #print(np.min(cache_traces_y))
        vocabs_ids = [i for i in range(len(vocab))]
        mesh = plt.pcolormesh(timepoints, vocabs_ids, cache_traces_y,
                        antialiased=False,
                        edgecolors='black', linewidths=0,
                        cmap=cmap,
                        vmin=120,vmax=280,
                        rasterized=True # avoid big file
                        )
        #OrRd_r blues_r GnBu_r PuBu_r autumn 'viridis' RdYlGn
        #print(plt.colormaps())
        #plt.xlim([144, 512])
        #plt.ylim([0, 50])
        #plt.show()
    
    if True:
        norm = Normalize(vmin=np.min(cache_traces_y), vmax=300)
        mappable = ScalarMappable(norm=norm, cmap=cmap)

        for j in range(len(vocab)):
            for i in range(num_tries):
                if cache_traces_y[j][i] < 190:
                    #plt.text(h[0], h[1], tokens2text(tokenizer, [vocab[h[1]]]).decode(),
                    #         fontsize=8, color='red', ha='center', va='center')
                    if False:
                        c = plt.Circle((i,j), 0.5, color=mappable.to_rgba(cache_traces_y[j][i]),
                                    fill=True)
                        plt.gca().add_artist(c)
                    
                    if False:
                        w=1
                        h=1
                        rect = patches.Rectangle((i-w/2,j-h/2), w, h,
                                                color=mappable.to_rgba(cache_traces_y[j][i]),
                                                fill=True, edgecolor='r', 
                                            facecolor='none', linestyle='--')  # (x, y), 宽度, 高度
                        plt.gca().add_patch(rect)

    p_start = 2
    
    y=0
    for i,t in enumerate(cache_traces_y[:,hits[p_start][0]]):
        if t < 190:
            y=max(y,i)
    # annotate the graph
   

    if False:
        w=40
        h=230#y+5
        plt.text(hits[p_start][0]-w/2-50, 2+h+10, 'Prompt', color='r')
        rect = patches.Rectangle([hits[p_start][0]-w/2, 2], w, h,
                                linewidth=0.7, edgecolor='r', 
                            facecolor='none', linestyle='--')  # (x, y), 宽度, 高度
        plt.gca().add_patch(rect)

        w=len(timepoints) - hits[3][0]-2
        gen_phase_start = hits[3][0]

        plt.text(gen_phase_start+w/2-100, 2+h+10, 'Generation', color='r')
        rect = patches.Rectangle([gen_phase_start, 2], w, h,
                                linewidth=0.7, edgecolor='r', 
                            facecolor='none', linestyle='-')  # (x, y), 宽度, 高度
        plt.gca().add_patch(rect)
    
    r_start = 4

    N = 4
    ct=0

    tx = 600
    ty=220

    if False:
        ann = plt.annotate('Prompt\nPhase', 
            fontsize=8,
            xy=(hits[p_start][0], hits[p_start][1]),
            xytext=(150, ty),
            arrowprops=dict(arrowstyle='-|>',facecolor='b',edgecolor='b',
                            linewidth=0.4),
            horizontalalignment='center')

    ax=0
    ay=0
    pts= []
    for j in range(len(vocab)):
        for i in range(num_tries):
            if cache_traces_y[j][i] < 190:
                ax=max(ax,i)
                ay=max(ay,j)
                pts.append((i,j))
    points = np.array(pts)
    midpoint = np.mean(points, axis=0)
    distances = np.linalg.norm(points - midpoint, axis=1)
    closest_index = np.argmin(distances)

    if False:
        plt.text(tx-160,ty,'Generation\nPhases',fontsize=8)
        for h in [hits[r_start], points[closest_index], (ax,ay)]:
            ann = plt.annotate('', 
                xy=(h[0], h[1]),
                xytext=(tx,ty),
                arrowprops=dict(arrowstyle='-|>',facecolor='b',edgecolor='b',
                                linewidth=0.4),
                horizontalalignment='center')
                
        

    if False:
        for h in hits[r_start:r_start+N]:
            label = "'"+tokens2text(tokenizer, [vocab_rank_rev[h[1]]]).decode()+"'"
            ann = plt.annotate(label, 
                xy=(h[0], h[1]),
                xytext=(gen_phase_start +20 + ct, 240),
                arrowprops=dict(arrowstyle='-|>',facecolor='red',edgecolor='red',
                                linewidth=0.5),
                horizontalalignment='center')
            
            ct += len(label)*25

    cbar = plt.colorbar(pad=0.03)
    cbar.set_label('Latency', rotation=270, labelpad=15)
    plt.gcf().set_size_inches(3.2, 1.5)
    plt.xlabel('Time Steps')
    plt.ylabel('Token Index')
    plt.yticks(np.arange(0,len(vocab)-1,50))
    plt.xlim([112, 260])
    plt.ylim([0, 70])
    #plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', nbins=100))
    plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    plt.savefig(f'{FIGURE_OUT_PATH}/cache_trace.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    llm_vendor,llm=VICTIM_LLM[3]
    # for attack result
    tokenizer = load_tokenizer(llm)
    plot_cache_trace(llm, tokenizer)
