import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from scipy.stats import pearsonr

import os
os.environ['NOT_INIT_OPENAI'] = '1'
from experiment_common import *

df = pd.read_csv('../survey/102_Privacy_Leakage_in_AI_Chat_Messages-responses-2024-11-21.csv')

vote = df.iloc[:, 37:]

# read the cos sim
cs = []
with open('../survey/cosine_similarity.csv', 'r') as fp:
    for l in fp.readlines():
        cs.append(float(l.split(',')[0]))

cos_sim = []
vote_score = []
for col_idx in range(vote.shape[1]):
    if cs[col_idx] >= 0.5: # only display the samples whose \phi > 0.5
        n_acc = 0
        n_rej = 0
        for row_idx in range(vote.shape[0]):
            v = vote.iat[row_idx,col_idx]
            if v=='Yes':
                n_acc += 1
            elif v=='No':
                n_rej += 1
            else:
                assert False, f'{row_idx}:{col_idx} {v}'
        vote_score.append(n_acc/(n_acc+n_rej))
        cos_sim.append(cs[col_idx])


sns.regplot(x=cos_sim, y=vote_score, ci=95,
    scatter_kws={"s": 25}
)


# 获取拟合直线的斜率和截距
slope, intercept = np.polyfit(cos_sim, vote_score, 1)

# 计算交点x0，即拟合直线与y=0.5的交点
y0 = 0.5
x0 = (0.5 - intercept) / slope
print(f'ASR threshold = {x0:.2f}')
# 绘制与交点x0对应的垂直线
plt.axhline(y=y0, color='gray', linestyle='--', label="y = 0.5")
plt.axvline(x=x0, color='red', linestyle='--', label=f"x = {x0:.2f}")
plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))
#plt.xticks(np.append(plt.xticks()[0], x0), rotation=45)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.annotate(f'({x0:.2f}, {y0:.2f})',  # 标注文本
            xy=(x0,y0),  # 标注点的位置
            xytext=(x0 + 0.04, y0 - 0.2),  # 文本位置
            textcoords='data',  # 文本坐标系统
            arrowprops=dict(
                 facecolor='red',  # 蓝绿色 (CadetBlue)
                 edgecolor='none', #'#79014d',  # 深石板灰 (DarkSlateGray)
                 linewidth=3,        # 细线宽更柔和
                 shrink=0.15,          # 适当缩短箭头
                 width=1.5,            # 减小尾部宽度
                headlength=8,         # 减小箭头头部长度
                 headwidth=8,         # 中等头部尺寸
                 alpha=0.7,            # 增加透明度
                 linestyle='-'        # 虚线边框
             ),
            fontsize=7,  # 字体大小
            color='red')  # 文本颜色

plt.ylim([0,1])
plt.xlim([min(cos_sim),max(cos_sim)])
print('min cos_sim = ', min(cos_sim))



if False:
    plt.scatter(cos_sim, vote_score)

    #calculate equation for trendline
    z = np.polyfit (cos_sim, vote_score, 1)
    p = np.poly1d (z)

    #add trendline to plot
    plt. plot (cos_sim, p(cos_sim))
    plt.ylim([0,1])
    plt.xlim([0.4,1])

plt.ylabel('Privacy Exposure (Average Vote)')
plt.xlabel(r'The $\phi$ between attack result and ground-truth')
plt.gca().grid(True, which="both", linestyle='--', linewidth=0.5)

plt.gcf().set_size_inches(4,2)

pearson_correlation, p_value = pearsonr(cos_sim, vote_score)
print("Pearson Correlation (scipy):", pearson_correlation)

plt.savefig(f'{FIGURE_OUT_PATH}/survey.pdf', bbox_inches='tight')
plt.show()