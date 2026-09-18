ORANGE='#FEB46A'# 0.6 alpha of '#FE8207'
GREEN='#269D26'
BLUE='#A6C7E1' #'#2274B6'
RED='#D1221D'
CYRAN='#6DFFB3'

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import seaborn as sns

fontsize=10

if False:
    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('axes', labelsize=fontsize)

    prop_legend_font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize,
    }

FIG_OUTDIR='../draft/usenix/paper/figures'