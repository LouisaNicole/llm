

# !!! require python > 3.7
# I am using 3.10.8
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, ScalarFormatter
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import math
from collections import defaultdict
import base64
import tiktoken.load
from tqdm import tqdm
import bisect

import scipy.signal
from scipy.interpolate import interp1d, CubicSpline
import scipy
from scipy.fft import fft
import scipy.signal as signal
import scipy.stats as stats
import Levenshtein
from rouge import Rouge
import sys
sys.setrecursionlimit(10000)  # make rouge working
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
import tiktoken # for token counting
from openai import OpenAI

import os
import copy
import random
from enum import Enum
from collections import OrderedDict
import argparse
from io import BytesIO

import difflib
from nltk.util import ngrams

from transformers import AutoTokenizer

from myjsonl import *

def create_directories(path):
    os.makedirs(path, exist_ok=True)

def compute_psd(signal, fs, nperseg, nfft, noverlap=None, window='hann', show_prog=False):
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap
    window = scipy.signal.get_window(window, nperseg)
    num_segments = (len(signal) - noverlap) // step
    assert num_segments > 0
    #print(f'num_segments = {num_segments}')
    psd = np.zeros(nfft // 2 + 1)

    for i in tqdm(range(num_segments), disable=not show_prog):
        start = i * step
        end = start + nperseg
        segment = signal[start:end] * window
        fft_segment = np.fft.rfft(segment, n=nfft)
        psd += np.abs(fft_segment) ** 2

    psd /= num_segments * nfft
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    return freqs, psd



# params for Intel 13900K
CRYSTAL_FREQ_RATIO = 78
CRYSTAL_FREQ = 38.4e6 # Hz
TSC_FREQ = CRYSTAL_FREQ * CRYSTAL_FREQ_RATIO
TSC_PERIOD = 1/TSC_FREQ # s
#print(f'TSC freq = {TSC_FREQ/1e9} GHz')

from dynamicavg import DynamicAvg

def dft(timepoints, f, nfft):
    s = 0
    omeage = 2*math.pi
    ct = 0
    for i in range(nfft):
        t = timepoints[i % len(timepoints)]
        s += np.exp(-1j * omeage *f * t)
    return s

def sampling(timepoints, fs):
    n_samples = math.ceil((timepoints[-1] - timepoints[0]) * fs)
    samples = np.zeros(n_samples+1)
    for t in timepoints:
        i = round((t-timepoints[0])*fs)
        samples[i] = 1
    # eliminate DC component
    samples -= np.mean(samples)

    # get time stamps
    sample_period = 1/fs
    samples_t = [timepoints[0] + t*sample_period for t in range(len(samples))]
    return samples_t, samples

def sampling_with_rev(timepoints, fs):
    n_samples = math.ceil((timepoints[-1] - timepoints[0]) * fs)
    samples = np.zeros(n_samples+1)
    rev = [None] * len(timepoints)
    for k, t in enumerate(timepoints):
        i = round((t-timepoints[0])*fs)
        samples[i] = 1
        rev[k] = i
    # eliminate DC component
    samples -= np.mean(samples)

    # get time stamps
    sample_period = 1/fs
    samples_t = [timepoints[0] + t*sample_period for t in range(len(samples))]
    return samples_t, samples, rev

from sortedcontainers import SortedSet

def find_closest(sorted_set, x):
    """
    找到有序集合中与给定浮点数 x 最接近的数
    :param sorted_set: 有序集合 (SortedSet)
    :param x: 给定浮点数
    :return: 与 x 最接近的数
    """
    # 找到 x 应该插入的位置
    if isinstance(sorted_set, np.ndarray):
        pos = bisect.bisect_left(sorted_set, x)
    else: # SortedSet
        print(type(sorted_set))
        pos = sorted_set.bisect_left(x)
    
    # 比较 pos 位置和相邻位置的值，确保找到最接近的
    if pos == 0:
        return sorted_set[0]  # x 小于等于所有元素
    elif pos == len(sorted_set):
        return sorted_set[-1]  # x 大于所有元素
    else:
        # 比较插入点的前后两个元素
        before = sorted_set[pos - 1]
        after = sorted_set[pos]
        return before if abs(before - x) <= abs(after - x) else after

def find_closest_indice(sorted_set, x):
    """
    找到有序集合中与给定浮点数 x 最接近的数
    :param sorted_set: 有序集合 (SortedSet)
    :param x: 给定浮点数
    :return: 与 x 最接近的数的indice
    """
    # 找到 x 应该插入的位置
    if isinstance(sorted_set, np.ndarray):
        pos = bisect.bisect_left(sorted_set, x)
    else: # SortedSet
        print(type(sorted_set))
        pos = sorted_set.bisect_left(x)
    
    # 比较 pos 位置和相邻位置的值，确保找到最接近的
    if pos == 0:
        return 0  # x 小于等于所有元素
    elif pos == len(sorted_set):
        return -1  # x 大于所有元素
    else:
        # 比较插入点的前后两个元素
        before = sorted_set[pos - 1]
        after = sorted_set[pos]
        return pos-1 if abs(before - x) <= abs(after - x) else pos


def plot_PSD(timepoints, ax, fs, nfft, nperseg, markpeek=None, markwidth=None, markpeek_opt=None, unit='KHz', use_dBm = True, color='tab:blue'):
    timepoints -= timepoints[0]
    samples_t, samples = sampling(timepoints, fs)
    
    print('n_samples = ', len(samples))
    print(f'fs={fs}, nperseg={nperseg}, nfft={nfft}')

    f, Pxx_den = compute_psd(samples, fs, nperseg=nperseg, nfft=nfft)
    if use_dBm:
        Pxx_den_dBm = 10 * np.log10(Pxx_den * 1e3)
    else:
        Pxx_den_dBm = Pxx_den * 1e3
    ax.plot(f, Pxx_den_dBm, color=color)

    if markpeek is not None:
        peak, _ = signal.find_peaks(Pxx_den_dBm,
                                    height=np.max(Pxx_den_dBm)*markpeek, width=markwidth)
        def get_label(freq):
            if freq > 1000:
                return f'{freq/1e3:.1f}'
            else:
                return f'{freq:.0f}'
            
        if markpeek_opt == 'arrow_first':
            if len(peak):
                ax.annotate(get_label(f[peak[0]]), 
                    xy=(f[peak[0]], Pxx_den_dBm[peak[0]]),
                    xytext=(f[peak[0]]+100e3, Pxx_den_dBm[peak[0]]),
                    arrowprops=dict(arrowstyle='-|>',facecolor='red',edgecolor='red'),
                    horizontalalignment='center')
                ax.scatter(f[peak[0]], Pxx_den_dBm[peak[0]], marker='o', color='red')

        else:
            ax.scatter(f[peak], Pxx_den_dBm[peak], marker='o', color='red')
            ps = SortedSet([0])
            for p in peak:
                if abs(f[p]-find_closest(ps, f[p])) < 10:
                    ps.add(f[p])
                    continue
                ps.add(f[p])
                ax.text(f[p]-30, Pxx_den_dBm[p]+4, get_label(f[p]),
                        fontsize=8)

        ax.set_ylim([min(Pxx_den_dBm), max(Pxx_den_dBm) + abs(max(Pxx_den_dBm))*0.3]) # reserve sapce fro label

    ax.set_xlabel(f'Frequency [{unit}]')#,fontsize=9)
    ax.set_ylabel('PSD [dBm/Hz]' if use_dBm else 'PSD [mW/Hz]')#,fontsize=9)
    ax.set_xlim([-max(f)*0.01,max(f)])

    def hz_formatter(x, pos):
        if unit == 'KHz':
            return '{:.1f}'.format(x / 1e3)
        elif unit == 'Hz':
            return '{:.1f}'.format(x)
        else:
            assert False
    ax.xaxis.set_major_formatter(hz_formatter)
    
    return samples_t, samples, f, Pxx_den_dBm


def plot_sfft(timepoints, ax, fs, nfft, nperseg, markpeek=False, unit='KHz'):
    #timepoints -= timepoints[0]
    samples_t, samples = sampling(timepoints, fs)
    
    print('n_samples = ', len(samples))
    print(f'fs={fs}, nperseg={nperseg}, nfft={nfft}')

    f, t, Zxx = signal.stft(samples, fs=fs, nperseg=nperseg)

    ax.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    #ax.line([t[0] + i*256 for i in range(len(t)/256)])
    ax.set_title('Short-Time Fourier Transform (STFT)')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    #ax.colorbar(label='Magnitude')
    #plt.ylim(0, 200)  # 限制 y 轴范围

    return samples_t, samples, f, Zxx

def load_tokenizer(founda_model):
    if is_osllm(founda_model):
        for llm_vendor,llm in VICTIM_LLM:
            if llm==founda_model:
                break
        else:
            assert False, f'unknown founda model {founda_model}'
        tokenizer_path = f'{TOKENIZER_PATH}/{llm_vendor}/{llm}'
        return AutoTokenizer.from_pretrained(tokenizer_path,
                                            add_prefix_space =False # need to avoiding wronly remove the leading space for certain tokenizers
                                            )
    else:
        return tiktoken.encoding_for_model(founda_model)

def tokens2text(tokenzier, tokens):
    if isinstance(tokenzier, tiktoken.Encoding):
        return tokenzier.decode(tokens)
    else:
        return tokenzier.decode(tokens, skip_special_tokens=True).encode()

def text2tokens(tokenizer, text):
    if isinstance(tokenizer, tiktoken.Encoding):
        return tokenizer.encode(text)
    else:
        return tokenizer.encode(text, add_special_tokens=False)

def plot_timeline(tokenizer, timepoints, tokens, ax):
    for i, timestamp in enumerate(timepoints):
        ax.scatter([timestamp], [1])
        if tokenizer is not None:
            ax.text(timestamp, 1, tokens2text(tokenizer, [tokens[i]]), verticalalignment='bottom', rotation=90)
        #print("'"+token_decoder.detoken(token)+"'" , end=',')
    #ax.set_xlim([0, max(samples_t)])
    ax.set_ylim([0,3])

def plot_timing_signal(timepoints, ax, value,label='', color='b', marker='o',markersize=None):
    if False:
        markerline, stemline, baseline, = ax.stem(timepoints-timepoints[0],[value]*len(timepoints),linefmt=color+'-',markerfmt=color+'o',basefmt=color+'.',
        label=label)
        plt.setp(stemline, linewidth = 1.25)
        plt.setp(markerline, markersize = 4)
    ax.scatter(timepoints-timepoints[0], [value]*len(timepoints), marker=marker, facecolors='none' if marker=='o' else color, edgecolors=color, label=label, s=markersize)

def plot_tbe_hist(timepoints, ax):
    timepoints -= timepoints[0]
    # get the smaple frequency
    tbe = timepoints[1:] - timepoints[:-1]
    ax.hist(10*np.log10(tbe), bins=10)

def split_prompt_response(timepoints, tokens, ret_n_clusters=False):
    prompt_start, prompt_end, response_start, response_end, _, _, n_clusters = identify_prompt_resp(timepoints)

    assert prompt_start != None, f'failed to detect the prompt start'
    assert response_start != None, f'failed to detect the response start'

    #print('prompt_start=', prompt_start, ' response_start = ', response_start)
    ret = [(timepoints[prompt_start:prompt_end+1],
                tokens[prompt_start:prompt_end+1]), \
            (timepoints[response_start:response_end],
                tokens[response_start:response_end])]
    if ret_n_clusters:
        return ret + [n_clusters]
    else:
        return ret


def get_tp_tk(sel_record):
    timepoints = np.array([timestamp for tk,latency,timestamp in sel_record['e']])
    timepoints -= timepoints[0]
    timepoints = np.array(timepoints, dtype=np.float64)
    timepoints *= TSC_PERIOD # to seconds
    tokens = [tk for tk,latency,timestamp in sel_record['e']]
    latency = np.array([latency for tk,latency,timestamp in sel_record['e']])
    return timepoints, tokens, latency

def get_precision_recall_f1(groundtruth_tokens, detected_tokens):
    groundtruth_set = set(groundtruth_tokens)
    detected_set = set(detected_tokens)
    TP = len(groundtruth_set & detected_set)
    FN = len(groundtruth_set) - len(groundtruth_set & detected_set) # check this?
    FP = len(detected_set) - len(groundtruth_set & detected_set)
    precision = (TP) / (TP + FP + 1e-20)
    recall = (TP) / (TP + FN + 1e-20)
    try:
        f1 = (2 * precision * recall) / (precision + recall + 1e-20)
    except ZeroDivisionError:
        f1 = 0.0
    return precision, recall, f1

class IndicedValue():
    def __init__(self, ind,v):
        self.ind = ind
        self.v = v
    def __eq__(self, a):
        return self.v == a.v

def longest_common_subsequence(X, Y):
    m = len(X)
    n = len(Y)

    X = [IndicedValue(i, v) for i,v in enumerate(X)]
    Y = [IndicedValue(i, v) for i,v in enumerate(Y)]
    
    L = [[0] * (n + 1) for i in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    
    # trace back
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs.append((X[i - 1], Y[j - 1]))
            i -= 1
            j -= 1
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    lcs.reverse()
    return lcs

def split_false_positives_tokens_indices(tokens, grounttruth_tokens):
    lcs = longest_common_subsequence(tokens, grounttruth_tokens)
    n_fp = len(tokens) - len(lcs)
    
    tp_indices = [o[0].ind for o in lcs]
    tp_indices_set = set(tp_indices)
    fp_indices = list(filter(lambda x : x not in tp_indices_set, range(len(tokens))))
    return tp_indices, fp_indices

def split_false_positives(timepoints, tokens, grounttruth_tokens):
    tp_indices, fp_indices = split_false_positives_tokens_indices(tokens, grounttruth_tokens)
    tokens = np.array(tokens)
    tp_timepoints = timepoints[tp_indices]
    tp_tokens = tokens[tp_indices]
    fp_timepoints = timepoints[fp_indices]
    fp_tokens = tokens[fp_indices]
    return (tp_timepoints, tp_tokens), (fp_timepoints, fp_tokens)

def get_false_negatives(timepoints, tokens, groundtruth_tokens):

    # compute the groundtruth by LCS
    lcs = longest_common_subsequence(tokens, groundtruth_tokens)
    # get false negative groundtruth
    gt_missing = []
    gt_mising_timeinterval = []
    results = []
    last_tk_resp_pos = -1
    last_gt_pos = -1
    for o in lcs:
        #         gt_con gt_ncon
        # tk_con   tp       fn
        # tk_ncon  fp      fn+fp

        # tk has skipped some tokens
        if last_tk_resp_pos + 1 != o[0].ind:
            # fp
            pass 
        # gt has skipped some tokens
        if last_gt_pos + 1 != o[1].ind:
            # missing
            gt_missing.append((last_tk_resp_pos, o[0].ind))
            tinv = (timepoints[last_tk_resp_pos] if last_tk_resp_pos != -1 else 0,
                        timepoints[o[0].ind])
            gt_mising_timeinterval.append(tinv)
            results.append((tinv[0]+tinv[1])/2)
        last_tk_resp_pos = o[0].ind
        last_gt_pos = o[1].ind
    
    return results


def get_sample_params(timepoints, fs_mul=8):
    tbe = timepoints[1:] - timepoints[:-1]
    fmax = 1/np.mean(tbe)
    fs = fs_mul*fmax
    sample_period = 1/fs
    n_samples = round(fs * (timepoints[-1]-timepoints[0]))
    nfft = n_samples
    nperseg = round(n_samples/3)
    return fs, nfft, nperseg, fmax

primes = None
def sieve_of_eratosthenes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    for number in range(2, int(limit**0.5) + 1):
        if is_prime[number]:
            for multiple in range(number * number, limit + 1, number):
                is_prime[multiple] = False

    return [num for num, prime in enumerate(is_prime) if prime]

def spectral_centroid(s, freqs):
    return np.dot(freqs, s) / np.sum(s)

def spectral_spread(s, freqs):
    u1 = spectral_centroid(s, freqs)
    return np.sqrt(np.dot((freqs-u1)**2, s) / np.sum(s))

def spectral_kurtosis(s, freqs, b1=0, b2=None):
    if b2 is None:
        b2 = len(s)
    u1 = spectral_centroid(s[b1:b2], freqs[b1:b2])
    u2 = spectral_spread(s[b1:b2], freqs[b1:b2])
    sm = np.sum(s[b1:b2]) 
    v = np.dot((freqs[b1:b2] - u1)**4, s[b1:b2]) / ((u2**4) * sm)
    if v > 1:
        print(sm, u2, u1, freqs[b1:b2], s[b1:b2] )
    return v

def estimate_f0(timepoints, tokens=None, tokenizer=None):
    # estimate the primary frequencey using PSD and SWIPE algorithm
    tbe = timepoints[1:] - timepoints[:-1]
    if len(timepoints) > 8:
        fs, nfft, nperseg, _ = get_sample_params(timepoints, fs_mul=8)
        sample_t, samples, sample_rev = sampling_with_rev(timepoints, fs)

        global primes
        if primes is None:
            primes = [1] + sieve_of_eratosthenes(10000)
        # remove DC componetns
        def butter_highass(cutoff, fs, order=5):
            nyq = 0.5 * fs  # Nyquist freq
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='high', analog=False) 
            return b, a
        b, a = butter_highass(10, fs, order=5)
        samples = signal.lfilter(b, a, samples)
        freqs, psd = compute_psd(samples, fs, nperseg, nfft)
        peak, _ = signal.find_peaks(psd, height=np.mean(psd))

        # search for the base frequency
        max_harmonic_sum = 0
        base_freq_ind = None
        # local search
        for p in peak:
            mul = 0
            harmonic_sum = 0
            while True:
                k = p * primes[mul]
                if round(k+p/2) >= len(psd):
                    break
                assert round(k-p/2) >= 0
                harmonic_sum += psd[k] - 0.5*(psd[round(k-p/2)] + psd[round(k+p/2)])
                mul += 1
            #print(f'{freqs[p]} {harmonic_sum}')
            if harmonic_sum > max_harmonic_sum:
                max_harmonic_sum = harmonic_sum
                base_freq_ind = p
        # get the phase of the base frequency
        if base_freq_ind is None:
            f0 = np.median(tbe)
        f0 =  freqs[base_freq_ind]
        print(f'f0 = {f0}')
    elif len(tbe) > 1:
        f0 = 1/np.median(tbe)
    else:
        f0 = 0
    return f0



MAX_TOL_FN = 4
def match_dense_cluster(tbe, MAX_BACTHED_TBE):
    K=4 # min length of prompt and responses
    segment_start = None
    for s in range(len(tbe)-K+1):
        found = True
        for k in range(K-1):
            if tbe[s+k] > MAX_BACTHED_TBE:
                found = False
                break
        if found:
            segment_start = s
            break
    if segment_start is None:
        return None, None

    segment_end = len(tbe)-1
    for s in range(segment_start, len(tbe)):
        if tbe[s] > MAX_BACTHED_TBE:
            segment_end = s
            break
    return segment_start, segment_end

def count_period(tbe, start, end, t0):
    cnt = 0
    for i in range(start, end):
        if abs(tbe[i]) < t0 + t0/2:
            cnt += 1
    return cnt

def find_prompt_boundary(timepoints, t0):
    tbe = timepoints[1:] - timepoints[:-1]
    start = 0
    clusters = []
    while start < len(tbe):
        seg_start, seg_end = match_dense_cluster(tbe[start:], t0/10 if t0 else 1e-3)
        if seg_start is None or (
                start and t0 and count_period(tbe, start, start + seg_start, t0) > 4): # avoid maching dense clusters stray in the response
            break
        clusters.append((start + seg_start, start + seg_end))
        start += seg_end + 1
    
    assert len(clusters) > 0, f'failed to detect the response start'
    if len(clusters) != 1:
        print('Dense clusters: ', len(clusters))

    return clusters[0][0], clusters[-1][1], len(clusters) # merge all the batches
    
def estimate_t0_with_both_prompt_and_resp(tbe):
    t0 = None
    if len(tbe) > 1:
        bin_quant=100
        cnt = defaultdict(int)
        for t in tbe:
            cnt[round(t*bin_quant)] += 1
        
        # find the two peaks in the distribution
        sorted_items = sorted(cnt.items(), key=lambda item: item[1], reverse=True)
        if False: # DEBUG
            s2 = sorted(cnt.items(), key=lambda item: item[1])
            plt.scatter(np.array([x for x,_ in s2]) / bin_quant, [y for _,y in s2])
            plt.show()
        max_key = sorted_items[0][0]
        second_max_key = sorted_items[1][0]
        if max_key > second_max_key:
            support = cnt[max_key]
        else:
            support = cnt[second_max_key]
        if support > 1:
            t0 = max(max_key, second_max_key) / bin_quant # find the period of the decoding slower than prefilling
            print('t0 = ',t0)
    return t0

def identify_prompt_resp(timepoints):
    tbe = timepoints[1:] - timepoints[:-1]
    t0 = estimate_t0_with_both_prompt_and_resp(tbe) 
    
    p_start, p_end, n_clusters = find_prompt_boundary(timepoints, t0)
    assert p_start is not None, 'failed to identify prompt'

    r_start = p_end+1
    r_end = len(timepoints)
    if r_start+1 < len(tbe):
        # detect the ending
        ema = tbe[r_start]
        alpha = 0.9
        for p in range(r_start+1, len(tbe)):
            #print(p,len(timepoints), tbe[p], (ema / (1-alpha**(p+1))) * MAX_TOL_FN)
            if (tbe[p] > (ema / (1-alpha**(p+1))) * MAX_TOL_FN
                    and p/(len(tbe)-1) > 0.95): # avoid truncate from the spare noise between the prompt and response
                r_end = p +1
                print(f'cut {r_end} / {len(timepoints)}')
                break
            ema = ema * alpha + tbe[p] * (1-alpha)
        
        if False and t0 is not None:
            #spacing = timepoints[r_start] + (timepoints[-1]-timepoints[r_start])*2/3 
            for p in range(r_start, len(tbe)):
                #print(tbe[p], t0 * MAX_TOL_FN ,count_period(tbe, r_start, p, t0) > 4)
                if tbe[p] > t0 * MAX_TOL_FN and (count_period(tbe, r_start, p, t0) > 4): # avoid truncate from the spare noise between the prompt and response
                    r_end = p+1
                    print(f'cut {r_end} / {len(timepoints)}')
                    break
            #exit(1)
    return p_start, p_end, r_start, r_end, None, None, n_clusters
    
    
        


def verify_identify_prompt_resp(ds, tokenizer):
    n_asr = 0
    n_p_pr = DynamicAvg()
    n_tires = 0
    for r, record in enumerate(tqdm(ds['results'])):
        timepoints, tokens, _ = get_tp_tk(record)

        p_start, p_end, r_start, corr, thr = identify_prompt_resp(timepoints)

        n_matched = False
        if len(set(tokens[p_start:p_end+1])):
            n_p_tp = len(set(record['g']['i']) & set(tokens[p_start:p_end+1]))
            n_p_pr.update(n_p_tp / len(set(tokens[p_start:p_end+1])))

            for k in range(1): # skip false positive
                if tokens[r_start+k] == record['g']['o'][0]:
                    n_matched = True
                    break
        else:
            n_p_pr.update(0)

        if n_matched:
            n_asr += 1
        elif False:
            fig, ax = plt.subplots(2,1, sharex=True)

            plot_timeline(tokenizer, timepoints, tokens, ax[0])

            ax[1].stem(timepoints[p_end+1:], corr)
            ax[1].axhline(y=thr, color='r', linewidth=2, linestyle='--')
            ax[0].axvline(x=timepoints[r_start], color='r')
            ax[0].axvline(x=timepoints[p_start], color='b')
            ax[0].axvline(x=timepoints[p_end], color='g')

            plt.show()
        n_tires += 1
    print(f'Find Resp Start ASR = {n_asr/n_tires*100:.2f}%')
    print(f'Avg Prompt token precision {n_p_pr.get()*100:.2f}%')


def plot_sfft(ds, tokenizer):
    # find the record with the most false postive, to ensure
    # adequate evalutaing samples
    max_nr_fp = 0
    for r, record in enumerate(tqdm(ds['results'])):
        #if r == 478:
        #    continue
        timepoints, tokens, _ = get_tp_tk(record)
        p_start, p_end, r_start, _ = identify_prompt_resp(timepoints)
        tk_resp = tokens[r_start:]

        FP = len(set(tk_resp)) - len(set(record['g']['o']) & set(tk_resp))
        if FP > max_nr_fp:
            max_nr_fp = FP
            sel_record = r
    sel_record = 1921
    print(f'Selected {sel_record}-th record, with {max_nr_fp} false positives')

    record = ds['results'][sel_record]
    timepoints, tokens, _ = get_tp_tk(record)

    print('GT:', tokens2text(tokenizer, record['g']['o']))

    fig, ax = plt.subplots(4,1, sharex=True)
    fs, nfft, nperseg, _ = get_sample_params(timepoints, freq_res=0.1, fs_mul=8)
    sfft_nperseg=256
    plot_sfft(timepoints, ax[0], fs, nfft, nperseg=sfft_nperseg, markpeek=True, unit='Hz')
    plt.show()

def figure_PSD_for_resp(ds, tokenizer):
    if False:
        # find the record with the most false postive, to ensure
        # adequate evalutaing samples
        max_nr_fp = 0
        for r, record in enumerate(tqdm(ds['results'])):
            #if r == 478:
            #    continue
            timepoints, tokens, _ = get_tp_tk(record)
            p_start, p_end, r_start, _ = identify_prompt_resp(timepoints)
            tk_resp = tokens[r_start:]

            FP = len(set(tk_resp)) - len(set(record['g']['o']) & set(tk_resp))
            if FP > max_nr_fp:
                max_nr_fp = FP
                sel_record = r
        print(f'Selected {sel_record}-th record, with {max_nr_fp} false positives')

    sel_record = 1921
    
    record = ds['results'][sel_record]
    timepoints, tokens, _ = get_tp_tk(record)
    p_start, p_end, r_start, _,_ = identify_prompt_resp(timepoints)
    tp_resp = timepoints[r_start:]
    tk_resp = tokens[r_start:]

    resp_groundtruth = record['g']['o']
    (tp_timepoints, tp_tokens), (fp_timepoints, fp_tokens) = split_false_positives(tp_resp, tk_resp, resp_groundtruth)
    if True:
        fig, ax = plt.subplots(2,1, sharex=True)
        print(tokens2text(tokenizer, resp_groundtruth))
        print('--')
        print(tokens2text(tokenizer, tp_tokens))
    
        plot_timeline(tokenizer, tp_timepoints, tp_tokens, ax[0])
        plot_timeline(tokenizer, fp_timepoints, fp_tokens, ax[1])
        #plt.show()

    fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
    fs, nfft, nperseg, fmax = get_sample_params(tp_timepoints)
    print(f'nperseg = {nperseg}')
    print(f'nfft = {nfft}')
    print(f'fmax = {fmax/1e3} KHz')
    print(f'fs = {fs/1e3} KHz')
    freq_res = 1/(timepoints[-1]-timepoints[0])
    print(f'Freq resolution = {freq_res} Hz')

    _, _,_, p1 = plot_PSD(tp_timepoints, ax[0], fs, nfft, nperseg, markpeek=0.8, unit='Hz')
    _, _,_, p2 = plot_PSD(fp_timepoints, ax[1], fs, nfft, nperseg, unit="Hz")
    a,b = min(np.min(p1), np.min(p2)), max(np.max(p1), np.max(p2))+10
    ax[0].set_ylim((a,b))
    ax[1].set_ylim((a,b))
    plt.show()

def find_elements_in_range(nums, L, R):
    def find_left_bound(L):
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] < L:
                left = mid + 1
            else:
                right = mid
        return left if left < len(nums) and nums[left] >= L else -1

    def find_right_bound(R):
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right + 1) // 2
            if nums[mid] > R:
                right = mid - 1
            else:
                left = mid
        return right if right >= 0 and nums[right] <= R else -1

    left = find_left_bound(L)
    right = find_right_bound(R)
    
    if left == -1 or right == -1 or left > right:
        return [], None, None
    else:
        return nums[left:right+1], left, right

def find_consecutive_sequences(seq):
    if not seq: 
        return []

    result = [] 
    current_seq = [0] 

    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]: 
            current_seq.append(i)
        else:
            result.append(current_seq)
            current_seq = [i] 

    if len(current_seq) > 1:
        result.append(current_seq)
    return result

def find_consecutive_sequences_value(seq, key=lambda x:x):
    if not seq: 
        return []

    result = [] 
    current_seq = [seq[0]] 

    for i in range(1, len(seq)):
        if key(seq[i]) == key(seq[i - 1]): 
            current_seq.append(seq[i])
        else:
            result.append(current_seq)
            current_seq = [seq[i]] 

    if len(current_seq) > 1:
        result.append(current_seq)
    return result

class Label(Enum):
    LABEL_TP = 0
    LABEL_FN = 1
    LABEL_SUS = 2
    LABEL_FP = 3

class LabeledData:
    def __init__(self, type, indice=None):
        self.type = type
        self.indice = indice

def extract_resp_tokens(timepoints, tokens, tokenizer):
    f0 = estimate_f0(timepoints, tokens, tokenizer)
    print(f"Estimated f0: {f0}")

    if False:
        # 1. blank T error acc
        def match(start, T, neighbor):
            if False:
                #assert neighbor < T, f'{T} {neighbor}'
                n_accepted = 0
                results = defaultdict(list)
                i = start
                results[timepoints[i]].append(i)
                next_pt = (timepoints[i] + T)
                while i < len(timepoints)-1:
                    if abs(timepoints[i+1] - next_pt) < neighbor:
                        # accept
                        results[timepoints[i+1]].append(i+1)
                        n_accepted += 1
                        next_pt = timepoints[i+1] + T
                        i += 1
                    elif timepoints[i+1] < next_pt:
                        # merge
                        results[timepoints[i]].append(i+1)
                        i += 1
                    elif timepoints[i+1] > next_pt:
                        # blank
                        #results[timepoints[i]+T].append(None)
                        results[next_pt].append(None)
                        next_pt += T
                return n_accepted, results
            else:
                results = dict()
                visited = [False] * len(timepoints)
                p = timepoints[start] + T
                results[timepoints[start]] = LabeledData(Label.LABEL_TP, [start])
                visited[start] = True
                n_accepted = 0
                while p < timepoints[-1] + neighbor/2:
                    l = p - neighbor/2
                    r = p + neighbor/2
                    pred, starti, endi = find_elements_in_range(timepoints, l,r)
                    if len(pred) == 1:
                        assert starti == endi
                        assert pred[0] not in results
                        results[pred[0]] = LabeledData(Label.LABEL_TP, [starti])
                        visited[endi] = True
                        n_accepted += 1
                        p = pred[0] + T
                    elif len(pred) > 1:
                        # merge to mean time
                        tm = np.mean(pred)
                        assert tm not in results
                        results[tm] = LabeledData(Label.LABEL_SUS, [i for i in range(starti, endi+1)])
                        for i in range(starti, endi+1):
                            visited[i] = True
                        p = tm + T
                        n_accepted += 1
                    else:
                        # blank
                        assert p not in results
                        results[p] = LabeledData(Label.LABEL_FN)
                        p += T
                
                # get false positives
                suspicious_indices = find_consecutive_sequences(visited)
                for ss in suspicious_indices:
                    if visited[ss[0]]:
                        continue
                    # merge
                    assert timepoints[ss[0]] not in results
                    results[timepoints[ss[0]]] = LabeledData(Label.LABEL_SUS, ss)
            
                
            return n_accepted, results

        period = 1/f0

        # select the neighbor
        u = np.median(tbe - period)
        sigma = np.std(tbe - period)
        n = period/3#u + sigma
        #assert n < period/2

        # search for the start point
        max_n_accepted = 0
        for start in range(0, math.ceil(len(timepoints)/3)):
            nac, ret = match(start, period, n)
            if nac > max_n_accepted:
                max_n_accepted = nac
                best_start = start
                best_solution = ret
            
    
    elif True:
        # 1. START point error  (hard limit of max missing)
        # 2. misclassify fp as tp, if the fp is the first occurs, acrossing
        #   some fn
        T = 1/f0
        neighbor = T/8

        def match(start):
            p = timepoints[start]
            labels = dict()
            labels[timepoints[start]] = LabeledData(Label.LABEL_TP, start)
            n_accept = 0
            for i in range(start+1,len(timepoints)):
                tbe = timepoints[i] - p
                if tbe < neighbor:
                    # now we unsure whether the previous prediciton is correct
                    # so modify it
                    labels[p].type = Label.LABEL_SUS
                    labels[timepoints[i]] = LabeledData(Label.LABEL_SUS, i)
                elif tbe > neighbor and tbe < T-neighbor:
                    # sus
                    labels[timepoints[i]] = LabeledData(Label.LABEL_FP, i)
                else:
                    # accept
                    p = timepoints[i]
                    labels[p] = LabeledData(Label.LABEL_TP, i)
                    n_accept += 1
            return n_accept, labels
        
        # search for the start point
        max_n_accepted = 0
        for start in range(0, math.ceil(len(timepoints)/3)):
            nac, labels = match(start)
            if nac > max_n_accepted:
                max_n_accepted = nac
                best_start = start
                best_labels = labels
        
        # predict the false negatives
        N_ENDING_MAX_MISSING = 6
        last_t = None
        for i in range(best_start,len(timepoints)):
            # to predict FN close to FP
            if timepoints[i] in labels and \
                    labels[timepoints[i]].type == Label.LABEL_FP:
                continue
            if last_t is not None:
                tbe = timepoints[i] - last_t
                n_fn = round(tbe / T)
                if n_fn >= 2 and n_fn < N_ENDING_MAX_MISSING:
                    ct = last_t + T
                    n_pt = 0
                    while ct < timepoints[i] and n_pt < n_fn-1:
                        assert ct not in best_labels
                        best_labels[ct] = LabeledData(Label.LABEL_FN)
                        n_pt += 1
                        ct += T
            last_t = timepoints[i]
        

    else:
        # 1. blank T erro acc
        T = 1/f0
        neighbor = T/7

        def match(start, p):
            labels = dict()
            n_accept = 0

            for i in range(start+1,len(timepoints)):
                tbe = timepoints[i] - p
                if tbe < neighbor:
                    # now we unsure whether the previous prediciton is correct
                    # so modify it
                    labels[p].type = Label.LABEL_SUS
                    labels[timepoints[i]] = LabeledData(Label.LABEL_SUS, i)
                elif tbe > neighbor and tbe < T-neighbor:
                    # sus
                    labels[timepoints[i]] = LabeledData(Label.LABEL_FP, i)
                else:
                    err = tbe / T - int(tbe / T)
                    if err < neighbor/T or err > 1 - neighbor/T:
                        # accept
                        p = timepoints[i]
                        labels[p] = LabeledData(Label.LABEL_TP, i)
                        n_accept += 1
                    else:
                        labels[p] = LabeledData(Label.LABEL_FP, i)
                        p += T # acc err
            return n_accept, labels
        
        # search for the start point
        max_n_accepted = 0
        for start in range(0, math.ceil(len(timepoints)/3)):
            nac, labels = match(start)
            if nac > max_n_accepted:
                max_n_accepted = nac
                best_start = start
                best_labels = labels
        
        # predict the false negatives
        N_ENDING_MAX_MISSING = 6
        last_t = None
        for i in range(best_start,len(timepoints)):
            if timepoints[i] in labels and \
                    labels[timepoints[i]].type == Label.LABEL_FP:
                continue
            if last_t is not None:
                tbe = timepoints[i] - last_t
                n_fn = round(tbe / T)
                if n_fn >= 2 and n_fn < N_ENDING_MAX_MISSING:
                    ct = last_t + T
                    n_pt = 0
                    while ct < timepoints[i] and n_pt < n_fn-1:
                        assert ct not in best_labels
                        best_labels[ct] = LabeledData(Label.LABEL_FN)
                        n_pt += 1
                        ct += T
            last_t = timepoints[i]
        
    # sort by time points
    best_solution = OrderedDict(sorted(best_labels.items(),
                                       key=lambda x:x[0]))

    print(f'estimated start: {best_start} accepted {max_n_accepted} / total {len(timepoints)}')

    if True:
        # detete suspicious endings with a long tailing of FP
        half_time = (timepoints[0] + timepoints[-1]) / 2
        consecutive_fn = 0
        kvs = copy.deepcopy(list(best_solution.items()))
        print(len(kvs), len(best_solution))
        for i, (timestamp, label) in enumerate(kvs):
            if timestamp >= half_time:
                if label.type ==Label.LABEL_FN:
                    consecutive_fn += 1
                    if consecutive_fn == 1:
                        consecutive_fn_begin = i
                        first_t = timestamp
                    if consecutive_fn > N_ENDING_MAX_MISSING:
                        # reserve one
                        #best_solution[first_t] = LabeledData(Label.LABEL_FN)

                        for j in range(consecutive_fn_begin, len(kvs)):
                            if kvs[j][1].type != Label.LABEL_FN:
                                break
                        
                        # check whether we have deleted some potenpotentially usefulf tokens
                        n_useful = 0
                        for k in range(j,len(kvs)):
                            if (kvs[k][1].type != Label.LABEL_FN):
                                n_useful += 1
                                
                        
                        if n_useful > 4:
                            for k in range(j,len(kvs)):
                                if (kvs[k][1].type == Label.LABEL_FN):
                                    print(k, kvs[k][0], 'blank')
                                else:
                                    id =kvs[k][1].indice
                                    print(k, kvs[k][0], tokens2text(tokenizer, [tokens[id]]))
                            assert False, f'n_useful = {n_useful}'

                        for j in range(consecutive_fn_begin, len(kvs)):
                            if kvs[j][1].type != Label.LABEL_FN:
                                break
                            best_solution.pop(kvs[j][0]) # remove by key
                        break # do not go head
                else:
                    consecutive_fn = 0
        

    return best_solution, best_start


def handle_illegal_encodings(bs):
    while True:
        try:
            dec = bs.decode()
            return dec
        except UnicodeDecodeError as e:
            bs = bs[:e.start] + b'?' + bs[e.start + 1:]

def escape_illegal_encodings(bs, escape_prefix : str):
    while True:
        try:
            dec = bs.decode()
            return dec
        except UnicodeDecodeError as e:
            bs = bs[:e.start] + escape_prefix.encode() + hex(bs[e.start]).encode() + bs[e.start + 1:]

def weighted_random_choice(k):
    # 创建权重，使用指数衰减（1, 1/2, 1/3, ..., 1/k）
    weights = [1/i for i in range(1, k + 1)]
    
    # 归一化权重，使得总和为1
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # 进行加权采样
    return np.random.choice(range(1, k + 1), p=normalized_weights)

def stat_tbe(ds, tokenizer):
    noise_tbe = []
    tp_tbe = []
    blank_tbe = []
    n_fp = 0
    n_fn = 0
    n_total = 0
    for r, record in enumerate(tqdm(ds['results'])):
        # using real cache trace for test
        timepoints, tokens, _ = get_tp_tk(record)
        (tp_prompt, tk_prompt), (tp_resp, tk_resp) = split_prompt_response(timepoints, tokens)

        tbe = tp_resp[1:] - tp_resp[:-1]
        t0 = 1/estimate_f0(tp_resp, tk_resp, tokenizer)
        # norm the TBE
        tbe /= t0

        for t in tbe:
            if t > 1.8 and t < 2.2:
                blank_tbe.append(t)
            elif t > 0.85 and t < 1.15:
                tp_tbe.append(t)
            else:
                noise_tbe.append(t)
        
        lcs = longest_common_subsequence(tk_resp, record['g']['o'])
        # get false negative groundtruth
        gt_missing = []
        gt_mising_timeinterval = []
        last_tk_resp_pos = -1
        last_gt_pos = -1
        for o in lcs:
            #         gt_con gt_ncon
            # tk_con   tp       fn
            # tk_ncon  fp      fn+fp

            # tk has skipped some tokens
            if last_tk_resp_pos + 1 != o[0].ind:
                # fp
                for j in range(last_tk_resp_pos + 1, o[0].ind):
                    n_fp += 1
                    
            # gt has skipped some tokens
            if last_gt_pos + 1 != o[1].ind:
                # missing
                gt_missing.append((last_tk_resp_pos, o[0].ind))
                gt_mising_timeinterval.append((tp_resp[last_tk_resp_pos] if last_tk_resp_pos != -1 else 0,
                                            tp_resp[o[0].ind]))
            last_tk_resp_pos = o[0].ind
            last_gt_pos = o[1].ind
        
        n_fn += len(gt_missing)
        n_total += len(tk_resp)

    print('FP prob = ', n_fp / n_total)
    print('FN prob = ', n_fn / n_total)

    def fit(data):
        mu, std = stats.norm.fit(data)
        print('std=', std)
        print('mean=', mu)
        x = np.linspace(min(data), max(data), 100)
        pdf_fitted = stats.norm.pdf(x, mu, std)

        plt.figure()
        plt.hist(data, bins=1000, density=True, alpha=0.6, color='g', label="Data")
        plt.plot(x, pdf_fitted, 'r-', lw=2, label=f'Fit: mu={mu:.2f}, std={std:.2f}')
        plt.legend()
        plt.xlabel('Data values')
        plt.ylabel('Density')
        plt.title('Histogram and Fitted Normal Distribution')
        
    print('tp:')
    fit(tp_tbe)
    print('noise:')
    fit(noise_tbe)
    print('fn:')
    fit(blank_tbe)

    plt.show()

def ngram_overlap_ratio(tokens, n, ngram_set):
    # Generate n-grams from the sentence
    sentence_ngrams = set(ngrams(tokens, n))
    
    # Calculate the intersection with the provided ngram set
    overlap = sentence_ngrams.intersection(ngram_set)
    overlap_count = len(overlap)
    total_count = len(sentence_ngrams)
    
    # Calculate the ratio of overlapping n-grams
    return overlap_count / total_count if total_count > 0 else 0

def sanitize_train_dataset(#input
                            test_file, val_file, train_file, tokenizer,
                            prompt_as_groundtruth : bool,
                            #output
                           train_sanitized_file,
                            gram_n=8):
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end_of_word = False

    class Trie:
        def __init__(self):
            self.root = TrieNode()

        def insert(self, word):
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end_of_word = True

        def starts_with_n(self, prefix, n):
            node = self.root
            for n in range(min(n,len(prefix))):
                char = prefix[n]
                if char not in node.children:
                    return False
                node = node.children[char]
            return True
    
    MAX_SHARED_TOKENS = 8
    
    tree = Trie()

    test_records = []
    if test_file is not None:
        test_records.extend(jsonl_read(test_file))
    if val_file is not None:
        test_records.extend(jsonl_read(val_file))
            
    def get_groundtruth(record):
        return record['g']['i'] if prompt_as_groundtruth else record['g']['o']

    test_ngrams = []
    for index, record in enumerate(tqdm(test_records)):
        bts = get_groundtruth(record) 
        tree.insert(bts)
        test_ngrams.extend(ngrams(bts, gram_n))
    test_ngrams = set(test_ngrams)

    train_records = jsonl_read(train_file)

    sanitized = []
    avg_overlap = DynamicAvg()
    n_abnormal = 0
    sant_stat = []
    for index, record in enumerate(tqdm(train_records)):
        this_tks = get_groundtruth(record)
        overlap = ngram_overlap_ratio(this_tks, gram_n, test_ngrams)
        
        share_prefix = tree.starts_with_n(this_tks, MAX_SHARED_TOKENS)
        
        drop = (overlap > 0.7 or share_prefix)
        if (not drop):
            sanitized.append(record)
        else:
            n_abnormal += 1
        avg_overlap.update(overlap)
    
        sant_stat.append({
            f'overlap{gram_n}': overlap,
            'sharedp': share_prefix,
            'drop': drop
        })

    print(f'avg {gram_n}-gram overlap ratio = {avg_overlap.get()}')
    print(f'Abnormal {n_abnormal}')

    json.dump(sant_stat, open(train_sanitized_file + '.stat.json', 'w'))
    jsonl_write(train_sanitized_file, sanitized)

# ================
# OpenAI stuffs
# ================

def generate_openai_dataset(ds, openai_pathname):
    with open(openai_pathname, 'w') as fp:
        fp.write(''.join([json.dumps(e['t']) + '\n' for e in ds]))

def openai_get_groundtruth_max_tokens(jsonl_ds, model):
    max_token = 0
    encoder = tiktoken.encoding_for_model(model)
    for line in open(jsonl_ds, 'r').readlines():
        entry = json.loads(line)['t']
        gt = entry['messages'][2]['content']
        encoding = encoder.encode(gt)
        max_token = max(max_token, len(encoding))
    return max_token

def openai_finetune_check(ft_ds_pathname, founda_model):
    with open(ft_ds_pathname, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line)['t'] for line in f.readlines()]

    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)
    
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
        assert False
    else:
        print("No errors found")
    
    encoding = tiktoken.encoding_for_model(founda_model)

    # not exact!
    # simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(values, name):
        print(f"\n#### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values)}, {np.median(values)}")
        print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")
    
    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
        
    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    assert n_missing_system == 0 and n_missing_user == 0
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    n_too_long = sum(l > 16385 for l in convo_lens)
    print(f"\n{n_too_long} examples may be over the 16,385 token limit, they will be truncated during fine-tuning")
    assert n_too_long == 0

    # Pricing and default n_epochs estimate
    if founda_model[:6] == 'gpt-4o':
        MAX_TOKENS_PER_EXAMPLE = 128000
    elif founda_model[:7] == 'gpt-3.5':
        MAX_TOKENS_PER_EXAMPLE = 16385

    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")

def myds_to_openai_bytes_io(ds_file):
    ds = jsonl_read(ds_file)
    return BytesIO(''.join([json.dumps(d['t'])+'\n' for d in ds]).encode())

def check_empty_dataset(ds_file):
    ds = jsonl_read(ds_file)
    for record in ds:
        assert len(record['t']['messages'][0]['content']) and len(record['t']['messages'][1]['content']), 'check the --will-test option'

def finetune_openai(#input
                    train_sanitized_file,
                    val_file,
                    # params
                    founda_model):

    #**************
    # fine tuning
    #**************

    openai_finetune_check(train_sanitized_file, founda_model)

    nt = openai_get_groundtruth_max_tokens(train_sanitized_file, founda_model)
    assert nt < max_tokens - 50, f'sorry, increase the max_tokens variable to > {nt}'
    if val_file:
        nt = openai_get_groundtruth_max_tokens(val_file, founda_model)
        assert nt < max_tokens - 50, f'sorry, increase the max_tokens variable to > {nt}'
    
    # both the training set and val set are required
    check_empty_dataset(train_sanitized_file)
    if val_file:
        check_empty_dataset(val_file)

    print(f'Uploading {train_sanitized_file}...')
    ret = client.files.create(
        file=myds_to_openai_bytes_io(train_sanitized_file),
        purpose="fine-tune",
        timeout=TIMEOUT
    )
    updated_train_sanitized_file_id = ret.id
    print(f'ID = {updated_train_sanitized_file_id}')

    if val_file:
        print(f'Uploading {val_file}...')
        ret = client.files.create(
            file=myds_to_openai_bytes_io(val_file),
            purpose="fine-tune",
            timeout=TIMEOUT
        )
        updated_val_file_id = ret.id
        print(f'ID = {updated_val_file_id}')
    else:
        updated_val_file_id = None

    ret = client.fine_tuning.jobs.create(
        training_file=updated_train_sanitized_file_id,
        validation_file=updated_val_file_id,
        model=founda_model,
        timeout=TIMEOUT
    )
    print(f'JOB = {ret}')
    return ret

def save_ft_metrics_openai(ft_loss_file,
                    ft_hyperparam_file,
                    ftjob):
    print(client.fine_tuning.jobs.list(limit=10, timeout=TIMEOUT))

    ft = client.fine_tuning.jobs.retrieve(ftjob,
                                            timeout=TIMEOUT)
    print(ft)
    if len(ft.result_files):
        with open(ft_loss_file, 'wb') as fp:
            fp.write(base64.b64decode(client.files.retrieve_content(ft.result_files[0])))
        
    with open(ft_hyperparam_file, 'w') as fp:
        fp.write(str(ft.hyperparameters))

def start_evaluate_openai(records,
                    ft_model_id,
                    founda_model
                    ):

    #**************
    # evaulating
    #**************
    if False: # pre-test the model (one query only)
        for record in records:
            completion = client.chat.completions.create(
                model=ft_model_id,
                messages=record['messages'][:2],
                timeout=TIMEOUT
            )
            print(record)
            print('===(ground truth):===')
            print(record['messages'][2]['content'])
            print('===(Model output):===')
            print(completion.choices[0].message.content)
            break
    
    tasks = []
    for index, record in enumerate(records):
        tasks.append(json.dumps({
            "custom_id": f"task-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": ft_model_id,
                "max_tokens": max_tokens,
                "messages": record['messages'][:2], # remove groundtruth
            }
        }) + '\n')

    #return None, ''.join(tasks)

    print(f'Uploading test')
    batch_file = client.files.create(
        file=BytesIO(''.join(tasks).encode()),
        purpose="batch",
        timeout=TIMEOUT
    )
    batched_test_file_id = batch_file.id
    print(f'batched_test_file_id={batched_test_file_id}')


    batch_job = client.batches.create(
        input_file_id=batched_test_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        timeout=TIMEOUT
    )
    print(batch_job)
    return batch_job, ''.join(tasks)

import time

def wait_batch_job(batch_id):
    while True:
        ret = client.batches.retrieve(batch_id, timeout=TIMEOUT+30)
        status  = ret.status
        print(f"Current status: {status}")
        
        if status == "completed":
            print("Batch processing completed successfully!")
            return ret
        elif status == "failed":
            print("Batch processing failed.",ret )
            assert False
        else:
            # have a rest
            time.sleep(30)

def wait_ft_job(ft_job_id):
    while True:
        ret = client.fine_tuning.jobs.retrieve(ft_job_id, timeout=TIMEOUT+30)
        status  = ret.status
        print(f"Current status: {status}")
        
        if status == "succeeded":
            print("Fine tuning completed successfully!")
            return ret
        elif status == "failed":
            print("Fine tuning failed.")
            assert False
        else:
            # have a rest
            time.sleep(30)

import sys
import tty
import termios

def wait_for_keypress():
    print("press any key...")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            if sys.stdin.read(1):
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


import string
import unicodedata

punctuation_chars = None
translator = None
def remove_punctuation(src):
    global punctuation_chars
    global translator
    if punctuation_chars is None: # lazzy init
        punctuation_chars = [
            chr(i) for i in range(0x110000)
            if unicodedata.category(chr(i)).startswith("P")
        ]
        punctuation_chars.append('\n')
        translator = str.maketrans('', '', ''.join(punctuation_chars))
    return src.translate(translator).strip()

def write_model_id(ft_model_id, fn):
    json.dump({"model_id": ft_model_id}, open(fn, 'w'))

def read_model_id(fn):
    ret = json.load(open(fn, 'r'))
    return ret['model_id']

def get_evaluate_openai(#input
                        test_file,
                        test_offset,
                        #output
                        test_results_file,
                        ft_model_id,
                        result_file_id):
    test_org = jsonl_read(test_file)

    responses = client.files.content(result_file_id,
                                    timeout=TIMEOUT).content
    lines = responses.splitlines()
    
    results = []
    for index, line in enumerate(tqdm(lines)):
        res_record = json.loads(line)
        assert res_record['response']['body']['model'] == ft_model_id
        assert res_record['custom_id'] == f'task-{index}'

        entry = copy.deepcopy(test_org[test_offset + index])
        entry['pred'] = res_record['response']['body']['choices'][0]['message']['content']
        results.append(entry)

    jsonl_write(test_results_file, results)

def safe_rouge(gt, pred):
    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l", "rouge-4"])
    if len(gt):
        return rouge.get_scores(gt, pred)
    else:
        f = 1 if not len(pred) else 0
        return [{'rouge-1': {'f': f},
                'rouge-2': {'f': f},
                'rouge-4': {'f': f},
                'rouge-l': {'f': f}}]


def evaluate_no_recon(test_ds, tokenizer, test_restored_response_file, pred_func):
    test_restored_response = []
    avg_f1 = DynamicAvg()
    avg_ld_score = DynamicAvg()
    avg_bleu = DynamicAvg()
    avg_rouge1 = DynamicAvg()
    avg_rouge2 = DynamicAvg()
    avg_rougeL = DynamicAvg()
    n_prompts = 0
    n_exact = 0
    verbose = False

    for r, record in enumerate(tqdm(test_ds['results'])):
        groundtruth, pred = pred_func(record)

        ld_score = 1 - Levenshtein.distance(groundtruth, pred) / max(len(groundtruth), len(pred))
        
        pr, rc, f1 = get_precision_recall_f1(groundtruth, pred)
        avg_f1.update(f1)
        avg_ld_score.update(ld_score)

        if verbose:
            print(f'LD score = {ld_score}')
            print(f'Pr {pr*100:.1f}% Rc {rc*100:.1f}% F1 {f1}')
        
        chencherry = SmoothingFunction()
        bleu_val = sentence_bleu([text2tokens(tokenizer, groundtruth)],
                                text2tokens(tokenizer, pred),
                                smoothing_function=chencherry.method1)
        avg_bleu.update(bleu_val)
        rouge_val = safe_rouge(groundtruth, pred)
        r1 = rouge_val[0]['rouge-1']['f']
        r2 = rouge_val[0]['rouge-2']['f']
        rL = rouge_val[0]['rouge-l']['f']
        avg_rouge1.update(r1)
        avg_rouge2.update(r2)
        avg_rougeL.update(rL)

        exact = remove_punctuation(groundtruth) == remove_punctuation(pred)

        test_restored_response.append({
            # dataset location tuple
            'd': record['d'],
            'bleu': bleu_val,
            'r1': r1, 'r2': r2,'rL': rL,
            'LD': ld_score,
            'gt': groundtruth,
            'extact': exact,
            'pred': pred
        })
        if exact:
            n_exact += 1
        #else:
        #    print('=== gt:', groundtruth)
        #    print('=== pred:', pred)
        #    wait_for_keypress()
        n_prompts += 1

    print(f'Avg LD score = {avg_ld_score.get()}')
    print(f'Avg F1 = {avg_f1.get()}')
    print(f'Avg Rouge-1 = {avg_rouge1.get()}')
    print(f'Avg Rouge-2 = {avg_rouge2.get()}')
    print(f'Avg Rouge-L = {avg_rougeL.get()}')
    print(f'Avg BLEU = {avg_bleu.get()}')
    print(f'Exact ASR = {n_exact/n_prompts*100}%')

    if test_restored_response_file is not None:
        json.dump(test_restored_response, open(test_restored_response_file, 'w'))

    return avg_ld_score.get(), \
        avg_f1.get(), \
        avg_rouge1.get(), \
        avg_rouge2.get(), \
        avg_rougeL.get(), \
        avg_bleu.get(), \
        n_exact/n_prompts*100

def dynamic_batching(texts, tokenizer, max_total_tokens):
        batches = []
        current_batch = []
        current_total_tokens = 0

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            seq_length = input_ids.shape[1]

            if len(current_batch) and current_total_tokens + seq_length > max_total_tokens: # avoid empty batch
                batches.append(current_batch)
                current_batch = []
                current_total_tokens = 0

            current_batch.append(text)
            current_total_tokens += seq_length

        if len(current_batch):
            batches.append(current_batch)

        return batches

def openai_dynamic_batching(test_file, founda_model):
    batches = []
    current_batch = []
    sum_tks = 0
    encoder = tiktoken.encoding_for_model(founda_model)
    ds = jsonl_read(test_file)
    for i, d in enumerate(ds):
        record = d['t']
        tks = len(encoder.encode(record['messages'][0]['content'])) + len(encoder.encode(record['messages'][1]['content']))
        if sum_tks + tks < OPENAI_MAX_ENQUEUE_TOKENS:
            current_batch.append(record)
            sum_tks += tks
        else:
            batches.append(current_batch)
            current_batch = [record]
            sum_tks = tks

    if len(current_batch):
        batches.append(current_batch)

    return batches

#########################
# Loading dataset and tokenizer
#########################

from constants import *


#########################
# Connect to OpenAI
#########################
client = None
TIMEOUT = None
def init_openai(mock_gpt : bool):
    global client
    global TIMEOUT
    import openai_apikey

    if mock_gpt:
        # before this, please start the mockai server
        client = OpenAI(base_url="http://localhost:12306/v1", api_key="sk-123456789")
        print('Warning: Using mock gpt server!')

    else:
        import httpx
        # Configure your proxy
        proxy = "127.0.0.1:7890"
        proxies = {'https://': f'http://{proxy}', 'http://': f'http://{proxy}'}

        client = OpenAI(api_key=openai_apikey.OPENAI_API_KEY,
                        http_client = httpx.Client(
                            proxies=proxies,
                            transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                        ))
    
    TIMEOUT = 25
    print('inited OpenAI client')

########################
# Set fonts for figures
########################
plt.rcParams.update({
    'font.size': 8, 
    #'font.family': 'Nimbus Roman No9 L',#'serif',
    'axes.titlesize': 8,  # 设置标题字体大小
    'axes.labelsize': 8,  # 设置坐标轴标签字体大小
    'xtick.labelsize': 8,  # 设置 x 轴刻度字体大小
    'ytick.labelsize': 8,  # 设置 y 轴刻度字体大小
    'legend.fontsize': 8,  # 设置图例字体大小
})


MY_COLOR_MAPS1 = [
    (0,0,0), # k
    (65,14,115), # p
    (140,42,129),
    (223,74,104),
    (252,154,107),
    (252,248,187) # yellow
]
MY_COLOR_MAPS1 = [(t[0]/255, t[1]/255, t[2]/255) for t in MY_COLOR_MAPS1]

# lighter
MY_COLOR_MAPS2 = ['#d73027',
'#fc8d59',
'#fee090',
'#91bfdb',
'#e0f3f8',
'#4575b4',
]

#deeper
MY_COLOR_MAPS3 = ['#d73027',
'#fc8d59',
'#fee090',
'#91bfdb',
'#e0f3f8',
'#4575b4',
]


MY_COLOR_MAPS4 = [
    '#4D4D4D', # k
    '#F5BF4C', # yellow
    '#4F92CD', # blue
    '#28BA91', # cyan
    '#F05F96', # magenta
]

MY_COLOR_MAPS5 = {
    '#F09135', # orange
    '#67C94D', # green
    '#4590F7', # blue
    '#B72F82', # magenta
    'F5BF4C', # yellow
}