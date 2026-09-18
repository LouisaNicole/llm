#!/bin/bash
set -e

python3 tab_performance.py
python3 tab_sca_data_ablation.py
python3 tab_microbench.py
python3 tab_hardware_eval.py
python3 tab_embd_quant_eval.py
python3 tab_os_eval.py

python3 exp_plot_psd_decode.py
python3 exp_plot_norm_time_diff.py
python3 exp_plot_cache_trace.py
python3 exp_plot_survey.py
python3 exp_plot_param_analysis.py
python3 exp_plot_data_sens_b.py
python3 exp_plot_baseline.py
python3 exp_plot_ablation_study.py