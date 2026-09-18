import subprocess
from constants import *
from hyperparam import *
import os
import pty
import sys
import multiprocessing
import math

NR_GPUS=1
PYTHON='python3'

def log(logfile, msg, echo=True):
    if isinstance(msg, str):
        msg = msg.encode()
    if echo:
        try:
            print(msg.decode(), end='')
        except:
            print(msg, end='')
    logfile.write(msg)

def run_command(logfile, command):
    def read(fd):
        data = os.read(fd, 1024)
        log(logfile, data, echo=False)
        return data
    
    print('run: ', command)
    exit_code = pty.spawn(command, read)
    return exit_code, None

def update_rc(logfile, rc):
    if rc:
        log(logfile, f'Sub process exited with return code {rc}.\n')
        exit(rc)

def get_fin_test(val : bool): 
    return 'eval_val' if val else 'test'

def run_per_attack(progfile, device, operationsys, framework, hardware, attack,
                    train_ds_collection_id, eval_ds_collection,
                    founda_model, use_valset : int, llms, params, prefill_bs, embd_quant='F16'):
    assert isinstance(use_valset, int)

    logfile_fn = f'runall.{founda_model}.{train_ds_collection_id}.{eval_ds_collection}'
    logfile = open(f'{logfile_fn}.log', 'ab', buffering=1)

    # train and evaluate uses the same script (exp_attack)
    params += [f"--attack={attack}",
               f"--train-ds-collection-id={train_ds_collection_id}"
                ]
    model_name = get_new_model_name_args(params)
    cmdline = ['exp_attack.py',
                    ] + params

    DEEPSPEED = ['deepspeed', f'--num_gpus={NR_GPUS}']

    # train the model
    train_probed_res_prefix = get_train_probed_res_prefix()
    model_id_file = f"{train_probed_res_prefix}/{GEN_BASE}/{model_name}/model_id_{train_ds_collection_id}_None.json" # not trained with different bs
    progfile.write(f'train {model_name} {model_id_file}\n')
    if not os.path.exists(model_id_file):
        log(logfile, f'Train {model_name} out: {model_id_file}\n')
        update_rc(logfile, run_command(logfile, DEEPSPEED + cmdline + [
                f'--eval-steps=2000',
                '--mode=train',
                f'--validate={use_valset}',
                f'--will-test={int(GEN_BASE_LLM_ID in llms)}'])[0])
    else:
        log(logfile, f'skipping training {model_name}\n')

    # evaluate
    probed_res_prefix = get_probed_res_prefix(device, operationsys, framework, hardware)
    eval_cmdlines = []
    messages = []
    eval_cmdlines2 = []
    messages2  = []
    for llm in llms:
        llm_vendor, llmname = VICTIM_LLM[llm]
        eval_results_file = f"{probed_res_prefix}/{llmname}/{model_name}/{get_fin_test(use_valset)}_results_{eval_ds_collection}{get_suffix(prefill_bs, embd_quant)}.jsonl"

        if not os.path.exists(eval_results_file):
            log(logfile, f'Eval {llmname} {model_name}\n')
            messages.append(f'eval {eval_results_file}\n')
            eval_cmdline = DEEPSPEED + cmdline + [
                f"--eval-ds-collection-id={eval_ds_collection}",
                '--mode=eval',
                f'--validate={use_valset}',
                f'--victim-llm={llm}']
            if prefill_bs:
                eval_cmdline += [f'--prefill-bs={prefill_bs}']
            if embd_quant != 'F16':
                eval_cmdline += [f'--embd-quant={embd_quant}']
            eval_cmdlines.append(eval_cmdline)
        else:
            progfile.write(f'skipped eval {llmname} {model_name}\n')
            log(logfile, f'skipping eval val={use_valset} llm={llmname} {model_name}\n')

        # analyze the results
        analyzed_results_file = f"{probed_res_prefix}/{llmname}/{model_name}/analyzed_{get_fin_test(use_valset)}_results_{eval_ds_collection}{get_suffix(prefill_bs, embd_quant)}.jsonl"
        if not os.path.exists(analyzed_results_file): 
            log(logfile, f'Analyze {llmname} {model_name}\n')
            messages2.append(f'analyze {llmname}/{model_name} {get_fin_test(use_valset)}\n')
            analyze_cmdline = DEEPSPEED + ['exp_analyze_results.py',
                f"--device={device}",
                f'--operationsys={operationsys}',
                f'--framework={framework}',
                f'--hardware={hardware}',
                f'--eval-ds-collection-id={eval_ds_collection}',
                f'--new-model-name={model_name}',
                f'--validate={use_valset}',
                f'--victim-llm={llm}']
            if prefill_bs:
                analyze_cmdline += [f'--prefill-bs={prefill_bs}']
            if embd_quant != 'F16':
                analyze_cmdline += [f'--embd-quant={embd_quant}']
            eval_cmdlines2.append(analyze_cmdline)
        else:
            progfile.write(f'skipped analyze {llmname}/{model_name} {get_fin_test(use_valset)}\n')
            log(logfile, f'skipping eval val={use_valset} llm={llmname} {model_name}\n')

    # execute the commands
    if is_osllm(founda_model):
        while len(eval_cmdlines) or len(eval_cmdlines2):
            if len(eval_cmdlines):
                progfile.write(messages.pop(0))
                update_rc(logfile, run_command(logfile, eval_cmdlines.pop(0))[0]) 
            if len(eval_cmdlines2):
                progfile.write(messages2.pop(0))
                update_rc(logfile, run_command(logfile, eval_cmdlines2.pop(0))[0]) 
       
    else:
        # OpenAI can be processed concurrently (actually it cannot for very long tokens)
        def run_async(cmdlines, messages):
            # TODO: out-of-order issue
            batch_size = 3
            n_batches = math.ceil(len(cmdlines)/batch_size)
            for b in range(n_batches):
                st = b*batch_size
                end = min(len(cmdlines), st + batch_size)
                for i, cmdline in enumerate(cmdlines[st:end]):
                    processes = []
                    process_cmdlines = []
                    logfile_per_proc = open(f'{logfile_fn}.{i}.log', 'ab', buffering=1)
                    def start_task(cmdline, logfile):
                        sys.exit(run_command(logfile, cmdline)[0])
                    process = multiprocessing.Process(target=start_task, args=(cmdline, logfile_per_proc))
                    process_cmdlines.append(cmdline)
                    process.start()
                    processes.append(process)
                    progfile.write(messages[i])

                    exitcode= []
                    for process in processes:
                        process.join()
                        exitcode.append(process.exitcode)
                    
                    for i, r in enumerate(exitcode):
                        if r:
                            log(logfile, 'Error in ', process_cmdlines[i])
                        update_rc(logfile, r)

        run_async(eval_cmdlines, messages)

        # after finished the eval, analyze the results serially
        for i, cmdline in enumerate(eval_cmdlines2):
            progfile.write(messages2[i])
            update_rc(logfile, run_command(logfile, cmdline)[0])

def run_rr_pr(progfile, device, operationsys, framework, hardware,
              train_ds_collection_id, eval_ds_collection_id,
              founda_model,
              use_valset : int,
              llms,
              rr_prob, pr_prob, sigma,
              has_finetune : int,
              rr_has_llm : int,
              pr_has_llm : int,
              abs_time : int,
              has_timing : int,
              has_resp : int,
              prefill_bs,
              has_sca = 1,
              embd_quant = 'F16'):

    for v in [use_valset, has_finetune, rr_has_llm, pr_has_llm, abs_time, has_timing, has_resp]:
        assert isinstance(v, int) 

    params_common=[f"--device={device}",
                   f'--operationsys={operationsys}',
                   f'--framework={framework}',
                   f'--hardware={hardware}',
                    f"--founda-model={founda_model}",
                    f'--scale=10',
                    f'--batch-size=2',
                    f'--rank=256',
                    f'--alpha=256',
                    f'--has-finetune={has_finetune}']

    rr_params=params_common + [
                f'--has-llm={rr_has_llm}',
                f'--prob={rr_prob}',
                f'--sigma={sigma}',
                f'--lr=8e-5',
                f'--epoch=3',
                f'--abs-time={abs_time}',
                f'--has-timing={has_timing}']

    rr_model_name=get_new_model_name_args(['--attack=rr',
                                            f'--train-ds-collection-id={train_ds_collection_id}']
                                            + rr_params)

    progfile.write(f'====\n RR val = {use_valset}\n====\n')
    run_per_attack(progfile, device, operationsys, framework, hardware, 'rr',
                    train_ds_collection_id, eval_ds_collection_id,
                    founda_model, use_valset, llms, rr_params, prefill_bs, embd_quant)
    
    progfile.write(f'====\n PR\n====\n')
    pr_params = params_common + [
                f'--has-llm={pr_has_llm}',
                f'--prob={pr_prob}',
                f'--lr=0.0002',
                f'--epoch=2',
                f'--has-resp={has_resp}',
                f'--rr-model={rr_model_name}']
    if not has_sca:
        pr_params.append(f'--has-sca=0')
    run_per_attack(progfile, device, operationsys, framework, hardware, 'pr',
                    train_ds_collection_id, eval_ds_collection_id,
                    founda_model, use_valset, llms, pr_params, prefill_bs, embd_quant)

def eval_main(progfile, founda_model, train_ds_collection_id, eval_ds_collection_id, device, operationsys, framework, hardware, llms, embd_quant='F16'):
    rr_prob=0.2
    pr_prob=0.2
    sigma=0.08
    has_finetune=1
    rr_has_llm=1
    pr_has_llm=1
    abs_time=0
    has_timing=1
    has_resp=1
    has_sca=1

    use_valset = 0 # for testing

    progfile.write(f'****\n {sys._getframe().f_code.co_name} {founda_model} \n****\n')
    run_rr_pr(progfile, device, operationsys, framework, hardware,
                train_ds_collection_id, eval_ds_collection_id,
                founda_model, use_valset, llms,
                rr_prob, pr_prob, sigma, has_finetune, rr_has_llm, pr_has_llm, abs_time, has_timing, has_resp, None, has_sca, embd_quant)

def prefill_bs_analysis(progfile, founda_model, ds_collection_id):
    train_ds_collection_id = ds_collection_id
    eval_ds_collection_id = ds_collection_id

    rr_prob=0.2
    pr_prob=0.2
    sigma=0.08
    has_finetune=1
    rr_has_llm=1
    pr_has_llm=1
    abs_time=0
    has_timing=1
    has_resp=1

    use_valset = 0 # for testing

    device, operationsys, framework, hardware = ('gpu', 'ubuntu22_04', 'llama.cpp', 'Intel 13900K')

    for pbs in [128, 64, 32]:
        progfile.write(f'****\n {sys._getframe().f_code.co_name} prefill_bs={pbs} \n****\n')
        run_rr_pr(progfile, device, operationsys, framework, hardware,
                    train_ds_collection_id, eval_ds_collection_id,
                    founda_model, use_valset, REDUCED_LLMS,
                    rr_prob, pr_prob, sigma, has_finetune, rr_has_llm, pr_has_llm, abs_time, has_timing, has_resp, pbs)

def ablation_study(progfile, founda_model, ds_collection_id):
    train_ds_collection_id = ds_collection_id
    eval_ds_collection_id = ds_collection_id

    prob=0.2
    sigma=0.08
    has_finetune=1
    rr_has_llm=1
    pr_has_llm=1
    abs_time=0
    has_resp=1

    use_valset=0 # using test set for eval

    device, operationsys, framework, hardware = ('gpu', 'ubuntu22_04', 'llama.cpp', 'Intel 13900K')
    
    #
    # Ablation study for has-timing
    #
    progfile.write(f'****\n {sys._getframe().f_code.co_name} : has-timing \n****\n')
    for has_timing in [1,0]:
        run_rr_pr(progfile, device, operationsys, framework, hardware,
            train_ds_collection_id, eval_ds_collection_id,
            founda_model, use_valset, REDUCED_LLMS,
            prob,prob, sigma, has_finetune, rr_has_llm, pr_has_llm, abs_time, has_timing, has_resp, None)
    has_timing = 1

    #
    # Ablation study for has-llm
    #
    progfile.write(f'****\n {sys._getframe().f_code.co_name} : has-llm\n****\n')
    for rr_has_llm in [1,0]:
        for pr_has_llm in [1,0]:
            run_rr_pr(progfile, device, operationsys, framework, hardware,
                train_ds_collection_id, eval_ds_collection_id,
                founda_model, use_valset, REDUCED_LLMS,
                prob,prob, sigma, has_finetune, rr_has_llm, pr_has_llm, abs_time, has_timing, has_resp, None)
    rr_has_llm = 1
    pr_has_llm = 1

    #
    # Ablation study for has_resp
    #
    progfile.write(f'****\n {sys._getframe().f_code.co_name} : has-resp\n****\n')
    for has_resp in [1,0]:
        run_rr_pr(progfile, device, operationsys, framework, hardware,
             train_ds_collection_id, eval_ds_collection_id,
            founda_model, use_valset, REDUCED_LLMS,
            prob,prob, sigma, has_finetune, rr_has_llm, pr_has_llm, abs_time, has_timing, has_resp, None)
    has_resp = 1

    #
    # Ablation study for abs-time
    #
    progfile.write(f'****\n {sys._getframe().f_code.co_name} : abs-time\n****\n')
    for abs_time in [1,0]:
        run_rr_pr(progfile, device, operationsys, framework, hardware,
            train_ds_collection_id, eval_ds_collection_id,
            founda_model, use_valset, REDUCED_LLMS,
            prob,prob, sigma, has_finetune, rr_has_llm, pr_has_llm, abs_time, has_timing, has_resp, None)
    abs_time = 0
    
    #
    # Ablation study for has-finetune
    #
    progfile.write(f'****\n {sys._getframe().f_code.co_name} : has-finetune\n****\n')
    for has_finetune in [1,0]:
        run_rr_pr(progfile, device, operationsys, framework, hardware,
            train_ds_collection_id, eval_ds_collection_id,
            founda_model, use_valset, REDUCED_LLMS,
            prob,prob, sigma, has_finetune, rr_has_llm, pr_has_llm, abs_time, has_timing, has_resp, None)
    has_finetune = 1

    #
    # Ablation study for has-sca
    #
    progfile.write(f'****\n {sys._getframe().f_code.co_name} : has-sca\n****\n')
    for has_sca in [1,0]:
        run_rr_pr(progfile, device, operationsys, framework, hardware,
            train_ds_collection_id, eval_ds_collection_id,
            founda_model, use_valset, REDUCED_LLMS,
            prob,prob, sigma, has_finetune, rr_has_llm, pr_has_llm, abs_time, has_timing, has_resp, None, has_sca)
    has_sca = 1


def sen_analysis(progfile, founda_model, ds_collection_id):
    train_ds_collection_id = ds_collection_id
    eval_ds_collection_id = ds_collection_id

    # hyperparams
    has_finetune=1
    rr_has_llm=1
    pr_has_llm=1
    abs_time=0
    has_timing=1
    has_resp=1

    use_valset=1 # using validation set for eval

    sigma_vals=[0.08, 0.02, 0.04, 0.16, 0.32, 0.64, 1.28]
    prob_vals=[0.2, 0.1, 0.4, 0.6, 0.8]

    device, operationsys, framework, hardware = ('gpu', 'ubuntu22_04', 'llama.cpp', 'Intel 13900K')

    progfile.write(f'****\n {sys._getframe().f_code.co_name} : prob\n****\n')
    sigma = 0.08
    for prob in prob_vals:
        run_rr_pr(progfile, device, operationsys, framework, hardware,
            train_ds_collection_id, eval_ds_collection_id,
            founda_model, use_valset, REDUCED_LLMS,
            prob,prob, sigma, has_finetune, rr_has_llm, pr_has_llm, abs_time, has_timing, has_resp, None)
    
    progfile.write(f'****\n {sys._getframe().f_code.co_name} : sigma\n****\n')
    prob = 0.2
    for sigma in sigma_vals:
        run_rr_pr(progfile, device, operationsys, framework, hardware,
            train_ds_collection_id, eval_ds_collection_id,
            founda_model, use_valset, REDUCED_LLMS,
            prob,prob, sigma, has_finetune, rr_has_llm, pr_has_llm, abs_time, has_timing, has_resp, None)

def framework_analysis(progfile, founda_model):
    train_ds_collection_id = 'natural-language_50000'
    eval_ds_collection_id = 'natural-language_50000_micro'
    if 1:
        hardware = 'Intel 13900K'
        operationsys = 'ubuntu22_04'
        frameworks = ['bitnet', 'gpt4all', 'llama.cpp',
                    'lmstudio', 'ollama', 'transformers',
                    'local-ai', 'koboldcpp', 'ipex-llm', 'powerinfer']
        victim_llms = ['Falcon3-7B-Instruct-1.58bit', 'Phi-3.5-mini-instruct', 'Mistral-7b-instruct',
                        'Mistral-7b-instruct', 'Mistral-7b-instruct', 'Falcon3-1B-Instruct',
                        'Mistral-7b-instruct', 'Mistral-7b-instruct', 'Mistral-7b-instruct', 'Mistral-7b-instruct']
        get_devices = lambda framework: ['cpu', 'gpu'] if framework != 'transformers' else ['cpu']

    elif 0:
        hardware = 'Intel 14900K'
        operationsys = 'ubuntu22_04'
        frameworks = ['llama.cpp']
        victim_llms = ['Mistral-7b-instruct']
        get_devices = lambda framework: ['gpu', 'cpu']

    elif 0:
        hardware = 'Intel 13900K'
        operationsys = 'debian12'
        frameworks = ['llama.cpp']
        victim_llms = ['Mistral-7b-instruct']
        get_devices = lambda framework: ['gpu', 'cpu']

    else:
        hardware = 'Intel 12700KF'
        operationsys = 'ubuntu22_04'
        frameworks = ['llama.cpp']
        victim_llms = ['Mistral-7b-instruct']
        get_devices = lambda framework: ['gpu', 'cpu']

    for f_i,framework in enumerate(frameworks):
        for device in get_devices(framework):
            progfile.write(f'****\n framework {framework} on {device}\n****\n')

            eval_main(progfile, founda_model,
                        train_ds_collection_id, eval_ds_collection_id,
                        device, operationsys, framework, hardware, [get_llm_index(victim_llms[f_i])])

def attack_thread_analysis(progfile, founda_model):
    train_ds_collection_id = 'natural-language_50000'
    eval_ds_collection_id = 'natural-language_50000_micro'
    hardware = 'Intel 13900K'
    operationsys = 'debian12'
    device = 'gpu'
    victim_llm = 'Mistral-7b-instruct'
    get_devices = lambda framework: ['cpu', 'gpu'] if framework != 'transformers' else ['cpu']

    for t_i,n_atk_threads in enumerate([1,4,8,12]):
            progfile.write(f'****\n attack thread {n_atk_threads} on {device}\n****\n')
            framework = f'llama.cpp.pt{n_atk_threads}'

            eval_main(progfile, founda_model,
                        train_ds_collection_id, eval_ds_collection_id,
                        device, operationsys, framework, hardware, [get_llm_index(victim_llm)])

def embd_quant_analysis(progfile, founda_model):
    train_ds_collection_id = 'natural-language_50000'
    eval_ds_collection_id = 'natural-language_50000_micro'

    hardware = 'Intel 13900K'
    operationsys = 'ubuntu22_04'
    framework = 'llama.cpp'
    victim_llms = [get_llm_index('Mistral-7b-instruct')]
    get_devices = lambda framework: ['gpu', 'cpu']
    embd_quants = ['Q8_0', 'BF16', 'F32']
    for quant in embd_quants:
        for device in get_devices(framework):
            progfile.write(f'****\n embd quant analysis {quant} {framework} on {device}\n****\n')

            eval_main(progfile, founda_model,
                        train_ds_collection_id, eval_ds_collection_id,
                        device, operationsys, framework, hardware, victim_llms, quant)

def operationsys_analysis(progfile, founda_model):
    train_ds_collection_id = 'natural-language_50000'
    eval_ds_collection_id = 'natural-language_50000_micro'

    hardware = 'Intel 13900K'
    framework = 'llama.cpp'
    victim_llms = [get_llm_index('Mistral-7b-instruct')]
    get_devices = lambda framework: ['gpu', 'cpu']
    embd_quant = 'F16'
    for operationsys in ['ubuntu22_04', 'windows11']:
        for device in get_devices(framework):
            progfile.write(f'****\n operation sys analysis {operationsys} {framework} on {device}\n****\n')

            eval_main(progfile, founda_model,
                        train_ds_collection_id, eval_ds_collection_id,
                        device, operationsys, framework, hardware, victim_llms, embd_quant)


def run_cgpt():
    founda_model='gpt-4o-mini-2024-07-18' #'gpt-4o-2024-08-06'
    ds_collection_id='natural-language_50000'
    progfile = open(f'progress.{founda_model}_{ds_collection_id}.log', 'w', buffering=1)

    if 1:
        eval_main(progfile, founda_model,
                    ds_collection_id, ds_collection_id,
                    device='gpu', operationsys='ubuntu22_04', framework='llama.cpp', hardware='Intel 13900K', llms=MAIN_VICTIM_LLMS)
    
    prefill_bs_analysis(progfile, founda_model, ds_collection_id)

def run_llama3_1_8b_it():
    founda_model='Llama-3.1-8B-Instruct'
    ds_collection_id='natural-language_50000'
    progfile = open(f'progress.{founda_model}_{ds_collection_id}.log', 'w', buffering=1)

    if 1:
        eval_main(progfile, founda_model,
                ds_collection_id, ds_collection_id,
                device='gpu', operationsys='ubuntu22_04', framework='llama.cpp', hardware='Intel 13900K', llms=MAIN_VICTIM_LLMS)

    prefill_bs_analysis(progfile, founda_model, ds_collection_id)

    ablation_study(progfile, founda_model, ds_collection_id)

    sen_analysis(progfile, founda_model, ds_collection_id)

    framework_analysis(progfile, founda_model)

    embd_quant_analysis(progfile, founda_model)

    operationsys_analysis(progfile, founda_model)

    attack_thread_analysis(progfile, founda_model)
    
if __name__ == '__main__':
    #run_cgpt()
    run_llama3_1_8b_it()
