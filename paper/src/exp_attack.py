import rich.traceback
import rich.console
from experiment_common import *
from experiment_opensource_llm_common import *
import shutil
import numpy as np
import deepspeed
from hyperparam import Cidtfrr_Hyperparam, Cidtfpr_Hyperparam, get_new_model_name
from args import *

def simulate_noise(sigma, prob_noise, tokenizer, groundtruth_tk):
    max_token_id = len(tokenizer)-1
    margin = MAX_TOL_FN
    # training set synthesis
    timeline = dict()
    deltas = np.random.normal(loc=1, scale=sigma, size=len(groundtruth_tk))
    ct = np.random.uniform(low=0,high=margin)
    tstart = ct
    maxt = ct + sum(deltas) + margin
    for i, tk in enumerate(groundtruth_tk):
        p1 = np.random.random()
        if p1 >= prob_noise: # add false negative
            timeline[ct] = tk
        p2 = np.random.random()
        if p2 < prob_noise: # add false positive
            timeline[np.random.uniform(low=0, high=maxt)] = random.randint(0,max_token_id)
        ct += deltas[i]

    # sort by time points
    timeline = OrderedDict(sorted(timeline.items(),
                                    key=lambda x:x[0]))
    tokens = []
    timepoints = np.zeros(len(timeline))
    for i,(ct, tk) in enumerate(timeline.items()):
        tokens.append(tk)
        timepoints[i] = ct
    return tokens, timepoints, tstart

class Attack:
    def __init__(self, prompt_as_groundtruth, attackname, args, hyperparam):
        self.prompt_as_groundtruth = prompt_as_groundtruth
        self.hyperparam = hyperparam 
        self.new_model_name = get_new_model_name(attackname, args.train_ds_collection_id, hyperparam, args.rr_model)

        if args.device == 'gpu':
            self.cur_estimate_f0 = estimate_f0
        elif args.device == 'cpu':
            def cur_estimate_f0(timepoints, tokens=None, tokenizer=None):
                return 1/np.median(timepoints[1:] - timepoints[:-1])
            self.cur_estimate_f0 = cur_estimate_f0
        else:
            assert False

class RR_Attack(Attack):
    def __init__(self, args):
        super().__init__(prompt_as_groundtruth=False,
            attackname='rr',
            args=args,
            hyperparam=Cidtfrr_Hyperparam(args))

    def gen_ds(self, # input
               victim_llm, # unused
                ds,
                rr_test_results_file, # unused
                # output
                fn, tokenizer, for_test):
        #stat_tbe(ds, tokenizer)
        #exit(0)

        box = get_delimiter_token(self.hyperparam.founda_model)
        broken_unicode = b'\\x'

        ft_ds = []
        empty_records = []
        def add_sample(record, tokens, timepoints, tstart):
            if not self.hyperparam.abs_time: # use TBE
                if len(tokens):
                    tbe = timepoints[1:] - timepoints[:-1]
                    assert len(tbe) == len(tokens)-1, f'{len(tbe)} {len(tokens)}'
                    extracted_resp_text = []
                    for i, tk in enumerate(tokens):
                        if i == 0:
                            tp = timepoints[0] - tstart
                        else:
                            tp = tbe[i-1]
                        if self.hyperparam.has_timing:
                            if self.hyperparam.spaced_tbe: # for base models that have more than 1 digits in a token
                                extracted_resp_text.extend(' '.join([c for c in str(round(tp*self.hyperparam.scale))]))
                            else:
                                extracted_resp_text.extend(str(round(tp*self.hyperparam.scale)))
                            extracted_resp_text.extend(':')
                        extracted_resp_text.extend(escape_illegal_encodings(tokens2text(tokenizer, [tk]), broken_unicode.decode()))
                        if i != len(tokens)-1:
                            extracted_resp_text.extend(box.decode())
                    extracted_resp_text = ''.join(extracted_resp_text)
                else:
                    extracted_resp_text = '0:'
                
                sys_prompt = f'Task: Restore the original text from a sequence in the format "<time interval>:<word>{box.decode()}".' \
                    'Guidelines: Each word includes a time interval since its previous word.\n' + \
                    'If the interval is shorter than usual, delete nearby incorrect words.\n' + \
                    'If the interval is longer, add any likely missing words.\n' + \
                    f'Return the reconstructed text without the <time interval>: or {box.decode()} symbols,' + \
                    f'and convert hexadecimal {broken_unicode.decode()} escape sequences into their original Unicode characters.'

            else: # use abs timepoints
                assert len(timepoints) == len(tokens)
                if len(tokens):
                    extracted_resp_text = []
                    for i, tk in enumerate(tokens):
                        tp = timepoints[i] - tstart
                        if self.hyperparam.has_timing:
                            if self.hyperparam.spaced_tbe: # for base models that have more than 1 digits in a token
                                extracted_resp_text.extend(' '.join([c for c in str(round(tp*self.hyperparam.scale))]))
                            else:
                                extracted_resp_text.extend(str(round(tp*self.hyperparam.scale)))
                            extracted_resp_text.extend(':')
                        extracted_resp_text.extend(escape_illegal_encodings(tokens2text(tokenizer, [tk]), broken_unicode.decode()))
                        if i != len(tokens)-1:
                            extracted_resp_text.extend(box.decode())
                    extracted_resp_text = ''.join(extracted_resp_text)
                else:
                    extracted_resp_text = '0:'

                sys_prompt = f'Task: Restore the original text from a sequence in the format "<timepoint>:<word>{box.decode()}".' \
                    'Guidelines: Preserve words with periodic time points.' \
                    'Add missing words based on the periodic pattern.' \
                    'Remove incorrect words with noise time points.\n' \
                    f'Return the reconstructed text without the <timepoint>: or {box.decode()} symbols,' + \
                    f'and convert hexadecimal {broken_unicode.decode()} escape sequences into their original Unicode characters.'

            if not self.hyperparam.has_timing:
                sys_prompt = f'Task: Restore the original text from a sequence in the format "<word>{box.decode()}".' \
                    f'Convert hexadecimal {broken_unicode.decode()} escape sequences into their original Unicode characters.'

            # there are illegal chars, beacuse
            # 1. as we truncate the generation by token numbers, some resp may be ended unexpected
            # in that case, we just dropp the ending (with little performance loss)
            # 2. the LLM generates the illegal chars by itself (with multiple binary chars).
            dec_gt_resp_text = handle_illegal_encodings(tokens2text(tokenizer, groundtruth))

            # training data template for OpenAI finetune
            train = {
                'messages': [
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': extracted_resp_text},
                    {'role': 'assistant', 'content': dec_gt_resp_text}
                ]
            }

            ds_record = copy.deepcopy(record)
            ds_record['t'] = train
            ft_ds.append(ds_record)
        
        for r, record in enumerate(tqdm(ds['results'])):
            groundtruth = record['g']['o']

            if for_test:
                # using real cache trace for test
                timepoints, tokens, _ = get_tp_tk(record)
                try:
                    (_, _), (tp_resp, tk_resp) = split_prompt_response(timepoints, tokens)
                    # norm the time
                    tp_resp *= self.cur_estimate_f0(tp_resp, tk_resp, tokenizer)
                    add_sample(record, tk_resp, tp_resp, tp_resp[0])
                except:
                    print(f'---- empty at {r} in generating {fn} ----')
                    empty_records.append({'id':r, 'r': record})
                    add_sample(record, [], [], 0)

            else:
                tokens, timepoints, tstart = simulate_noise(self.hyperparam.sigma, self.hyperparam.prob_noise, tokenizer, groundtruth)
                add_sample(record, tokens, timepoints, tstart)

                if 0 and r==0: # for DEBUG only
                    real_timepoints, real_tokens, _ = get_tp_tk(record)
                    (_, _), (real_tp_resp, real_tk_resp) = split_prompt_response(real_timepoints, real_tokens)

                    t0 = 1/self.cur_estimate_f0(real_tp_resp, real_tk_resp, tokenizer)
                    # norm the TBE
                    real_tp_resp -= real_tp_resp[0]
                    real_tp_resp /= t0

                    fig, ax = plt.subplots(4,1, sharex=True)
                    real_tp_indices, real_fp_indices = split_false_positives_tokens_indices(real_tk_resp, groundtruth)
                    real_tk_resp = np.array(real_tk_resp)
                    plot_timeline(tokenizer, real_tp_resp[real_tp_indices], real_tk_resp[real_tp_indices], ax[0])
                    plot_timeline(tokenizer, real_tp_resp[real_fp_indices], real_tk_resp[real_fp_indices], ax[1])

                    tp_indices, fp_indices = split_false_positives_tokens_indices(tokens, groundtruth)
                    tk_resp = np.array(tokens)
                    plot_timeline(tokenizer, timepoints[tp_indices], tk_resp[tp_indices], ax[2])
                    plot_timeline(tokenizer, timepoints[fp_indices], tk_resp[fp_indices], ax[3])

                    #plt.show()
                    #exit(0)
                    plt.savefig('demo-sim-cache-trace.pdf')
                    exit(1)
                
        jsonl_write(fn, ft_ds)
        jsonl_write(fn + '.empty.jsonl', empty_records)

class PR_Attack(Attack):
    def __init__(self, args):
        super().__init__(prompt_as_groundtruth=True,
            attackname='pr',
            args=args,
            hyperparam=Cidtfpr_Hyperparam(args))
        self.template_tokens = {} # cache for templates

    def remove_template(self, tokenizer, tokens, victim_llm):
        ret = []

        # fill caches
        # the template can be found in the tokenizer config json
        if victim_llm == 'Llama-3.1-8b-instruct':
            if victim_llm not in self.template_tokens:
                self.template_tokens[victim_llm] = text2tokens(tokenizer, 'user') + text2tokens(tokenizer, 'assistant')

        elif victim_llm == 'gemma-2-9b-it':
            # user model
            if victim_llm not in self.template_tokens:
                self.template_tokens[victim_llm] = text2tokens(tokenizer, 'user') + text2tokens(tokenizer, 'model')

        elif victim_llm == 'Falcon3-10B-Instruct':
            # <|user|> <|assistant|> 
            if victim_llm not in self.template_tokens:
                self.template_tokens[victim_llm] = text2tokens(tokenizer, '<|user|>') + text2tokens(tokenizer, '<|assistant|>')

        else:
            return copy.deepcopy(tokens)

        template_tokens = copy.deepcopy(self.template_tokens[victim_llm])
        for tk in tokens:
            try:
                idx = template_tokens.index(tk)
            except ValueError:
                idx = -1
            if idx >= 0:
                del template_tokens[idx]
            else:
                ret.append(tk)
        return ret
    
    def gen_ds(self, # input
               victim_llm,
                ds,
                rr_test_results_file, # can be None if we're generating the training set
                # output
                fn, tokenizer, for_test):
        box = get_delimiter_token(self.hyperparam.founda_model)
        broken_unicode = b'\\x'
        
        ft_ds = []
        empty_records = []
        if for_test and rr_test_results_file is not None:
            resp = jsonl_read(rr_test_results_file)
        else:
            print(f'Warning: {fn} cannot be used for further evaluating, sine you set --will-test=0, we will check it again when runing evaluating !')
            resp = None

        for dsi, ds_record in enumerate(tqdm(ds['results'])):
            if not for_test or resp:
                query = bytearray()
                if self.hyperparam.has_sca:
                    #
                    # Has SCA
                    #
                    if self.hyperparam.has_resp:
                        # append responses (we use the groundtruth for traing)
                        # while in the testing, we will reaplce it with the real value predicted by RR
                        # in the subsequnet code
                        query.extend(b'<Response>\n\n')
                        if for_test:
                            # using the predicted responses for testing
                            query.extend(resp[dsi]['pred'].encode())
                        else:
                            # using the original response for training
                            query.extend(escape_illegal_encodings(tokens2text(tokenizer, ds_record['g']['o']), broken_unicode.decode()).encode())
                        query.extend(b'</Response>\n\n')
                        sys_prompt = f'Given the response from a large model, identify the original prompt from a scrambled sequence of its tokens. Some tokens are unnecessary, others missing. Unicode characters are given as the hexadecimal escape {broken_unicode.decode()}XX.'
                    else:
                        sys_prompt = f'Identify the original prompt from a scrambled sequence of its tokens. Some tokens are unnecessary, others missing. Unicode characters are given as the hexadecimal escape {broken_unicode.decode()}XX.'

                    query.extend(b'\n\n<Tokens>\n\n')

                    if for_test:
                        resp_record = resp[dsi]
                        assert ds_record['g']['i'] == resp_record['g']['i'], f"{dsi}: {ds_record['g']['i']} != {resp_record['g']['i']}"
                        assert ds_record['g']['o'] == resp_record['g']['o']

                        try:
                            # using the real cache trace for testing
                            timepoints, tokens, _ = get_tp_tk(ds_record)
                            (_, query_tks), (_, _) = split_prompt_response(timepoints, tokens)
                            query_tks = self.remove_template(tokenizer, query_tks, victim_llm)
                        except:
                            print(f'---- empty at {dsi} in generating {fn} ----')
                            empty_records.append({'id':dsi, 'r': ds_record})
                            query_tks = []

                    else:
                        # for training, use the shuffled token list
                        shuffled_tk = self.remove_template(tokenizer, ds_record['g']['i'], victim_llm)
                        random.shuffle(shuffled_tk)
                        # add false positive and fn noise
                        query_tks, _, _ = simulate_noise(1, self.hyperparam.prob_noise, tokenizer, shuffled_tk)
                    
                    for i,tk in enumerate(query_tks):
                        tkstr = escape_illegal_encodings(tokens2text(tokenizer, [tk]), broken_unicode.decode())
                        if len(tkstr.strip()): # skip empty str tokens
                            query.extend(tkstr.encode() + box)

                    query.extend(b'\n\n</Tokens>')
                    
                else:
                    #
                    # No SCA
                    #
                    assert self.hyperparam.has_resp
                    # append responses (we use the groundtruth for traing)
                    # while in the testing, we will reaplce it with the real value predicted by RR
                    # in the subsequnet code
                    query.extend(b'<Response>\n\n')
                    if for_test:
                        # using the predicted responses for testing
                        query.extend(resp[dsi]['pred'].encode())
                    else:
                        # using the original response for training
                        query.extend(escape_illegal_encodings(tokens2text(tokenizer, ds_record['g']['o']), broken_unicode.decode()).encode())
                    query.extend(b'</Response>\n\n')
                    sys_prompt = f'Given the response from a large model, identify the original prompt. Some tokens are unnecessary, others missing. Unicode characters are given as the hexadecimal escape {broken_unicode.decode()}XX.'

            else:
                # empty prompts, do not use it for evaluating
                sys_prompt = ''
                query = b''

            gt_dec_prompts = handle_illegal_encodings(tokens2text(tokenizer, self.remove_template(tokenizer, ds_record['g']['i'], victim_llm)))

            # training data template for OpenAI finetune
            entry = copy.deepcopy(ds_record)
            entry['t'] = {
                'messages': [
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': query.decode()},
                    {'role': 'assistant', 'content': gt_dec_prompts}
                ]
            }
            ft_ds.append(entry)
                
        jsonl_write(fn, ft_ds)
        jsonl_write(fn + '.empty.jsonl', empty_records)


def train(args, attack):
    probed_res_prefix = get_probed_res_prefix(args.device, args.operationsys, args.framework, args.hardware)
    ATK_RESULT_PREFIX=f'{probed_res_prefix}/{GEN_BASE}/{attack.new_model_name}'
    create_directories(ATK_RESULT_PREFIX)

    # downstream task datasets
    train_file = f'{ATK_RESULT_PREFIX}/train_{args.train_ds_collection_id}.jsonl'
    train_sanitized_file = f'{ATK_RESULT_PREFIX}/train_{args.train_ds_collection_id}_sanitized.jsonl'
    val_file = f'{ATK_RESULT_PREFIX}/train_val_{args.train_ds_collection_id}.jsonl'
    test_file =  f'{ATK_RESULT_PREFIX}/test_{args.train_ds_collection_id}.jsonl'

    # load collected cache traces
    train_ds = json.load(open(probed_ds_train_fn(probed_res_prefix, GEN_BASE, args.train_ds_collection_id), 'r'))
    val_ds = json.load(open(probed_ds_val_fn(probed_res_prefix, GEN_BASE, args.train_ds_collection_id, None, 'F16'), 'r')) # used for logging only
    test_ds = json.load(open(probed_ds_test_fn(probed_res_prefix, GEN_BASE, args.train_ds_collection_id, None, 'F16'), 'r'))

    # for attack result
    tokenizer = load_tokenizer(GEN_BASE)

    # extract response and generate the FT dataset from the measured dataset
    ft_job_history_file = f'{ATK_RESULT_PREFIX}/openai_ft_job.json'
    if is_osllm(args.founda_model) or not os.path.exists(ft_job_history_file): # do not regenerate the dataset if train set was submitted to OpenAI
        # gen train set
        attack.gen_ds(GEN_BASE, train_ds, None, train_file, tokenizer,
                    for_test=False)
        
        # gen val set
        if isinstance(attack, RR_Attack):
            # we can provide the right logging ONLY for RR
            attack.gen_ds(GEN_BASE, val_ds, None, val_file, tokenizer,
                        for_test=True)

        elif isinstance(attack, PR_Attack):
            # MARK1
            # The val is ONLY used for the GROUND-TRUTH of prompt (see sanitize_train_dataset())
            # and for osllm: valid set for RR, and unsed for PR
            #     for openai: valid set for RR, and unused for PR
            # Thus, the rr_results_file are not used at all!
            # We will check this again when running evaluating.
            attack.gen_ds(GEN_BASE, val_ds, None, val_file, tokenizer,
                        for_test=True)
        else:
            assert False

        # gen test set
        if isinstance(attack, PR_Attack):
            assert args.rr_model is not None
            rr_test_results_file = f'{probed_res_prefix}/{GEN_BASE}/{args.rr_model}/test_results_{args.train_ds_collection_id}.jsonl'
        else:
            rr_test_results_file = None

        attack.gen_ds(GEN_BASE, test_ds,
                    rr_test_results_file if args.will_test else None, # similar to MARK1, reduce the need of RR results when evaluating PR on LLMs that do not include the GEN_BASE
                    test_file, tokenizer,
                    for_test=True)

        sanitize_train_dataset(#input
                                test_file, val_file, train_file, tokenizer,
                                attack.prompt_as_groundtruth,
                                #output
                                train_sanitized_file)

    with open(train_file + '.modelname', 'w') as fp:
        fp.write(f'{GEN_BASE_VENDOR} {GEN_BASE}')


    if is_osllm(args.founda_model):
        ft_model, ft_tok = osllm_load_model_for_finetune(args.founda_model,
                                                        rank=attack.hyperparam.rank,
                                                        alpha=attack.hyperparam.alpha)

        if False:
            # show token length freqency
            lens=defaultdict(int)
            for d in tqdm(train_ds['text']):
                lens[len(ft_tok.encode(d))]+=1
            print(sorted(lens.items(), key=lambda x : x[0]))
            exit(0)

        train_sanitized_ds = osllm_load_dataset(args.founda_model, train_sanitized_file)
        
        if isinstance(attack, PR_Attack):
            fin_val_ds = None
        else:
            fin_val_file = test_file if args.validate else val_file # used for logging only, cross check
            fin_val_ds = osllm_load_dataset(args.founda_model, fin_val_file)
        
        print(str(attack.hyperparam))
        new_model_path = osllm_finetune(args.founda_model, attack.new_model_name, ft_model, ft_tok, train_sanitized_ds,
                    val_ds=fin_val_ds, # used for logging only
                    lr=attack.hyperparam.lr,
                    train_batch_size=attack.hyperparam.batch_size,
                    epoch=attack.hyperparam.epoch,
                    eval_steps=args.eval_steps)
        write_model_id(new_model_path, f'{ATK_RESULT_PREFIX}/model_id_{args.train_ds_collection_id}_None.json')

    else: # OpenAI

        if not os.path.exists(ft_job_history_file):
            openai_finetune_check(train_sanitized_file, args.founda_model)
            
            assert not args.validate # openai was not used for validating
            fin_val_file = None if isinstance(attack, PR_Attack) else val_file

            ft_job = finetune_openai(train_sanitized_file,
                fin_val_file,
                args.founda_model)
            ftjob = ft_job.id
            print(f'ftjob = {ftjob}')
            json.dump({'ft_job_id': ft_job.id}, open(ft_job_history_file, 'w'))
        else:
            print(f'SKIPPED submit fine-tune job of {attack.new_model_name}')
            ftjob = json.load(open(ft_job_history_file, 'r'))['ft_job_id']

        # wait for the fine tuning 
        ret = wait_ft_job(ftjob)

        ft_model_id = ret.fine_tuned_model
        print(f'ft_model_id = {ret.fine_tuned_model}')
        
        ft_loss_file = f'{ATK_RESULT_PREFIX}/train_results_{args.train_ds_collection_id}.csv'
        ft_hyperparam_file = f'{ATK_RESULT_PREFIX}/train_hyperparam_{args.train_ds_collection_id}.json'
        save_ft_metrics_openai(ft_loss_file, ft_hyperparam_file,
            ftjob = ftjob)

        print(f'ftjob = {ftjob}')
        print(f'ft_model_id = {ft_model_id}')
        write_model_id(ft_model_id, f'{ATK_RESULT_PREFIX}/model_id_{args.train_ds_collection_id}_None.json')

def evaluate(args, attack):
    probed_res_prefix = get_probed_res_prefix(args.device, args.operationsys, args.framework, args.hardware)
    llm_vendor,llm = VICTIM_LLM[args.victim_llm]
    ATK_RESULT_PREFIX=f'{probed_res_prefix}/{llm}/{attack.new_model_name}'
    create_directories(ATK_RESULT_PREFIX)

    # for attack result
    tokenizer = load_tokenizer(llm)

    if attack.hyperparam.has_llm:
        if args.rr_model is not None:
            if args.prefill_bs or args.embd_quant != 'F16':
                rr_test_results_file = f'{probed_res_prefix}/{llm}/{args.rr_model}/test_results_{args.eval_ds_collection_id}{get_suffix(args.prefill_bs, args.embd_quant)}.jsonl'
                rr_val_results_file = None # unsued
            else:
                suffix = get_suffix(args.prefill_bs, args.embd_quant)
                rr_test_results_file = f'{probed_res_prefix}/{llm}/{args.rr_model}/test_results_{args.eval_ds_collection_id}{suffix}.jsonl'
                rr_val_results_file = f'{probed_res_prefix}/{llm}/{args.rr_model}/eval_val_results_{args.eval_ds_collection_id}{suffix}.jsonl'
        else:
            rr_test_results_file = None
            rr_val_results_file = None

        # select the right set for evaluating
        if args.validate:
            assert args.prefill_bs is None
            assert args.embd_quant == 'F16'
            val_file = f'{ATK_RESULT_PREFIX}/eval_val_{args.eval_ds_collection_id}.jsonl'
            val_results_file = f'{ATK_RESULT_PREFIX}/eval_val_results_{args.eval_ds_collection_id}.jsonl'
            fin_test_file=val_file
            fin_test_results_file=val_results_file
        else:
            suffix = get_suffix(args.prefill_bs, args.embd_quant)
            test_file =  f'{ATK_RESULT_PREFIX}/test_{args.eval_ds_collection_id}{suffix}.jsonl'
            test_results_file = f'{ATK_RESULT_PREFIX}/test_results_{args.eval_ds_collection_id}{suffix}.jsonl'
            fin_test_file=test_file
            fin_test_results_file=test_results_file

        batch_job_history_file = f'{ATK_RESULT_PREFIX}/openai_batch_job_history.0.json'
        if is_osllm(args.founda_model) or not os.path.exists(batch_job_history_file): # do not regenerate the datasets if they were submitted to OpenAI
            # test on validation set or test set
            if args.validate:
                val_ds = json.load(open(probed_ds_val_fn(probed_res_prefix, llm, args.eval_ds_collection_id, args.prefill_bs, args.embd_quant), 'r'))
                # generating test involves no random, thus the overwritten test_file will not be changed
                attack.gen_ds(llm, val_ds, rr_val_results_file, val_file, tokenizer,
                                for_test=True)
            else:
                test_ds = json.load(open(probed_ds_test_fn(probed_res_prefix, llm, args.eval_ds_collection_id, args.prefill_bs, args.embd_quant), 'r'))
                # generating test involves no random, thus the overwritten test_file will not be changed
                attack.gen_ds(llm, test_ds, rr_test_results_file, test_file, tokenizer,
                                for_test=True)

        print(attack.hyperparam)

        if is_osllm(args.founda_model):
            if attack.hyperparam.has_finetune:
                model, tokenizer = osllm_load_finetuned_model(args.founda_model, attack.new_model_name)
            else:
                print('Will not use finetuned model !!!')
                model, tokenizer = osllm_load_founda_model(args.founda_model)
            osllm_evaluate(model, args.founda_model, tokenizer,
                        100, # eval batch size
                        fin_test_file,
                        fin_test_results_file)
        else: # OpenAI
            if attack.hyperparam.has_finetune:
                ft_model_id = read_model_id(f'{probed_res_prefix}/{GEN_BASE}/{attack.new_model_name}/model_id_{args.eval_ds_collection_id}_None.json')
            else:
                ft_model_id = args.founda_model

            # check the test set
            nt = openai_get_groundtruth_max_tokens(test_file, args.founda_model)
            assert nt < max_tokens - 50, f'sorry, increase the max_tokens variable to > {nt}'
            check_empty_dataset(test_file)

            batches = openai_dynamic_batching(test_file, args.founda_model)

            print(f'Num batches {len(batches)}')
            batch_job_ids = []
            for bi, batch in enumerate(batches):
                print(f'Eval batch {bi+1}/{len(batches)}')
                batch_job_history_file = f'{ATK_RESULT_PREFIX}/openai_batch_job_history.{bi}.json'
                if not os.path.exists(batch_job_history_file):
                    batch_job, tasks = start_evaluate_openai(batch,
                        ft_model_id = ft_model_id,
                        founda_model = args.founda_model)
                    with open(f'{ATK_RESULT_PREFIX}/openai_batch_input.{bi}.jsonl', 'w') as fp:
                        fp.write(tasks)
                    batch_job_id = batch_job.id
                    print(f'batch_job_id = {batch_job_id}')
                    json.dump({'batch_job_id': batch_job_id}, open(batch_job_history_file, 'w'))
                else:
                    print(f'SKIPPED submit batch job for {attack.new_model_name}')
                    batch_job_id = json.load(open(batch_job_history_file, 'r'))['batch_job_id']
                batch_job_ids.append(batch_job_id)

            # wait for the batch jobs
            test_offset = 0
            for bi, batch_job_id in enumerate(batch_job_ids):
                part_file = fin_test_results_file + f'.part{bi}'
                if not os.path.exists(part_file):
                    ret = wait_batch_job(batch_job_id)
                    result_file_id = ret.output_file_id
                    get_evaluate_openai(fin_test_file,          
                        test_offset,
                        part_file,
                        # params
                        ft_model_id,
                        result_file_id = result_file_id)
                else:
                    print(f'SKIPPED get results for {batch_job_id}')
                    
                test_offset += len(batches[bi])
            
            # merge the batches
            lines = []
            for bi in range(len(batches)):
                lines.extend(open(fin_test_results_file + f'.part{bi}', 'r').readlines()) # contains \n
            with open(fin_test_results_file, 'w') as fp:
                fp.write(''.join(lines))
        
    else: # no LLM
        print('no LLM')
        if args.validate:
            test_file = f'{ATK_RESULT_PREFIX}/eval_val_{args.eval_ds_collection_id}.jsonl'
            test_results_file = f'{ATK_RESULT_PREFIX}/eval_val_results_{args.eval_ds_collection_id}.jsonl'
            test_ds = json.load(open(probed_ds_val_fn(probed_res_prefix, llm, args.eval_ds_collection_id, args.prefill_bs, args.embd_quant), 'r'))

        else:
            suffix = get_suffix(args.prefill_bs, args.embd_quant)
            test_file =  f'{ATK_RESULT_PREFIX}/test_{args.eval_ds_collection_id}{suffix}.jsonl'
            test_results_file = f'{ATK_RESULT_PREFIX}/test_results_{args.eval_ds_collection_id}{suffix}.jsonl'
            test_ds = json.load(open(probed_ds_test_fn(probed_res_prefix, llm, args.eval_ds_collection_id, args.prefill_bs, args.embd_quant), 'r'))

        res = []
        with open(test_results_file, 'w') as fp:
            for r, record in enumerate(tqdm(test_ds['results'])):
                timepoints, tokens, _ = get_tp_tk(record)
                (_, tk_prompt), (_, tk_resp) = split_prompt_response(timepoints, tokens)

                if attack.prompt_as_groundtruth:
                    pred = tokens2text(tokenizer, tk_prompt).decode()
                    gt = record['g']['i']
                else:
                    pred = tokens2text(tokenizer, tk_resp).decode()
                    gt = record['g']['o']

                entry = copy.deepcopy(record)
                entry['pred'] = pred 
                entry['t'] = {'messages': [
                    {'role': 'system', 'content': ''},
                    {'role': 'user', 'content': ''},
                    {'role': 'assistant', 'content':  tokens2text(tokenizer, gt).decode()} 
                ]}
                res.append(entry)
        jsonl_write(test_results_file, res)

def main():
    parser = deepspeed.add_config_arguments(get_attack_argparser())
    args = parser.parse_args()

    # check param combinations
    assert args.attack != 'pr' or (args.rr_model is not None and args.has_resp is not None)
    assert args.attack != 'pr' or (args.has_sca or args.has_resp)
    assert args.attack != 'rr' or (args.sigma is not None)
    assert args.mode != 'eval' or (args.eval_ds_collection_id is not None and args.victim_llm is not None)
    assert args.mode != 'train' or (args.train_ds_collection_id is not None) 
    
    if args.founda_model[:4] == 'gpt-':
        init_openai(args.mock_gpt)
        pass

    ATTACKS = {
        'pr': PR_Attack,
        'rr': RR_Attack,
    }
    attack = ATTACKS[args.attack](args)

    print('============')
    print(f"=== {args.mode} {attack.new_model_name} llm={VICTIM_LLM[args.victim_llm][1] if args.mode=='eval' else GEN_BASE} val={args.validate}")
    print('============')
    
    if args.mode=='train':
        if args.has_llm and args.has_finetune:
            train(args, attack)

    elif args.mode=='eval':
        evaluate(args, attack)
    else:
        assert False

if __name__ == '__main__':
    main()
