from experiment_common import *
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from peft import PeftModel, PeftConfig

class Angle_Embedding:
    def __init__(self):
        peft_model_id = EMBEDDING_MODELS_PATH + '/SeanLee97/angle-llama-7b-nli-20231027'
        config = PeftConfig.from_pretrained(peft_model_id)
        base_model_name_or_path = 'NousResearch/Llama-2-7b-hf'
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path,
                                                          device_map='auto',
                                                          torch_dtype=torch.bfloat16,
                                                          attn_implementation="flash_attention_2",
                                                          offload_folder="offload/" # reserved for PEFT
                                                          )
        self.model = PeftModel.from_pretrained(self.model, peft_model_id,
                                                offload_folder="offload/")#.cuda()
    
    def encode(self, s : list, batch_size):
        def set_prompt(text: str):
            # This is the standard prompt recommened by the official 
            # see https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py and
            # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 for details
            return f'Summarize sentence "{text}" in one word:"'

        def eval_batch(bs):
            tok = self.tokenizer([set_prompt(i) for i in bs],
                                 padding=True, truncation=True, max_length=4096, return_tensors='pt')
            for k, v in tok.items():
                tok[k] = v.cuda()
            # pool the hidden states of the last layer
            outputs = self.model(output_hidden_states=True, **tok).hidden_states[-1]
            vec = torch.sum(
                    outputs * tok["attention_mask"][:, :, None], dim=1) / tok["attention_mask"].sum(dim=1).unsqueeze(1)
            return vec

        res = []
        for bi, b in enumerate(tqdm(dynamic_batching(s, self.tokenizer, 100*batch_size))):
            res.append(eval_batch(b))
        return torch.cat(res, dim=0)

    def cos_sim(self, s1 : list, s2 : list, batch_size):
        return F.cosine_similarity(self.encode(s1, batch_size), self.encode(s2, batch_size))

def test():
    embed = Angle_Embedding()
    embed.cos_sim(["Compose a poem all in lowercase letters about my friend Barnet.",
                '"Breach Posterior" is a targeted cure for cancer. Make a pitch deck aimed towards moms that uses the word "clearly" at least 2 times and has a casual tone.'],
                ['Compose a poem about my friend Barnet. Use all lowercase letters.',
                'Breach Posterior is a startup that has a cure for cancer. Write a casual pitch deck for it that\'s targeted towards moms. Make sure to use the word "clearly" at least 2 times.'
                ]
                )

def analyze_results(# input
                    test_results_file,
                    # output
                    test_restored_response_file,
                    target_tokenizer,
                    ):
    '''
    @param test_file giving the groundtruth
    '''
    embed = Angle_Embedding()
    test_restored_response = []
    avg_f1 = DynamicAvg()
    avg_ld_score = DynamicAvg()
    avg_bleu = DynamicAvg()
    avg_rouge1 = DynamicAvg()
    avg_rouge2 = DynamicAvg()
    avg_rouge4 = DynamicAvg()
    avg_rougeL = DynamicAvg()
    avg_cos_sim = DynamicAvg()
    n_prompts = 0
    n_exact = 0
    verbose = False
    with open(test_results_file, 'r') as fp_res:
        groundtruths = []
        preds = []
        fp_lines = fp_res.readlines()
        for index, line in enumerate(tqdm(fp_lines)):
            res_record = json.loads(line)
            preds.append(res_record['pred'].strip())
            assert res_record['t']['messages'][2]['role']=='assistant'
            groundtruths.append(res_record['t']['messages'][2]['content'].strip())

        cos_sims = embed.cos_sim(preds, groundtruths, batch_size=EMBEDDING_BATCH_SIZE)

        for index, line in enumerate(tqdm(fp_lines)):
            res_record = json.loads(line)
            pred = preds[index] 
            groundtruth = groundtruths[index] 
            if len(groundtruth) == 0:
                ld_score = 1 if len(pred) == 0 else 0
            else:
                ld_score = 1 - Levenshtein.distance(groundtruth, pred) / max(len(groundtruth), len(pred))
            
            pr, rc, f1 = get_precision_recall_f1(groundtruth, pred)
            avg_f1.update(f1)
            avg_ld_score.update(ld_score)

            if verbose:
                print(f'LD score = {ld_score}')
                print(f'Pr {pr*100:.1f}% Rc {rc*100:.1f}% F1 {f1}')

            chencherry = SmoothingFunction()
            bleu_val = sentence_bleu([text2tokens(target_tokenizer, groundtruth)],
                                    text2tokens(target_tokenizer, pred),
                                    smoothing_function=chencherry.method1)
            avg_bleu.update(bleu_val)
            rouge_val = safe_rouge(groundtruth, pred)
            r1 = rouge_val[0]['rouge-1']['f']
            r2 = rouge_val[0]['rouge-2']['f']
            r4 = rouge_val[0]['rouge-4']['f']
            rL = rouge_val[0]['rouge-l']['f']
            cos_sim = cos_sims[index].item()
            avg_rouge1.update(r1)
            avg_rouge2.update(r2)
            avg_rouge4.update(r4)
            avg_rougeL.update(rL)
            avg_cos_sim.update(cos_sim)
            
            a = remove_punctuation(groundtruth)
            b = remove_punctuation(pred)
            exact = a == b
            
            if False and ld_score > 0.9 and not exact:
                diff = difflib.unified_diff(groundtruth.splitlines(), pred.splitlines(), fromfile='gt', tofile='pred')
                #print('\n'.join(diff))
                #print(index)
                print('gt---:',a)
                print('pred---:', b)
                #exit(1)

            entry = copy.deepcopy(res_record)
            entry['bleu'] = bleu_val
            entry['r1'] = r1
            entry['r2'] = r2
            entry['r4'] = r4
            entry['rL'] = rL
            entry['LD'] = ld_score
            entry['exact'] = exact
            entry['cos_sim'] = cos_sim
            test_restored_response.append(entry)
            if (exact):
                n_exact += 1
            #else:
            #    print('=== gt:', groundtruth)
            #    print('=== pred:', pred)
            #    wait_for_keypress()
            n_prompts += 1

    print(test_results_file)
    print(f'Avg LD score = {avg_ld_score.get()}')
    print(f'Avg F1 = {avg_f1.get()}')
    print(f'Avg Rouge-1 = {avg_rouge1.get()}')
    print(f'Avg Rouge-2 = {avg_rouge2.get()}')
    print(f'Avg Rouge-4 = {avg_rouge4.get()}')
    print(f'Avg Rouge-L = {avg_rougeL.get()}')
    print(f'Avg BLEU1^8 = {avg_bleu.get()}')
    print(f'Avg cos sim = {avg_cos_sim.get()}')
    print(f'Exact ASR = {n_exact/n_prompts*100}%')

    jsonl_write(test_restored_response_file, test_restored_response)
    return test_restored_response

def analyze_single_llm(victim_llm, device, operationsys, framework, hardware, new_model_name, validate, eval_ds_collection_id, prefill_bs, embd_quant):
    llm_vendor,llm = VICTIM_LLM[victim_llm]
    probed_res_prefix = get_probed_res_prefix(device, operationsys, framework, hardware)
    PATH_PREFIX=f'{probed_res_prefix}/{llm}/{new_model_name}'
    if not validate:
        suffix = get_suffix(prefill_bs, embd_quant)
        test_results_file = f'{PATH_PREFIX}/test_results_{eval_ds_collection_id}{suffix}.jsonl'
        analyzed_test_results_file = f'{PATH_PREFIX}/analyzed_test_results_{eval_ds_collection_id}{suffix}.jsonl'
    else:
        assert not prefill_bs, 'unused'
        assert embd_quant == 'F16', 'unused'
        test_results_file = f'{PATH_PREFIX}/eval_val_results_{eval_ds_collection_id}.jsonl'
        analyzed_test_results_file = f'{PATH_PREFIX}/analyzed_eval_val_results_{eval_ds_collection_id}.jsonl'

    # for attack result
    tokenizer = load_tokenizer(llm)
    return analyze_results(test_results_file, analyzed_test_results_file, tokenizer)

def main():
    import args 
    import deepspeed
    parser = args.get_analyze_results_argparser()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.victim_llm is not None:
        analyze_single_llm(args.victim_llm,
                           args.device, args.operationsys, args.framework, args.hardware,
                           args.new_model_name, args.validate, args.eval_ds_collection_id, args.prefill_bs, args.embd_quant)

    else:
        assert False, 'unused'
        bleu = DynamicAvg()
        r1 = DynamicAvg()
        r2 = DynamicAvg()
        r4 = DynamicAvg()
        rL = DynamicAvg()
        ld_score = DynamicAvg()
        exact = DynamicAvg()
        cos_sim = DynamicAvg()
        for i in MAIN_VICTIM_LLMS:
            ret = analyze_single_llm(i, args.new_model_name, args.validate, args.eval_ds_collection_id, args.prefill_bs, args.embd_quant)
            for entry in ret: 
                bleu.update(entry['bleu'])
                r1.update(entry['r1'])
                r2.update(entry['r2'])
                r4.update(entry['r4'])
                rL.update(entry['rL'])
                ld_score.update(entry['LD'])
                exact.update(entry['exact'])
                cos_sim.update(entry['cos_sim'])
        
        ANALYZED_PATH = f'{probed_res_prefix}/summary/{args.new_model_name}'
        create_directories(ANALYZED_PATH)
        entry = {
            'bleu': bleu.get(),
            'r1': r1.get(),
            'r2': r2.get(),
            'r4': r4.get(),
            'rL': rL.get(),
            'LD': ld_score.get(),
            'exact': exact.get(),
            'cos_sim': cos_sim.get(),
        }
        print('summary', entry)
        json.dump(entry, open(f'{ANALYZED_PATH}/analyzed_{"eval_val" if args.validate else "test"}_results_{args.eval_ds_collection_id}.json', 'w')) 

if __name__ == '__main__':
    main()