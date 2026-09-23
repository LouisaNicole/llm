

from datasets import load_dataset, Dataset

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, Trainer, TrainingArguments

from transformers import AutoTokenizer, GenerationConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTConfig, SFTTrainer
import torch
from collections.abc import Mapping
import json
from tqdm import tqdm
import math
import numpy as np

import Levenshtein
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

from experiment_common import DynamicAvg, jsonl_read, jsonl_write, dynamic_batching, safe_rouge
from constants import _osllm_model_fn, _osllm_out_ckp_fn, _osllm_out_lora_model_fn, max_new_tokens, max_in_tokens
import copy

def osllm_load_dataset(founda_model, fn):
    def preprocess_function(examples):
        if founda_model == 'Llama-3.1-8B-Instruct':
            inputs = [f"<|start_header_id|>system<|end_header_id|>{sys}" +
                    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{usr}"
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" + gt + "<|eot_id|><|end_of_text|>"
                        for sys, usr, gt in zip(examples["system"], examples["user"], examples["assistant"])]
        else:
            assert False
        #return tokenizer(inputs, padding="max_length", truncation=True, max_length=32)

        model_inputs = {}
        model_inputs['text'] = inputs
        return model_inputs

    sys = []
    usr = []
    gt = []
    with open(fn, 'r') as fp:
        for line in fp.readlines():
            r = json.loads(line)['t']
            assert r['messages'][0]['role'] == 'system'
            sys.append(r['messages'][0]['content'] )
            assert r['messages'][1]['role'] == 'user' 
            usr.append(r['messages'][1]['content'] )
            assert r['messages'][2]['role'] == 'assistant'
            gt.append(r['messages'][2]['content'] )

    dataset = Dataset.from_dict({"system": sys, "user": usr,
                                "assistant": gt})

    return dataset.map(preprocess_function, batched=True,num_proc=8)

def osllm_get_pad_token_id(founda_model, tokenizer):
    if founda_model == 'Llama-3.1-8B-Instruct':
        pad_token_id = tokenizer.encode('<|finetune_right_pad_id|>',add_special_tokens=False)[0]
        assert tokenizer.decode(pad_token_id) == '<|finetune_right_pad_id|>'
    else:
        assert False
    return pad_token_id

def osllm_load_backbone_model(model_name, founda_model):
    #bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    #)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pad_token_id = osllm_get_pad_token_id(founda_model, tokenizer)
    
    tokenizer.padding_side = "right"
    if founda_model == 'Llama-3.1-8B-Instruct':
        tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'}) # for llama3
    else:
        assert False

    model = AutoModelForCausalLM.from_pretrained(model_name,
                    pad_token_id=pad_token_id,
                    device_map='cuda',
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2"
                    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer

def osllm_load_model_for_finetune(founda_model, rank=64, alpha=16):
    model, tokenizer = osllm_load_backbone_model(_osllm_model_fn(founda_model), founda_model)

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        bias="none",
        task_type='CAUSAL_LM'
    )
    lora_model = get_peft_model(model, peft_config)

    return lora_model, tokenizer

def osllm_load_founda_model(founda_model):
    return osllm_load_backbone_model(_osllm_model_fn(founda_model),
                                             founda_model)

def osllm_load_finetuned_model(founda_model, new_model_name):
    return osllm_load_backbone_model(_osllm_out_lora_model_fn(new_model_name),
                                             founda_model)

# NOTE that the logging of val set is invalid and dropped
def osllm_finetune(founda_model, new_model_name, ft_model, tokenizer : AutoTokenizer, train_ds, val_ds,
                   lr=3e-5, train_batch_size=2, epoch=3, eval_steps=200):
    deepspeed_config = {
        "train_batch_size": "auto", # read from training_args
        "gradient_accumulation_steps": "auto", # read from training_args
        "zero_optimization": {
            "stage": 0,  # ZeRO Stage 3: Parameter Partitioning
            "stage3_gather_16bit_weights_on_model_save":True,
            #"offload_param": {
            #    "device": "cpu",
            #    "pin_memory": True
            #},
            #"offload_optimizer": {
            #    "device": "auto",
            #    "pin_memory": True
            #},
            "overlap_comm": True,
            "contiguous_gradients": False,
        },
        "bf16": {"enabled": True}, 
        "pipeline": {"parallelization": True}, # pipeline parallelization
    }
    
    max_seq_length = 6000+4096 #4096
    training_args = SFTConfig(
        bf16=True,
        output_dir=_osllm_out_ckp_fn(new_model_name),
        eval_strategy="steps" if val_ds else 'no',
        eval_steps=eval_steps if val_ds else None,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        per_device_train_batch_size=train_batch_size // 1, # 1 GPU
        per_device_eval_batch_size=3,
        num_train_epochs=epoch,
        #max_steps=1, # for DEBUG only
        logging_steps=8,
        weight_decay=0,
        logging_dir=_osllm_out_ckp_fn(new_model_name) + f"/logs",
        save_strategy="no",
        #save_steps=1, # for DEBUG only
        save_total_limit=3,
        max_seq_length=max_seq_length,
        packing=False,
        dataset_text_field="text",
        deepspeed=deepspeed_config,
        overwrite_output_dir=True
    )

    rouge = Rouge()

    end_header_id_token_id = tokenizer.encode('<|end_header_id|>',
                                              add_special_tokens=False)[0]

    def get_response_pos(gt):
        cnt = 0
        for i in range(len(gt)):
            if gt[i] == end_header_id_token_id:
                cnt += 1
                if cnt == 3:
                    p = i
                    break
        else:
            assert False
        return p+1
    
    def compute_metrics(pred):
        # not suitable for tokens
        labels_ids = np.array(pred.label_ids)
        pred_ids = np.array(pred.predictions[0])
        #print(labels_ids.shape, pred_ids.shape)
        #pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        #print(pred_str)
        bleu_score = DynamicAvg()
        ld_score = DynamicAvg()
        r2_score = DynamicAvg()
        rl_score = DynamicAvg()
        rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l", "rouge-4"])
        for l,p in zip(labels_ids, pred_ids):
            valid_indices = l >= 0
            resp_pos = get_response_pos(l[valid_indices])
            gt = l[valid_indices][resp_pos:]
            pred = p[valid_indices][resp_pos:]
            if False:
                print("Gt:", tokenizer.decode(gt))
                print('====')
                print('Pd:', tokenizer.decode(pred))
            chencherry = SmoothingFunction()
            bleu_score.update(sentence_bleu([gt],
                                    pred,
                                    smoothing_function=chencherry.method1))
            gt_t = tokenizer.decode(gt, skip_special_tokens=True)
            pred_t = tokenizer.decode(pred, skip_special_tokens=True)
            ld_score.update(1- Levenshtein.distance(gt_t, pred_t) / max(len(gt_t), len(pred_t)))
            rouge_val = rouge.get_scores(gt_t, pred_t)
            r2_score.update(rouge_val[0]['rouge-4']['f'])
            rl_score.update(rouge_val[0]['rouge-l']['f'])
        
        return {'BLEU': bleu_score.get(),
                'rougel': rl_score.get(),
                'rouge2': r2_score.get(),
                'LD': ld_score.get()} 

    def preprocess_logits_for_metrics(logits, labels):
        """
        This is a workaround to avoid storing too many and huge logits tensors
        that are not needed.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    for ent in train_ds:
        assert len(ent['system']) and len(ent['user']), 'check the --will-test option'
        n = len(tokenizer.encode(ent['text']))
        assert n < max_seq_length-11, f'too long {n}'

    trainer = SFTTrainer(
        model=ft_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds.select(range(60)) if val_ds else None, # for logging only, not used for early stop or other meta optimization
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    # FIXME: the displayed 'eval_loss' metric seems be wrong. (but not affecting the training results)

    trainer.train()
    new_model_path = _osllm_out_lora_model_fn(new_model_name)
    print(f'saving to "{new_model_path}"')
    trainer.save_model(new_model_path)
    return new_model_path

def osllm_evaluate(model, founda_model, tokenizer, batch_size,
                   # input
                   test_file,
                   # output
                   test_result_file):

    model.eval()

    tokenizer.padding_side = "left" # for genration  only
    pad_token_id = osllm_get_pad_token_id(founda_model, tokenizer)
    tokenizer.pad_token_id = pad_token_id
    
    generation_config = GenerationConfig(
        penalty_alpha=1,do_sample = True,
        top_k=5,temperature=1,repetition_penalty=1, # no penalty
        max_new_tokens=max_new_tokens, pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    def eval_batch(inputs, device='cuda'):
        tk = tokenizer(inputs, padding=True, truncation=True,
                        return_tensors="pt") #max_length=max_in_tokens,

        for sid, ids_per_seq in enumerate(tk['input_ids']):
            assert len(ids_per_seq) < 6000, f'{sid}: input too long {len(ids_per_seq)}'

        #print('mkt=', max([len(tks) for tks in tk['input_ids']]))
        with torch.no_grad(): 
            input_ids = tk['input_ids'].to(device)
            out = model.generate(input_ids=input_ids,
                attention_mask=tk['attention_mask'].to(device),
                generation_config=generation_config,
            )
            res = []
            for i, o in enumerate(out):
                res.append(tokenizer.decode(o[len(input_ids[i]):],
                                            skip_special_tokens=True))
            return res

    # read openai formatted test file
    sys = []
    usr = []
    gt = []
    test_ds = jsonl_read(test_file)
    for index, record in enumerate(tqdm(test_ds)):
        msg = record['t']
        assert msg['messages'][0]['role'] == 'system'
        sys.append(msg['messages'][0]['content'])
        assert msg['messages'][1]['role'] == 'user'

        assert len(msg['messages'][0]['content']) and len(msg['messages'][1]['content']), 'check the --will-test option'

        usr.append(msg['messages'][1]['content'])
        assert msg['messages'][2]['role'] == 'assistant'
        gt.append(msg['messages'][2]['content'])

    # template for Llama3.1
    inputs = [f"<|start_header_id|>system<|end_header_id|>{sys}" +
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{usr}"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                for sys, usr in zip(sys, usr)]
    res = []
    pos = 0
    for bi, b in enumerate(tqdm(dynamic_batching(inputs, tokenizer, 120*batch_size))):
        ret = eval_batch(b)
        for si, s in enumerate(ret):
            groundtruth = gt[pos]
            pos += 1
            pred = s

            if si==len(ret)-1:
                print('Gt: ', groundtruth)
                print('====')
                print('Pd: ', pred)

                ld_score = 1 - Levenshtein.distance(groundtruth, pred) / max(len(groundtruth), len(pred))
                
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
                pr, rc, f1 = get_precision_recall_f1(groundtruth, pred)

                print(f'LD score = {ld_score}')
                print(f'Pr {pr*100:.1f}% Rc {rc*100:.1f}% F1 {f1}')

                chencherry = SmoothingFunction()
                bleu_val = sentence_bleu([tokenizer.encode(groundtruth)],
                                        tokenizer.encode(pred),
                                        smoothing_function=chencherry.method1)
                
                rouge_val = safe_rouge(groundtruth, pred)
                r1 = rouge_val[0]['rouge-1']['f']
                r2 = rouge_val[0]['rouge-2']['f']
                rL = rouge_val[0]['rouge-l']['f']

                print(f'BLEU = {bleu_val}')
                print(f'rougle {r1} {r2} {rL}')

        res.extend(ret)

    out = []
    for index, r in enumerate(res):
        entry = copy.deepcopy(test_ds[index])
        entry['pred'] = r
        out.append(entry)
    jsonl_write(test_result_file, out)
