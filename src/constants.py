VICTIM_LLM=[
    ('google', 'gemma-2-9b-it'),
    ("microsoft", 'Phi-3.5-mini-instruct'),
    ("tiiuae", 'Falcon3-10B-Instruct'),
    ("meta", 'Llama-3.1-8b-instruct'),
    ('mistral', 'Mistral-7b-instruct'),
    ('tiiuae', 'Falcon3-7B-Instruct-1.58bit'),
    ('tiiuae', 'Falcon3-1B-Instruct'),
]

MAIN_VICTIM_LLMS = range(5)

REDUCED_LLMS = [0, 1]

def get_llm_index(llm_name):
    for llm_i, (_, llm) in enumerate(VICTIM_LLM):
        if llm_name == llm:
            break
    else:
        assert False, f'Unknown model name {llm_name}'
    return llm_i

FIGURE_OUT_PATH='../output_figures/'


EMBEDDING_MODELS_PATH='../thirdparty_models'
TOKENIZER_PATH='../thirdparty_models/tokenizers'

EMBEDDING_BATCH_SIZE=15 # reduce this value if you encountered CUDA out of memory

# path to the constructed dataset
GEN_DS_PREFIX='../dataset/generated'

def _osllm_model_fn(founda_model):
    ''' get the pathname of the base model '''
    return f"../thirdparty_models/{founda_model}"
def _osllm_out_ckp_fn(new_model):
    return f"../models/checkpoints-{new_model}"
def _osllm_out_lora_model_fn(new_model):
    return f'../models/{new_model}'

def get_probed_res_prefix(device, operationsys, framework, hardware):
    return f'../results/{device}/{operationsys}/{framework}/{hardware}'

def get_train_probed_res_prefix():
    return f'../results/gpu/ubuntu22_04/llama.cpp/Intel 13900K' # path to the collected LLM output and input text. These text serve as the corpus dataset.
                                             # Note that the cache traces that depends on the (device,operationsys,frmaework,hadrware)
                                             # will NOT be used to train models!
                                             # Instead, the training phases ONLY use the out and in text (tokens).
                                             # see functions in the exp_attack.py for details.

# prompting dataset
def prompting_ds_fn(ds_collection_id):
    return GEN_DS_PREFIX+ f'/{ds_collection_id}.json'
def prompting_ds_test_fn(ds_collection_id):
    return GEN_DS_PREFIX+ f'/{ds_collection_id}_test.json'
def prompting_ds_train_fn(ds_collection_id):
    return GEN_DS_PREFIX+ f'/{ds_collection_id}_train.json'
def prompting_ds_val_fn(ds_collection_id):
    return GEN_DS_PREFIX+ f'/{ds_collection_id}_val.json'

def get_suffix(prefill_bs, embd_quant):
    suffix = ''
    if prefill_bs:
        suffix = f'_bs{prefill_bs}'
    if embd_quant != 'F16':
        suffix += f'_embd{embd_quant}'
    return suffix

# probed dataset (collected from the victim)
def probed_ds_train_fn(probed_res_prefix, victim_llm, ds_collection_id):
    return f'{probed_res_prefix}/{victim_llm}_{ds_collection_id}_train.json'
def probed_ds_val_fn(probed_res_prefix, victim_llm, ds_collection_id, prefill_bs, embd_quant):
    assert prefill_bs is None, 'unused'
    assert embd_quant == 'F16', 'unsued'
    return f'{probed_res_prefix}/{victim_llm}_{ds_collection_id}_val.json'
def probed_ds_test_fn(probed_res_prefix, victim_llm, ds_collection_id, prefill_bs, embd_quant):
    return f'{probed_res_prefix}/{victim_llm}_{ds_collection_id}_test{get_suffix(prefill_bs, embd_quant)}.json'

def is_osllm(founda_model):
    return founda_model[:4] != 'gpt-'

max_in_tokens = 4096
max_new_tokens = 4096
max_tokens = max_in_tokens + max_new_tokens 
OPENAI_MAX_ENQUEUE_TOKENS = 1000000 #2000000 - 1000

ASR_COS_THRESHOLD = 0.77 # drawn from the survey

#########################
# Generate dataset
#########################
GEN_BASE_LLM_ID=3
GEN_BASE_VENDOR, GEN_BASE=VICTIM_LLM[GEN_BASE_LLM_ID]

# model dependent parameters
def get_delimiter_token(founda_model):
    # note that all of thee speical symbols must be 1-token for the FT model
    return b'|'

def get_spaced_tbe(founda_model):
    if founda_model == 'Llama-3.1-8B-Instruct':
       return True
    elif founda_model[:6] == 'gpt-4o':
       return True
    else:
        # add new config before here
        assert f'Unknown model {args.founda_model}'

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--query-llm-base', action="store_true", default=False)
    parser.add_argument('--query-llm-id', type=int, default=None)
    args = parser.parse_args()

    if args.query_llm_base:
        print(GEN_BASE)
    elif args.query_llm_id is not None:
        print(VICTIM_LLM[args.query_llm_id][1])
    else:
        assert False
