#ifndef TOKENIZER_HH_
#define TOKENIZER_HH_

#include "gguf.h"
#include <stdarg.h>
#include <limits.h>
#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>

typedef int32_t llama_pos;
typedef uint32_t llama_token;
typedef int32_t llama_seq_id;

enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_SPM = 0, // SentencePiece
    LLAMA_VOCAB_TYPE_BPE = 1, // Byte Pair Encoding
};

enum llama_token_type {
    LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
    LLAMA_TOKEN_TYPE_NORMAL       = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
    LLAMA_TOKEN_TYPE_CONTROL      = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED       = 5,
    LLAMA_TOKEN_TYPE_BYTE         = 6,
};

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;
    using ttype = llama_token_type;

    struct token_data {
        token text;
        float score;
        ttype type;
    };

    enum llama_vocab_type type = LLAMA_VOCAB_TYPE_SPM;

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    std::unordered_map<token, id> special_tokens_cache;

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // default LLaMA special tokens
    id special_bos_id = 1;
    id special_eos_id = 2;
    id special_unk_id = 0;
    id special_sep_id = -1;
    id special_pad_id = -1;

    id linefeed_id       = 13;
    id special_prefix_id = 32007;
    id special_middle_id = 32009;
    id special_suffix_id = 32008;
    id special_eot_id    = 32010;

    int find_bpe_rank(std::string token_left, std::string token_right) const {
        GGML_ASSERT(token_left.find(" ") == std::string::npos);
        GGML_ASSERT(token_left.find("\n") == std::string::npos);
        GGML_ASSERT(token_right.find(" ") == std::string::npos);
        GGML_ASSERT(token_right.find("\n") == std::string::npos);

        auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
        if (it == bpe_ranks.end()) {
            return -1;
        }

        return it->second;
    }
};


//
// gguf constants (sync with gguf.py)
//

enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    LLM_ARCH_BAICHUAN,
    LLM_ARCH_GROK,
    LLM_ARCH_GPT2,
    LLM_ARCH_GPTJ,
    LLM_ARCH_GPTNEOX,
    LLM_ARCH_MPT,
    LLM_ARCH_STARCODER,
    LLM_ARCH_REFACT,
    LLM_ARCH_BERT,
    LLM_ARCH_NOMIC_BERT,
    LLM_ARCH_JINA_BERT_V2,
    LLM_ARCH_BLOOM,
    LLM_ARCH_STABLELM,
    LLM_ARCH_QWEN,
    LLM_ARCH_QWEN2,
    LLM_ARCH_QWEN2MOE,
    LLM_ARCH_PHI2,
    LLM_ARCH_PHI3,
    LLM_ARCH_PLAMO,
    LLM_ARCH_CODESHELL,
    LLM_ARCH_ORION,
    LLM_ARCH_INTERNLM2,
    LLM_ARCH_MINICPM,
    LLM_ARCH_MINICPM3,
    LLM_ARCH_GEMMA,
    LLM_ARCH_GEMMA2,
    LLM_ARCH_STARCODER2,
    LLM_ARCH_MAMBA,
    LLM_ARCH_XVERSE,
    LLM_ARCH_COMMAND_R,
    LLM_ARCH_DBRX,
    LLM_ARCH_OLMO,
    LLM_ARCH_OLMOE,
    LLM_ARCH_OPENELM,
    LLM_ARCH_ARCTIC,
    LLM_ARCH_DEEPSEEK2,
    LLM_ARCH_CHATGLM,
    LLM_ARCH_BITNET,
    LLM_ARCH_T5,
    LLM_ARCH_T5ENCODER,
    LLM_ARCH_JAIS,
    LLM_ARCH_NEMOTRON,
    LLM_ARCH_EXAONE,
    LLM_ARCH_RWKV6,
    LLM_ARCH_GRANITE,
    LLM_ARCH_GRANITE_MOE,
    LLM_ARCH_CHAMELEON,
    LLM_ARCH_BAMBOO,

    LLM_ARCH_UNKNOWN,
};

void llm_load_arch(struct gguf_context *ctx_gguf, llm_arch &arch);
void llm_load_hparams(struct gguf_context * ctx, llm_arch arch,
                        uint32_t &n_vocab, uint32_t &n_embd);
int llm_load_vocab(struct gguf_context * ctx, llama_vocab &vocab, llm_arch arch);

std::vector<llama_vocab::id> tokenize(const llama_vocab & vocab, std::string raw_text, bool bos, bool special = false);
int llama_token_to_piece(unsigned n_vocab, const llama_vocab &voacb,
                        llama_token token, char * buf, int length);

#endif