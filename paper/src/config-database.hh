#ifndef CONFIG_DATABASE_HH_
#define CONFIG_DATABASE_HH_

#include <iostream>
#include <cstdint>
#include <cassert>
#include <string>
#include <functional>
#include "util.h"

struct Config {
    // hardware
    uint64_t high_threshold, low_threshold{0};

    // software
    std::string token_embd_name{"token_embd.weight"},
                layer0_q_weight_name{"blk.0.attn_q.weight"},
                layer0_k_weight_name{"blk.0.attn_k.weight"},
                layer0_v_weight_name{"blk.0.attn_v.weight"};
    unsigned n_gpu_layers;
    std::function<std::string (const std::string &)> chat_template;

    Config(const std::string &llm) {
        if (!llm.compare("Mistral-7b-instruct") || !llm.compare("Llama-2-7b-chat")) {
            n_gpu_layers = 33;
            chat_template = [](const std::string &user_prompt) -> std::string {
                // ref: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
                return std::string("[INST]") + user_prompt + "[/INST]";
            };

        } else if (!llm.compare("Llama-3.1-8b-instruct")) {
            n_gpu_layers = 33;
            chat_template = [](const std::string &user_prompt) -> std::string {
                // ref: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/#special-tokens-used-with-meta-llama-3
                return std::string("<|start_header_id|>user<|end_header_id|>\n\n")
                    + user_prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";                    
            };
        } else if (!llm.compare("Phi-3.5-mini-instruct")) {
            n_gpu_layers = 33;
            chat_template = [](const std::string &user_prompt) -> std::string {
                return std::string("<|user|>") + user_prompt + "<|end|><|assistant|>";
            };
        } else if (!llm.compare("gemma-2-9b-it")){
            n_gpu_layers = 43;
            chat_template = [](const std::string &user_prompt) -> std::string {
                return std::string("<start_of_turn>user\n") + user_prompt + "<end_of_turn>\n<start_of_turn>model\n";
            };

        } else if (!llm.compare("Falcon3-10B-Instruct") || !llm.compare("Falcon3-7B-Instruct-1.58bit") || !llm.compare("Falcon3-1B-Instruct")) {
            n_gpu_layers = 41; // FIXME
            chat_template = [](const std::string &user_prompt) -> std::string {
                return std::string("<|user|>\n") + user_prompt + "\n<|assistant|>\n";
            };

        } else {
            std::cerr << "Unknown model " << llm << std::endl;
            assert(0);
        }
    }
};

struct ConfigIntel12700KF : public Config {
    ConfigIntel12700KF(const std::string &llm, bool skip_l1_hit) : Config(llm) {
        high_threshold = 290;
        if (skip_l1_hit) {
            low_threshold = 80;
        }
    }
};

struct ConfigIntel13900K : public Config {
    ConfigIntel13900K(const std::string &llm, bool skip_l1_hit) : Config(llm) {
        high_threshold = 190;
        if (skip_l1_hit) {
            low_threshold = 90;
        }
    }
};

struct ConfigIntel14900K : public Config {
    ConfigIntel14900K(const std::string &llm, bool skip_l1_hit) : Config(llm) {
        high_threshold = 260;
        if (skip_l1_hit) {
            low_threshold = 80;
        }
    }
};


extern const Config *g_config;
static forced_inline const Config& cfg() { return *g_config; }

void load_config(const std::string &model, const std::string &llm, bool skip_l1_hit);

#endif // CONFIG_DATABASE_HH_
