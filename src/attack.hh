#ifndef ATTACK_HH_
#define ATTACK_HH_

#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <memory>
#include <functional>
#include "util.h"
#include "tokenizer.hh"

#include "async-ringbuffer.hh"

namespace safetensors {
    struct safetensors_t;
}

#define EMBED_N_DIMS 2

class Attacker {
    class Tensor {
    public:
        uint8_t *ptr;
        unsigned nb[EMBED_N_DIMS];
        size_t nbytes;

        int init(struct gguf_context *ctx_gguf, uint8_t *tgt_image, const char *name);
        inline size_t get_nbytes() const {
            return nbytes;
        }
    };
public:
    class safetensors::safetensors_t *st;
    const uint8_t *embed_ptr;
    size_t embed_nb[EMBED_N_DIMS];
    Tensor q_ts, k_ts, v_ts;
    llm_arch arch;
    unsigned num_vocab, n_embd;
    llama_vocab vocab;
    std::vector<unsigned> candiate_tokens;
    uint8_t *trigger_func_ptr;
    llama_mlock mem_lock;
    struct CacheTraceElement {
        size_t row;
        uint64_t latency, timestamp;
    };
    std::vector<CacheTraceElement> detected;

    struct partition {
        std::atomic<bool> running;
        AsyncRingBuffer<CacheTraceElement> cache_trace{1000000};
        std::thread *td;
    };
    std::vector<partition *> pool;

public:
    ~Attacker();
    int init(const std::string &weight_file, const std::string &model_type);
    int loop(const std::function<int()> &query, int echo = 1, long query_time_us = 10000);
    int save_results(const std::string &database);
    std::vector<CacheTraceElement> get_detected();
   
};


#endif // ATTACK_HH_
