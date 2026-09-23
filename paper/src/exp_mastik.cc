#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <memory>
#include <atomic>
#include <random>
#include <algorithm>
#include <thread>
#include <cassert>
#include <optional>
#include <map>

#include <unistd.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/types.h>
//#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
//#include <spawn.h>

#include <semaphore.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "util.h"

extern "C" {
#include "thirdparty/mastik-0.02/src/fr.h"
}

#include "config-database.hh"

#define PROB_ACCESS 0.5
#define VICTIM_PROC_CORE 0
#define ATTACKER_PROC_CORE 2
#define MAX_LATENCY uint64_t(400)
#define SPIN


static std::optional<std::string> llm_model,
                                hardware_model;
static std::optional<size_t> num_targets;
static std::optional<unsigned> num_rounds;

#ifdef SPIN
static std::atomic<bool> victim_start, victim_done;
#else
sem_t victim_start, victim_done;
#endif
static std::atomic<bool> flag_end;
static std::atomic<size_t> victim_accessed;

using fn_calc_target = std::function<uint8_t *(size_t i)>;

static unsigned embed_nb[2];
static uint8_t *embed_ptr;
static unsigned nb1;
int init(const std::string &llm_model, size_t &n_targets, fn_calc_target& calc_target) {
    size_t n_vocab, n_embd;
    size_t embed_tab_size;

    if (!llm_model.compare("llama-2-7b-chat")) {
        n_vocab=32000;
        n_embd=512;//4096;
        embed_nb[0]=2;
        embed_nb[1]=embed_nb[0]*n_embd;
        embed_tab_size = embed_nb[1]*n_vocab;
    } else {
        printf("invalid model '%s'\n", llm_model.c_str());
        return 1;
    }
    embed_ptr = (uint8_t*)aligned_malloc(embed_tab_size, PAGE_SIZE);
    for (size_t i = 0; i < embed_tab_size; ++i)
        embed_ptr[i] = rand(); // padding non-zero values to avoid sharing the zero page
                                // and to commit the virtual pages
                                // and to enable masked va
    n_targets = n_vocab;
    nb1 = embed_nb[1];
    calc_target = [](size_t i) -> uint8_t * {
        return embed_ptr + i*nb1 + nb1/2;
    };
    assert((uintptr_t(embed_ptr) & 4095) == 0);

#ifndef SPIN
    sem_init(&victim_start, 0, 0);
    sem_init(&victim_done, 0, 0);
#endif
    return 0;
}

static void victim_loop(size_t n_targets, const fn_calc_target &cal_target) {
    pin_processor(VICTIM_PROC_CORE);
    set_highest_unprivileged_priority();

    static std::mt19937 rng(124);
    std::cout << "victim_loop: randomly access to " << n_targets << " targets" << std::endl;
    std::uniform_int_distribution<size_t> idist(0,n_targets-1);

    double *dst = new double[embed_nb[1]];

    volatile float v=100;
    bool run = true;
    do {
#ifdef SPIN
        while(!victim_start.load())
            ;
        victim_start.store(false);
#else
        sem_wait(&victim_start);
#endif

        if (flag_end.load())
            run = false;
        else {
            auto vocab = idist(rng);
            // computing the embedding vector
            for (unsigned i=0;i<embed_nb[1]/sizeof(float);++i) {
                auto f16 = *(volatile float *)((embed_ptr + vocab*nb1 + nb1/2)) + i;
                dst[i] = double(f16);
            }
            mfence();
            victim_accessed.store(vocab);
        }

#ifdef SPIN
        victim_done.store(true);
#else
        sem_post(&victim_done);
#endif
    } while(run);
}

static void invoke_victim() {
#ifdef SPIN
    victim_start.store(true);
    while(!victim_done.load())
        ;
    victim_done.store(false);
#else
    sem_post(&victim_start);
    sem_wait(&victim_done);
#endif
}

static void restart_victim() {
    flag_end.store(false);
#ifdef SPIN
    victim_start.store(false);
    victim_done.store(false);
#else
#endif
}

static void end_victim(std::thread *& victim) {
    flag_end.store(true);
    invoke_victim();
    victim->join();
    delete victim; victim=nullptr;
}

static void yield() {
    sleep(1); // make a context switch
}

void test(const std::vector<uint8_t *> &targets) {
    static std::mt19937 rng;
    std::uniform_int_distribution<size_t> idist(0,targets.size()-1);
    victim_accessed = idist(rng);
    *(volatile uint8_t *)targets[victim_accessed];
}

#define FOOLPF_INIT(policy_name) foolpf_ ## policy_name ## _init()
#define FOOLPF_GET(policy_name) foolpf_ ## policy_name ## _get()
#define FOOLPF_IS_ACCESSED(policy_name) foolpf_ ## policy_name ## _is_accessed()

#define foolpf_none_init()
#define foolpf_none_get() \
    calc_target(i)
#define foolpf_none_is_accessed() \
    [&](size_t victim_accessed, size_t i) -> bool { \
        return victim_accessed == i; \
    }

#define ATTK_TEMPLATE(_name, _sname, _foolpf_policy_name, _init, _phase1, _phase2, _phase3) \
static void _name(size_t n_targets, const fn_calc_target &calc_target \
                    ) { \
    _init ; \
    FOOLPF_INIT(_foolpf_policy_name) ; \
\
    std::unordered_map<size_t,uint64_t> dead_set; \
    size_t nr_fp = 0, nr_tp = 0, nr_fn = 0, nr_tn = 0; \
    Frequency nac(0, MAX_LATENCY), ac(0, MAX_LATENCY); \
\
    auto victim = new std::thread(victim_loop, n_targets, calc_target); \
    restart_victim(); \
\
    clock_t t0 = clock(); \
    for(size_t i=0;i<*num_rounds;++i) { \
\
        _phase1 ; \
\
        invoke_victim(); \
        _phase2 ; \
\
        uint64_t latency; \
        for(size_t i = 0; i < n_targets; ++i) { \
            _phase3 ; \
            /*bool pred = (latency < calib->get_llc_hit_latency() && latency > calib->get_l1_hit_latency());*/ \
            bool pred = (/*latency > 110 &&*/ latency < 170); \
            if (FOOLPF_IS_ACCESSED(_foolpf_policy_name)(victim_accessed, i)) { \
                if (pred) { \
                    nr_tp++; \
                } else { \
                    nr_fn++; \
                } \
                ac.sample(std::min(latency, MAX_LATENCY)); \
            } else { \
                if (pred) { \
                    nr_fp++; \
                } else { \
                    nr_tn++; \
                } \
                nac.sample(std::min(latency, MAX_LATENCY));  \
            } \
            if (latency < 90) { \
                dead_set[i]++; \
            } \
        } \
    } \
    clock_t tm = clock() - t0; \
    assert(tm > CLOCKS_PER_SEC*0.01); /* check if iterations is too less \
                                            to match the time precision */ \
    double total_time = double(tm) / CLOCKS_PER_SEC; \
\
    /* validate the result */ \
    assert(nr_tp + nr_fn == *num_rounds); \
    assert(nr_tn + nr_fp == *num_rounds * (n_targets-1)); \
\
    double recall = double(nr_tp)/(nr_tp+nr_fn); \
    double precision = double(nr_tp)/(nr_tp+nr_fp); \
    double fpps = double(nr_fp)/total_time;\
    double pps = double(n_targets)/total_time*(*num_rounds); \
    std::cout << "recall " << recall*100 << "% " \
              << "precision " << precision*100 << "% " \
              << "fpps " << fpps << "/s " \
              << "pps " << pps << "/s" << std::endl; \
    end_victim(victim); \
    std::cout << "done" << std::endl; \
}


#define FR_FF_init()

#define foolpf_flush_reload_FUNNAME(base,foolpf_policy) base ## _ ## foolpf_policy

#define FR_FF_init_1() FR_FF_init()

#define DECL_FOOLPF_flush_reload(foolpf_policy_name) \
    ATTK_TEMPLATE(foolpf_flush_reload_FUNNAME(flush_reload, foolpf_policy_name), "fr", \
        foolpf_policy_name, \
        FR_FF_init_1(), \
        {for(size_t i = 0; i < n_targets; ++i) { \
            flush(calc_target(i)); \
        }}, \
        {}, \
        { \
        latency = reload(FOOLPF_GET(foolpf_policy_name)); \
        flush(FOOLPF_GET(foolpf_policy_name)); }) \

/*
 * Fool prefetcher policy 1: shuffled address array
 */
#define foolpf_shuffled_addr_array_init() \
    std::vector<size_t> indices(n_targets); \
    for(size_t i = 0; i < n_targets; ++i) { \
        indices[i] = i; \
    } \
    std::random_shuffle(indices.begin(), indices.end()); \
    uint8_t * volatile *shuffled_addr = new uint8_t *[n_targets]; \
    for(size_t i = 0; i < n_targets; ++i) { \
        shuffled_addr[i] = calc_target(indices[i]); \
    }

#define foolpf_shuffled_addr_array_get() \
    shuffled_addr[i]

#define foolpf_shuffled_addr_array_is_accessed() \
    [&](size_t victim_accessed, size_t i) -> bool { \
        return victim_accessed == indices[i]; \
    }

/*
 * Fool prefetcher policy 2: shuffled indice addr array
 */
#define foolpf_shuffled_indice_addr_array_init() \
    std::vector<uint8_t *> addr_ptrs(n_targets); \
    for(size_t i = 0; i < n_targets; ++i) { \
        addr_ptrs[i] = calc_target(i); \
    } \
    std::vector<size_t> indices(n_targets); \
    for(size_t i = 0; i < n_targets; ++i) { \
        indices[i] = i; \
    } \
    std::random_shuffle(indices.begin(), indices.end());

#define foolpf_shuffled_indice_addr_array_get() \
    addr_ptrs[indices[i]]

#define foolpf_shuffled_indice_addr_array_is_accessed() \
    [&](size_t victim_accessed, size_t i) -> bool { \
        return victim_accessed == indices[i]; \
    }

/*
 * Fool prefetcher policy 3: shuffled indices
 */
#define foolpf_shuffled_indices_init() \
    std::vector<size_t> indices(n_targets); \
    for(size_t i = 0; i < n_targets; ++i) { \
        indices[i] = i; \
    } \
    std::random_shuffle(indices.begin(), indices.end());

#define foolpf_shuffled_indices_get() \
    calc_target(indices[i])

#define foolpf_shuffled_indices_is_accessed() \
    [&](size_t victim_accessed, size_t i) -> bool { \
        return victim_accessed == indices[i]; \
    }

/*
 * Fool prefetcher policy 4: permutation
 */
#define foolpf_permutation_init()

static inline size_t get_permutation(const fn_calc_target &calc_target,
                                        size_t i, size_t n) {
    return (i * 167 + 13) % n;
}

#define foolpf_permutation_get() \
    calc_target(get_permutation(calc_target, i, n_targets))

#define foolpf_permutation_is_accessed() \
    [&](size_t victim_accessed, size_t i) -> bool { \
        return victim_accessed == get_permutation(calc_target, i, n_targets); \
    }

DECL_FOOLPF_flush_reload(none)
DECL_FOOLPF_flush_reload(shuffled_addr_array)
DECL_FOOLPF_flush_reload(shuffled_indice_addr_array)
DECL_FOOLPF_flush_reload(shuffled_indices)
DECL_FOOLPF_flush_reload(permutation)

static void call_mastki(size_t n_targets, const fn_calc_target &calc_target, unsigned n_experiment) {
  auto fr = fr_prepare();

  auto *trace = new uint16_t[n_targets];

  std::vector<size_t> indices(n_targets); 
  for(size_t i = 0; i < n_targets; ++i) {
    // fool the prefetchers via pseudo random permutation
      indices[i] = get_permutation(calc_target, i, n_targets); 
  }
  for(size_t i = 0; i < n_targets; ++i) { 
      fr_monitor(fr, calc_target(indices[i]));
  }

  // flush all
  fr_probe(fr, trace);

  size_t nr_fp = 0, nr_tp = 0, nr_fn = 0, nr_tn = 0; 
  Frequency nac(0, MAX_LATENCY), ac(0, MAX_LATENCY); 

  auto victim = new std::thread(victim_loop, n_targets, calc_target); 
  restart_victim(); 

  clock_t t0 = clock(); 
  for(size_t i=0;i<*num_rounds;++i) { 
      invoke_victim(); 

      fr_probe(fr, trace);

      uint64_t latency; 
      for(size_t i = 0; i < n_targets; ++i) { 
          uint64_t latency = trace[i];
          bool pred = (latency < 170); 
          if (victim_accessed == indices[i]) { 
              if (pred) { 
                  nr_tp++; 
              } else { 
                  nr_fn++; 
              } 
              ac.sample(std::min(latency, MAX_LATENCY)); 
          } else { 
              if (pred) { 
                  nr_fp++; 
              } else { 
                  nr_tn++; 
              } 
              nac.sample(std::min(latency, MAX_LATENCY));  
          } 
      } 
  }
  clock_t tm = clock() - t0; 
  assert(tm > CLOCKS_PER_SEC*0.01); /* check if iterations is too less 
                                          to match the time precision */ 
  double total_time = double(tm) / CLOCKS_PER_SEC; 

  /* validate the result */ 
  assert(nr_tp + nr_fn == *num_rounds); 
  assert(nr_tn + nr_fp == *num_rounds * (n_targets-1)); 

  double recall = double(nr_tp)/(nr_tp+nr_fn); 
  double precision = double(nr_tp)/(nr_tp+nr_fp); 
  double fpps = double(nr_fp)/total_time;
  double pps = double(n_targets)/total_time*(*num_rounds); 
  std::cout << "recall " << recall*100 << "% " 
            << "precision " << precision*100 << "% " 
            << "fpps " << fpps << "/s " 
            << "pps " << pps << "/s" << std::endl; 
  end_victim(victim); 
  std::cout << "done" << std::endl; 
}

static void experiment(size_t n_targets, const fn_calc_target &calc_target, unsigned n_experiment) {
    pin_processor(ATTACKER_PROC_CORE);
    set_highest_unprivileged_priority();

    std::cout << "---- F+R ----" << std::endl;
    yield();

    std::cout << (++n_experiment) << ". Mastik v0.02" << std::endl; \
    call_mastki(n_targets, calc_target, n_experiment);

    #define do_flush_reload(foolpf_policy_name) \
        std::cout << (++n_experiment) << ". " # foolpf_policy_name << std::endl; \
        foolpf_flush_reload_FUNNAME(flush_reload, foolpf_policy_name)(n_targets, calc_target \
                        ); \
        yield();

    do_flush_reload(shuffled_addr_array)
    do_flush_reload(shuffled_indice_addr_array)
    do_flush_reload(shuffled_indices)
    do_flush_reload(permutation)
}

template<typename T>
static int parseint(const char *argname, const char *arg, std::optional<T> &out) {
    char *strtol_end = nullptr;
    long long val = strtoll(arg, &strtol_end, 10);
    if (strtol_end != arg + strlen(arg)) {
        printf("invalid str %s\n", argname);
        return 1;
    }
    if (val > std::numeric_limits<T>::max() || val < std::numeric_limits<T>::min()) { // FIXME
        printf("invalid range of %s\n", argname);
        return 1;
    }
    out = val;
    return 0;
}

int main(int argc, char *argv[]) {
    for(int i=1;i<argc;++i) {
        auto advance_arg = [&]() {
            if (i==argc-1) {
                std::cerr << "missing vaule for the option '" << argv[i] << "'" << std::endl;
                exit(1);
            }
            ++i;
        };
        if(!strcmp(argv[i], "--llm")) {
            advance_arg();
            llm_model = argv[i];
        } else if(!strcmp(argv[i], "--hw")) {
            advance_arg();
            hardware_model = argv[i];
        } else if(!strcmp(argv[i], "--rounds")) {
            advance_arg();
            if (parseint("--rounds", argv[i], num_rounds)) {
                return 1;
            }
        } else if(!strcmp(argv[i], "--targets")) {
            advance_arg();
            if (parseint("--targets", argv[i], num_targets)) {
                return 1;
            }
        } else {
            printf("unknown option '%s'\n", argv[i]);
            return 1;
        }
    }

    if (!llm_model.has_value()) {
        printf("must specify --llm\n");
        return 1;
    }

    if (!hardware_model.has_value()) {
        printf("must specify --hw\n");
        return 1;
    }
    if (!num_rounds.has_value()) {
        printf("must specify --rounds\n");
        return 1;
    }
    if (!num_targets.has_value()) {
        printf("must specify --targets\n");
        return 1;
    }
    
    assert(PAGE_SIZE == get_page_size());

    fn_calc_target calc_target;
    size_t n_targets;
    if (init(*llm_model, n_targets, calc_target)) {
        return 1;
    }
    n_targets = std::min(n_targets, *num_targets);

    unsigned n_experiment = 0;
    experiment(n_targets, calc_target, n_experiment);

    return 0;
}