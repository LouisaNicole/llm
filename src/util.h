#ifndef UTIL_H_
#define UTIL_H_

#include <stdint.h>
#include <stddef.h>

#define VERBOSE 0

#define CACHELINE_SIZE 64
#define PAGE_SIZE 4096

#define COMB64(_lo, _hi) ((uint64_t(_hi)<<32)|(_lo))

#define forced_inline inline __attribute__((always_inline))

#ifdef __cplusplus

#include <vector>
#include <sstream>

template<typename T>
static inline T round_up(T N, size_t S) {
    return ((N + S - 1) / S) * S;
}

template<typename T>
static inline T round_down(T p, uintptr_t s) {
    return ((p / s) * s);
}

static inline intptr_t calc_stride(uintptr_t a, uintptr_t b) {
    return __int128(a) - __int128(b);
}

template<typename T>
static inline bool fsign(T x) {
    return (x>=0) ? 0 : 1;
}

class Average {
    double avg;
    size_t n;
public:
    Average() : avg(0), n(0) {}

    void sample(double val) {
        avg = avg * (double(n)/ (n+1)) + val / (n+1);
        ++n;
    }
    double operator()() const {
        return avg;
    }
};

class Frequency {
    unsigned start;
public:
    std::vector<size_t> freq;
    
    Frequency(unsigned start, unsigned end)
        : start(start),
          freq(end - start + 1)
    {}

    void sample(unsigned val) {
        freq[val]++;
    }
    void clear() {
        size_t oldsize = freq.size();
        freq.clear();
        freq.resize(oldsize);
    }
};

struct llama_mlock {
    void * addr = NULL;
    size_t size = 0;

    bool failed_already = false;

    llama_mlock() {}
    llama_mlock(const llama_mlock &) = delete;

    ~llama_mlock();

    void init(void * ptr);

    bool grow_to(size_t target_size);

    bool raw_lock(void * ptr, size_t len) const;
    void raw_unlock(void * addr, size_t size);
};

template<typename T>
static std::string tostr(const T &printable) {
    std::stringstream ss;
    ss << printable;
    return ss.str();
}

static std::vector<std::string> str_split(const std::string &str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

extern "C" {
#endif

void random_shuffle(unsigned *arr, unsigned n);
void pin_processor(unsigned core);
void set_highest_unprivileged_priority();
void* aligned_malloc(size_t required_bytes, size_t alignment);
void aligned_free(void *p2);
void set_page_executable(void *mem, size_t size, unsigned modify);
void *get_shared_memory(const char *file, unsigned long *size);
void *map_exec(const char *binfile);
void *mapview(const char *file, unsigned long *size,
                size_t view_offset, size_t view_size);
size_t get_page_size();
size_t get_phy_ram_size();
size_t phy_cores();
void yield_cpu();

struct Process;
Process *create_process(const char *argv[], unsigned argc, bool disable_io);
bool wait_process(Process *proc, int &status);
bool wait_process_blocked(Process *proc, int &status);
void free_process(Process *&proc);

struct NamedPipe;
NamedPipe *create_named_pipe();
char *get_named_pipe_filename(NamedPipe *pipe); // free()
void destory_named_pipe(NamedPipe *&pipe);

const char *get_temp_dir();

double get_tsc_freq();

#ifdef __cplusplus
}
#endif

static __inline__ uint64_t rdtsc(void) {
    unsigned int hi, lo;
    __asm__ __volatile__("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

double tsc2sec(uint64_t tsc);

#define mfence() asm volatile("mfence" : : : "memory")
#define sfence() asm volatile("sfence" : : : "memory")

static forced_inline void flush(void *p) {
    asm __volatile__ (
        "clflush 0(%0) \n"
        : : "b" (p)
        : "memory"
    );
}

static forced_inline uint64_t reload(void *p) {
    uint32_t lo, hi, lo0, hi0;
    asm __volatile__ (
        "mfence         \n"
        "rdtsc          \n"
        "movl %%eax, %%esi \n"
        "movl %%edx, %%edi \n"
        "movl (%4), %%eax		\n"
        "mfence         \n"
        "rdtsc          \n"
        "movl %%esi, %0 \n"
        "movl %%edi, %1 \n"
            : "=r" (lo0), "=r" (hi0), "=a" (lo), "=d" (hi)
            : "b" (p)
            : "%esi", "%edi", "memory");
    return (COMB64(lo, hi) - COMB64(lo0, hi0));
}

static forced_inline uint64_t flushed_reload(void *p) {
    uint32_t lo, hi, lo0, hi0;
#if 0
    asm __volatile__ (
        "mfence         \n"
        "rdtsc          \n"
        "movl %%eax, %%esi \n"
        "movl %%edx, %%edi \n"
        "movl (%4), %%eax		\n"
        "mfence         \n"
        "rdtsc          \n"
        "clflush 0(%4)  \n"
        "movl %%esi, %0 \n"
        "movl %%edi, %1 \n"
            : "=r" (lo0), "=r" (hi0), "=a" (lo), "=d" (hi)
            : "b" (p)
            : "%esi", "%edi", "memory");
#endif
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

    return (COMB64(lo, hi) - COMB64(lo0, hi0));
}

static forced_inline uint64_t timed_flush(void *p) {
    uint32_t lo, hi, lo0, hi0;
    asm __volatile__ (
        "mfence         \n"
        "rdtsc          \n"
        "movl %%eax, %%esi \n"
        "movl %%edx, %%edi \n"
        "clflush 0(%4) \n"
        "mfence         \n"
        "rdtsc          \n"
        "movl %%esi, %0 \n"
        "movl %%edi, %1 \n"
            : "=r" (lo0), "=r" (hi0), "=a" (lo), "=d" (hi)
            : "b" (p)
            : "%esi", "%edi", "memory");
    return (COMB64(lo, hi) - COMB64(lo0, hi0));
}

static forced_inline void maccess(void *p) {
    *(volatile uint8_t *)p;
}

#endif