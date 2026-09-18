#ifndef GGUF_HH_
#define GGUF_HH_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
  #include <atomic>
  using std::atomic_int;
  using std::memory_order;
  using std::memory_order_acquire;
#else /* not __cplusplus */
#if defined(_WIN32)
#    include "atomic_windows.h"
#else
#    include <stdatomic.h>
#endif
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_API 

#define GGML_MAX_DIMS           4
#define GGML_MAX_OP_PARAMS      64
#define GGML_MAX_SRC            6

#define GGUF_MAGIC "GGUF"
#define GGUF_POWERINFER_MAGIC "PWRI"

#define GGUF_VERSION 3

#define GGML_MAX_NAME           64

#define GGUF_DEFAULT_ALIGNMENT 32

#if UINTPTR_MAX == 0xFFFFFFFF
    #define GGML_MEM_ALIGN 4
#else
    #define GGML_MEM_ALIGN 16
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#define GGML_ALIGNED_MALLOC(size) _aligned_malloc(size, GGML_MEM_ALIGN)
#define GGML_ALIGNED_FREE(ptr)    _aligned_free(ptr)
#else
inline static void * ggml_aligned_malloc(size_t size) {
    if (size == 0) {
        printf("WARNING: Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        return NULL;
    }
    void * aligned_memory = NULL;
#ifdef GGML_USE_CPU_HBM
    int result = hbw_posix_memalign(&aligned_memory, 16, size);
#elif GGML_USE_METAL
    int result = posix_memalign(&aligned_memory, sysconf(_SC_PAGESIZE), size);
#else
    int result = posix_memalign(&aligned_memory, GGML_MEM_ALIGN, size);
#endif
    if (result != 0) {
        // Handle allocation failure
        const char *error_desc = "unknown allocation error";
        switch (result) {
            case EINVAL:
                error_desc = "invalid alignment value";
                break;
            case ENOMEM:
                error_desc = "insufficient memory";
                break;
        }
        printf("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        return NULL;
    }
    return aligned_memory;
}
#define GGML_ALIGNED_MALLOC(size) ggml_aligned_malloc(size)
#ifdef GGML_USE_CPU_HBM
#define GGML_ALIGNED_FREE(ptr)    if(NULL != ptr) hbw_free(ptr)
#else
#define GGML_ALIGNED_FREE(ptr)    free(ptr)
#endif
#endif

#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define GGML_ASSERT assert

enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 (5) support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    // k-quantizations
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
#if 0
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
#else
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_Q4_0_4_4 = 31,
    GGML_TYPE_Q4_0_4_8 = 32,
    GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_I2_S    = 36,
    GGML_TYPE_I8_S    = 37,
    GGML_TYPE_TL1      = 38,
    GGML_TYPE_TL2      = 39,
    GGML_TYPE_COUNT,
#endif
};

enum ggml_backend_type {
    GGML_BACKEND_CPU = 0,
    GGML_BACKEND_GPU = 10,
    GGML_BACKEND_GPU_SPLIT = 20,
};

enum ggml_sparse_deriv {
    GGML_DENSE_INFERENCE = 0,
    GGML_SPARSE_INFERENCE = 1,
};

struct ggml_init_params {
    // memory pool
    size_t mem_size;   // bytes
    void * mem_buffer; // if NULL, memory will be allocated internally
    bool   no_alloc;   // don't allocate memory for the tensor data
};


// available tensor operations:
enum ggml_op {
    GGML_OP_NONE = 0,

    GGML_OP_DUP,
    GGML_OP_ADD,
    GGML_OP_ADD1,
    GGML_OP_ACC,
    GGML_OP_SUB,
    GGML_OP_MUL,
    GGML_OP_DIV,
    GGML_OP_SQR,
    GGML_OP_SQRT,
    GGML_OP_LOG,
    GGML_OP_SUM,
    GGML_OP_SUM_ROWS,
    GGML_OP_MEAN,
    GGML_OP_ARGMAX,
    GGML_OP_REPEAT,
    GGML_OP_REPEAT_BACK,
    GGML_OP_CONCAT,
    GGML_OP_SILU_BACK,
    GGML_OP_NORM, // normalize
    GGML_OP_RMS_NORM,
    GGML_OP_RMS_NORM_BACK,
    GGML_OP_GROUP_NORM,

    GGML_OP_MUL_MAT,
    GGML_OP_AXPY,
    GGML_OP_OUT_PROD,

    GGML_OP_SCALE,
    GGML_OP_SET,
    GGML_OP_CPY,
    GGML_OP_CONT,
    GGML_OP_RESHAPE,
    GGML_OP_VIEW,
    GGML_OP_PERMUTE,
    GGML_OP_TRANSPOSE,
    GGML_OP_GET_ROWS,
    GGML_OP_GET_ROWS_BACK,
    GGML_OP_DIAG,
    GGML_OP_DIAG_MASK_INF,
    GGML_OP_DIAG_MASK_ZERO,
    GGML_OP_SOFT_MAX,
    GGML_OP_SOFT_MAX_BACK,
    GGML_OP_ROPE,
    GGML_OP_ROPE_BACK,
    GGML_OP_ALIBI,
    GGML_OP_CLAMP,
    GGML_OP_CONV_TRANSPOSE_1D,
    GGML_OP_IM2COL,
    GGML_OP_CONV_TRANSPOSE_2D,
    GGML_OP_POOL_1D,
    GGML_OP_POOL_2D,

    GGML_OP_UPSCALE, // nearest interpolate

    GGML_OP_FLASH_ATTN,
    GGML_OP_FLASH_FF,
    GGML_OP_FLASH_ATTN_BACK,
    GGML_OP_WIN_PART,
    GGML_OP_WIN_UNPART,
    GGML_OP_GET_REL_POS,
    GGML_OP_ADD_REL_POS,

    GGML_OP_UNARY,

    GGML_OP_MAP_UNARY,
    GGML_OP_MAP_BINARY,

    GGML_OP_MAP_CUSTOM1_F32,
    GGML_OP_MAP_CUSTOM2_F32,
    GGML_OP_MAP_CUSTOM3_F32,

    GGML_OP_MAP_CUSTOM1,
    GGML_OP_MAP_CUSTOM2,
    GGML_OP_MAP_CUSTOM3,

    GGML_OP_CROSS_ENTROPY_LOSS,
    GGML_OP_CROSS_ENTROPY_LOSS_BACK,

    GGML_OP_COUNT,
};

// n-dimensional tensor
struct ggml_tensor {
    enum ggml_type         type;
    enum ggml_backend_type backend;

    struct ggml_backend_buffer * buffer;

    int     n_dims;
    int64_t ne[GGML_MAX_DIMS]; // number of elements
    size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                // nb[0] = ggml_type_size(type)
                                // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                // nb[i] = nb[i-1] * ne[i-1]

    // compute data
    enum ggml_op op;

    // op params - allocated as int32_t for alignment
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

    bool is_param;

    struct ggml_tensor * grad;
    struct ggml_tensor * src[GGML_MAX_SRC];

    // performance
    atomic_int is_finish;
    int     perf_runs;
    int64_t perf_cycles;
    int64_t perf_time_us;

    struct ggml_tensor * view_src;
    size_t               view_offs;

    void * data;

    char name[GGML_MAX_NAME];

    void * extra; // extra things e.g. for ggml-cuda.cu

    //char padding[12];
    uint32_t node_id;
    char padding[8];
};

#if defined(__ARM_NEON) && defined(__CUDACC__)
    typedef half ggml_fp16_t;
#elif defined(__ARM_NEON)
    typedef __fp16 ggml_fp16_t;
#else
    typedef uint16_t ggml_fp16_t;
#endif

#ifdef  __cplusplus
// restrict not standard in C++
#define GGML_RESTRICT
#else
#define GGML_RESTRICT restrict
#endif

/////////////////////////////////////////////////////

#define QK4_0 32
typedef struct {
    ggml_fp16_t d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    ggml_fp16_t d;          // delta
    ggml_fp16_t m;          // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_fp16_t) + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
typedef struct {
    ggml_fp16_t d;         // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct {
    ggml_fp16_t d;         // delta
    ggml_fp16_t m;         // min
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
typedef struct {
    ggml_fp16_t d;         // delta
    int8_t  qs[QK8_0];     // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
typedef struct {
    float d;               // delta
    float s;               // d * sum(qs[i])
    int8_t  qs[QK8_1];     // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(float) + QK8_1, "wrong q8_1 block size/padding");

//
// Super-block quantization structures
//

// Super-block size
#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.5625 bits per weight
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    ggml_fp16_t d;           // super-block scale for quantized scales
    ggml_fp16_t dmin;        // super-block scale for quantized mins
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_fp16_t) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 3.4375 bits per weight
#ifdef GGML_QKK_64
typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
    uint8_t scales[2];
    ggml_fp16_t d;             // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + 2, "wrong q3_K block size/padding");
#else
typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
    uint8_t scales[12];        // scales, quantized with 6 bits
    ggml_fp16_t d;             // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");
#endif

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
#ifdef GGML_QKK_64
typedef struct {
    ggml_fp16_t d[2];          // super-block scales/mins
    uint8_t scales[2];         // 4-bit block scales/mins
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + QK_K/2 + 2, "wrong q4_K block size/padding");
#else
typedef struct {
    ggml_fp16_t d;             // super-block scale for quantized scales
    ggml_fp16_t dmin;          // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");
#endif

// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
#ifdef GGML_QKK_64
typedef struct {
    ggml_fp16_t d;               // super-block scale
    int8_t  scales[QK_K/16];     // 8-bit block scales
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == sizeof(ggml_fp16_t) + QK_K/2 + QK_K/8 + QK_K/16, "wrong q5_K block size/padding");
#else
typedef struct {
    ggml_fp16_t d;               // super-block scale for quantized scales
    ggml_fp16_t dmin;            // super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE];   // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];          // quants, high bit
    uint8_t qs[QK_K/2];          // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");
#endif

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_fp16_t d;           // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_fp16_t) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");

// This is only used for intermediate quantization and dot products
typedef struct {/////////////////////////////////////////////////////

    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");


#if 1 // BitNet
typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;
typedef struct { uint16_t bits; } ggml_bf16_t;

// (Almost) "true" 2-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 2.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == sizeof(ggml_half) + QK_K/8*sizeof(uint16_t), "wrong iq2_xxs block size/padding");

// 2.3125 bpw quants
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
} block_iq2_xs;
static_assert(sizeof(block_iq2_xs) == sizeof(ggml_half) + QK_K/8*sizeof(uint16_t) + QK_K/32, "wrong iq2_xs block size/padding");

// 2.5625 bpw quants
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t scales[QK_K/32];
} block_iq2_s;
static_assert(sizeof(block_iq2_s) == sizeof(ggml_half) + QK_K/4 + QK_K/16, "wrong iq2_s block size/padding");

// (Almost) "true" 3-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 3.0625 bpw because of the 16-bit scale for each block of 256.
typedef struct {
    ggml_half d;
    uint8_t qs[3*QK_K/8];
} block_iq3_xxs;
static_assert(sizeof(block_iq3_xxs) == sizeof(ggml_half) + 3*(QK_K/8), "wrong iq3_xxs block size/padding");

// 3.4375 bpw
#define IQ3S_N_SCALE QK_K/64
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
static_assert(sizeof(block_iq3_s) == sizeof(ggml_half) + 13*(QK_K/32) + IQ3S_N_SCALE, "wrong iq3_s block size/padding");

// 1.5625 bpw
typedef struct {
    ggml_half d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
} block_iq1_s;
static_assert(sizeof(block_iq1_s) == sizeof(ggml_half) + QK_K/8 + QK_K/16, "wrong iq1_s block size/padding");

// 1.75 bpw
typedef struct {
    uint8_t  qs[QK_K/8];      // grid index, low 8 bits
    uint8_t  qh[QK_K/16];     // grid index, high 3 bits + grid shift bit (for two groups of 8)
    uint8_t  scales[QK_K/32]; // 3-bit block scales (4-bit if QK_K == 64)
} block_iq1_m;
static_assert(sizeof(block_iq1_m) == QK_K/8 + QK_K/16 + QK_K/32, "wrong iq1_m block size/padding");

// Used by IQ1_M quants
typedef union {
    ggml_half f16;
    uint16_t  u16;
} iq1m_scale_t;

// Non-linear quants
#define QK4_NL 32
typedef struct {
    ggml_half d;
    uint8_t qs[QK4_NL/2];
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL/2, "wrong iq4_nl block size/padding");

typedef struct {
    ggml_half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;
static_assert(sizeof(block_iq4_xs) == sizeof(ggml_half) + sizeof(uint16_t) + QK_K/64 + QK_K/2, "wrong iq4_xs block size/padding");

//
// Ternary quantization
//

// 1.6875 bpw
typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; // 5 elements per byte (3^5 = 243 < 256)
    uint8_t qh[QK_K/64]; // 4 elements per byte
    ggml_half d;
} block_tq1_0;
static_assert(sizeof(block_tq1_0) == sizeof(ggml_half) + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5, "wrong tq1_0 block size/padding");

// 2.0625 bpw
typedef struct {
    uint8_t qs[QK_K/4]; // 2 bits per element
    ggml_half d;
} block_tq2_0;
static_assert(sizeof(block_tq2_0) == sizeof(ggml_half) + QK_K / 4, "wrong tq2_0 block size/padding");

#endif

/////////////////////////////////////////////////////


//
// gguf
//

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,       // marks the end of the enum
};

struct gguf_str {
    uint64_t n;  // GGUFv2
    char * data;
};

union gguf_value {
    uint8_t  uint8;
    int8_t   int8;
    uint16_t uint16;
    int16_t  int16;
    uint32_t uint32;
    int32_t  int32;
    float    float32;
    uint64_t uint64;
    int64_t  int64;
    double   float64;
    bool     bool_;

    struct gguf_str str;

    struct {
        enum gguf_type type;

        uint64_t n;  // GGUFv2
        void * data;
    } arr;
};

struct gguf_kv {
    struct gguf_str key;

    enum  gguf_type  type;
    union gguf_value value;
};

struct gguf_header {
    char magic[4];
    uint32_t version;
    uint64_t n_tensors; // GGUFv2
    uint64_t n_kv;      // GGUFv2
};

struct gguf_tensor_info {
    struct gguf_str name;

    uint32_t n_dims;
    uint64_t ne[GGML_MAX_DIMS];

    enum ggml_type type;

    uint64_t offset; // offset from start of `data`, must be a multiple of `ALIGNMENT`

    // for writing API
    const void * data;
    size_t size;
};

struct gguf_context {
    struct gguf_header header;
    enum ggml_sparse_deriv sparse_deriv;

    struct gguf_kv          * kv;
    struct gguf_tensor_info * infos;

    size_t alignment;
    size_t offset;    // offset of `data` from beginning of file
    size_t size;      // size of `data` in bytes

    //uint8_t * padding;
    void * data;
};

struct gguf_init_params {
    bool no_alloc;
#if 0
    // if not NULL, create a ggml_context and allocate the tensor data in it
    struct ggml_context ** ctx;
#endif
};

GGML_API struct gguf_context * gguf_init_empty(void);
GGML_API struct gguf_context * gguf_init_empty_sparse(void);
GGML_API struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);
//GGML_API struct gguf_context * gguf_init_from_buffer(..);

GGML_API void gguf_free(struct gguf_context * ctx);

GGML_API const char * gguf_type_name(enum gguf_type type);

GGML_API int    gguf_get_version    (const struct gguf_context * ctx);
GGML_API size_t gguf_get_alignment  (const struct gguf_context * ctx);
GGML_API size_t gguf_get_data_offset(const struct gguf_context * ctx);
GGML_API void * gguf_get_data       (const struct gguf_context * ctx);

GGML_API int          gguf_get_n_kv(const struct gguf_context * ctx);
GGML_API int          gguf_find_key(const struct gguf_context * ctx, const char * key);
GGML_API const char * gguf_get_key (const struct gguf_context * ctx, int key_id);

GGML_API enum gguf_type gguf_get_kv_type (const struct gguf_context * ctx, int key_id);
GGML_API enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int key_id);

// will abort if the wrong type is used for the key
GGML_API uint8_t      gguf_get_val_u8  (const struct gguf_context * ctx, int key_id);
GGML_API int8_t       gguf_get_val_i8  (const struct gguf_context * ctx, int key_id);
GGML_API uint16_t     gguf_get_val_u16 (const struct gguf_context * ctx, int key_id);
GGML_API int16_t      gguf_get_val_i16 (const struct gguf_context * ctx, int key_id);
GGML_API uint32_t     gguf_get_val_u32 (const struct gguf_context * ctx, int key_id);
GGML_API int32_t      gguf_get_val_i32 (const struct gguf_context * ctx, int key_id);
GGML_API float        gguf_get_val_f32 (const struct gguf_context * ctx, int key_id);
GGML_API uint64_t     gguf_get_val_u64 (const struct gguf_context * ctx, int key_id);
GGML_API int64_t      gguf_get_val_i64 (const struct gguf_context * ctx, int key_id);
GGML_API double       gguf_get_val_f64 (const struct gguf_context * ctx, int key_id);
GGML_API bool         gguf_get_val_bool(const struct gguf_context * ctx, int key_id);
GGML_API const char * gguf_get_val_str (const struct gguf_context * ctx, int key_id);
GGML_API int          gguf_get_arr_n   (const struct gguf_context * ctx, int key_id);
GGML_API const void * gguf_get_arr_data(const struct gguf_context * ctx, int key_id);
GGML_API const char * gguf_get_arr_str (const struct gguf_context * ctx, int key_id, int i);

GGML_API enum ggml_sparse_deriv gguf_get_sparse_deriv(const struct gguf_context * ctx);
GGML_API int    gguf_get_n_tensors    (const struct gguf_context * ctx);
GGML_API int    gguf_find_tensor      (const struct gguf_context * ctx, const char * name);
GGML_API size_t gguf_get_tensor_offset(const struct gguf_context * ctx, int i);
GGML_API char * gguf_get_tensor_name  (const struct gguf_context * ctx, int i);

// overrides existing values or adds a new one
GGML_API void gguf_set_val_u8  (struct gguf_context * ctx, const char * key, uint8_t  val);
GGML_API void gguf_set_val_i8  (struct gguf_context * ctx, const char * key, int8_t   val);
GGML_API void gguf_set_val_u16 (struct gguf_context * ctx, const char * key, uint16_t val);
GGML_API void gguf_set_val_i16 (struct gguf_context * ctx, const char * key, int16_t  val);
GGML_API void gguf_set_val_u32 (struct gguf_context * ctx, const char * key, uint32_t val);
GGML_API void gguf_set_val_i32 (struct gguf_context * ctx, const char * key, int32_t  val);
GGML_API void gguf_set_val_f32 (struct gguf_context * ctx, const char * key, float    val);
GGML_API void gguf_set_val_u64 (struct gguf_context * ctx, const char * key, uint64_t val);
GGML_API void gguf_set_val_i64 (struct gguf_context * ctx, const char * key, int64_t  val);
GGML_API void gguf_set_val_f64 (struct gguf_context * ctx, const char * key, double   val);
GGML_API void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool     val);
GGML_API void gguf_set_val_str (struct gguf_context * ctx, const char * key, const char * val);
GGML_API void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n);
GGML_API void gguf_set_arr_str (struct gguf_context * ctx, const char * key, const char ** data, int n);

// set or add KV pairs from another context
GGML_API void gguf_set_kv(struct gguf_context * ctx, struct gguf_context * src);

// manage tensor info
GGML_API void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor);
GGML_API void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type);
GGML_API void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data, size_t size);

// writing gguf files can be done in 2 ways:
//
// - write the entire gguf_context to a binary file in a single pass:
//
//   gguf_write_to_file(ctx, fname);
//
// - first prepare a file with a placeholder for the meta data, write the tensor data, then write the meta data:
//
//   FILE * f = fopen(fname, "wb");
//   fseek(f, gguf_get_meta_size(ctx), SEEK_SET);
//   fwrite(f, ...);
//   void * data = gguf_meta_get_meta_data(ctx);
//   fseek(f, 0, SEEK_SET);
//   fwrite(f, data, gguf_get_meta_size(ctx));
//   free(data);
//   fclose(f);
//

// write the entire context to a binary file
GGML_API void gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta);

// get the size in bytes of the meta data (header, kv pairs, tensor info) including padding
GGML_API size_t gguf_get_meta_size(const struct gguf_context * ctx);
GGML_API void   gguf_get_meta_data(const struct gguf_context * ctx, void * data);

GGML_API int ggml_blck_size(enum ggml_type type);
GGML_API size_t ggml_type_size(enum ggml_type type);
GGML_API const char *get_type_name(enum ggml_type type);

#ifdef  __cplusplus
}
#endif

#endif // GGUF_HH_