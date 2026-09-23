
#include <stdio.h>
#include <inttypes.h>
#include "gguf.h"

typedef void (*ggml_to_float_t)  (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int k);
typedef void (*ggml_from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int k);
typedef void (*ggml_vec_dot_t)   (const int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT x, const void * GGML_RESTRICT y);

typedef struct {
    const char      * type_name;
    int               blck_size;
    int64_t                  blck_size_interleave; // interleave elements in blocks
    size_t            type_size;
    bool              is_quantized;
    ggml_to_float_t   to_float;
    ggml_from_float_t from_float;
    ggml_from_float_t        from_float_ref;
    ggml_from_float_t from_float_reference;
    ggml_vec_dot_t    vec_dot;
    enum ggml_type    vec_dot_type;
    int64_t                  nrows; // number of rows to process simultaneously
    int64_t                  ncols; // number of columns to process simultaneously
} ggml_type_traits_t;

static const ggml_type_traits_t type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = sizeof(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = sizeof(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = sizeof(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = sizeof(float),
        .is_quantized             = false,
        .vec_dot_type             = GGML_TYPE_F32,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = sizeof(ggml_fp16_t),
        .is_quantized             = false,
        .vec_dot_type             = GGML_TYPE_F16,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_0,
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = sizeof(block_q4_1),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_1,
    },
    [4] = { // GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = GGML_TYPE_COUNT,
    },
    [5] = { // GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = GGML_TYPE_COUNT,
    },
    [GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_size                = QK5_0,
        .type_size                = sizeof(block_q5_0),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_0,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = sizeof(block_q5_1),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_1,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = sizeof(block_q8_0),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_0,
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = sizeof(block_q8_1),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_1,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q2_K),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_K,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q3_K),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_K,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q4_K),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_K,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q5_K),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_K,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q6_K),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_K,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q8_K),
        .is_quantized             = true,
    },
    
#if 1 // BitNet
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_xxs),
        .is_quantized             = true,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_xs),
        .is_quantized             = true,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq3_xxs),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq3_s),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_s),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq1_s),
        .is_quantized             = true,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq1_m),
        .is_quantized             = true,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = sizeof(block_iq4_nl),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq4_xs),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    
    [GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = sizeof(ggml_bf16_t),
        .is_quantized             = false,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0_4_4] = {
        .type_name                = "q4_0_4x4",
        .blck_size                = QK4_0,
        .blck_size_interleave     = 4,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
        .ncols                    = 4,
    },
    [GGML_TYPE_Q4_0_4_8] = {
        .type_name                = "q4_0_4x8",
        .blck_size                = QK4_0,
        .blck_size_interleave     = 8,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
        .ncols                    = 4,
    },
    [GGML_TYPE_Q4_0_8_8] = {
        .type_name                = "q4_0_8x8",
        .blck_size                = QK4_0,
        .blck_size_interleave     = 8,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
        .ncols                    = 8,
    },
    [GGML_TYPE_TQ1_0] = {
        .type_name                = "tq1_0",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_tq1_0),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_tq2_0),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I2_S] = {
        .type_name                = "i2_s",
        .blck_size                = 1,
        .type_size                = sizeof(int8_t),
        .is_quantized             = true,
        .vec_dot_type             = GGML_TYPE_I8_S,
        .nrows                    = 1,
    },
    [GGML_TYPE_I8_S] = {
        .type_name                = "i8_s",
        .blck_size                = 1,
        .type_size                = sizeof(int8_t),
        .is_quantized             = true,
    }

#endif
};

int ggml_blck_size(enum ggml_type type) {
    return type_traits[type].blck_size;
}
size_t ggml_type_size(enum ggml_type type) {
    return type_traits[type].type_size;
}
const char *get_type_name(enum ggml_type type) {
    return type_traits[type].type_name;
}

static 
size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    size_t nbytes;
    size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    else {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    return nbytes;
}

static const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8]   = sizeof(uint8_t),
    [GGUF_TYPE_INT8]    = sizeof(int8_t),
    [GGUF_TYPE_UINT16]  = sizeof(uint16_t),
    [GGUF_TYPE_INT16]   = sizeof(int16_t),
    [GGUF_TYPE_UINT32]  = sizeof(uint32_t),
    [GGUF_TYPE_INT32]   = sizeof(int32_t),
    [GGUF_TYPE_FLOAT32] = sizeof(float),
    [GGUF_TYPE_BOOL]    = sizeof(bool),
    [GGUF_TYPE_STRING]  = sizeof(struct gguf_str),
    [GGUF_TYPE_UINT64]  = sizeof(uint64_t),
    [GGUF_TYPE_INT64]   = sizeof(int64_t),
    [GGUF_TYPE_FLOAT64] = sizeof(double),
    [GGUF_TYPE_ARRAY]   = 0, // undefined
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

static const char * GGUF_TYPE_NAME[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8]   = "u8",
    [GGUF_TYPE_INT8]    = "i8",
    [GGUF_TYPE_UINT16]  = "u16",
    [GGUF_TYPE_INT16]   = "i16",
    [GGUF_TYPE_UINT32]  = "u32",
    [GGUF_TYPE_INT32]   = "i32",
    [GGUF_TYPE_FLOAT32] = "f32",
    [GGUF_TYPE_BOOL]    = "bool",
    [GGUF_TYPE_STRING]  = "str",
    [GGUF_TYPE_ARRAY]   = "arr",
    [GGUF_TYPE_UINT64]  = "u64",
    [GGUF_TYPE_INT64]   = "i64",
    [GGUF_TYPE_FLOAT64] = "f64",
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

static bool gguf_fread_el(FILE * file, void * dst, size_t size, size_t * offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

static bool gguf_fread_str(FILE * file, struct gguf_str * p, size_t * offset) {
    p->n    = 0;
    p->data = NULL;

    bool ok = true;

    ok = ok && gguf_fread_el(file, &p->n,    sizeof(p->n), offset); p->data = (char*)calloc(p->n + 1, 1);
    ok = ok && gguf_fread_el(file,  p->data, p->n,         offset);

    return ok;
}

struct gguf_context * gguf_init_empty(void) {
    struct gguf_context * ctx = GGML_ALIGNED_MALLOC(sizeof(struct gguf_context));

    memcpy(ctx->header.magic, GGUF_MAGIC, sizeof(ctx->header.magic));
    ctx->header.version   = GGUF_VERSION;
    ctx->header.n_tensors = 0;
    ctx->header.n_kv      = 0;

    ctx->kv    = NULL;
    ctx->infos = NULL;

    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;
    ctx->offset    = 0;
    ctx->size      = 0;

    ctx->data = NULL;

    return ctx;
}

struct gguf_context * gguf_init_empty_sparse(void) {
    struct gguf_context * ctx = gguf_init_empty();
    memcpy(ctx->header.magic, GGUF_POWERINFER_MAGIC, sizeof(ctx->header.magic));
    return ctx;
}

struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params) {
    FILE * file = fopen(fname, "rb");
    if (!file) {
        return NULL;
    }

    // offset from start of file
    size_t offset = 0;

    char magic[4];
    enum ggml_sparse_deriv sparse_deriv;

    // check the magic before making allocations
    {
        gguf_fread_el(file, &magic, sizeof(magic), &offset);

        if (strncmp(magic, GGUF_MAGIC, sizeof(magic)) == 0) {
            sparse_deriv = GGML_DENSE_INFERENCE;
        } else if (strncmp(magic, GGUF_POWERINFER_MAGIC, sizeof(magic)) == 0) {
            sparse_deriv = GGML_SPARSE_INFERENCE;
        } else {
            fprintf(stderr, "%s: invalid magic characters %s.\n", __func__, magic);
            fclose(file);
            return NULL;
        }
    }

    bool ok = true;

    struct gguf_context * ctx = GGML_ALIGNED_MALLOC(sizeof(struct gguf_context));
    ctx->sparse_deriv = sparse_deriv;

    // read the header
    {
        strncpy(ctx->header.magic, magic, 4);


        ctx->kv    = NULL;
        ctx->infos = NULL;
        ctx->data  = NULL;

        ok = ok && gguf_fread_el(file, &ctx->header.version,   sizeof(ctx->header.version),   &offset);
        ok = ok && gguf_fread_el(file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
        ok = ok && gguf_fread_el(file, &ctx->header.n_kv,      sizeof(ctx->header.n_kv),      &offset);

        if (ctx->header.version == 1) {
            fprintf(stderr, "%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read header\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }
    }

    // read the kv pairs
    {
        ctx->kv = malloc(ctx->header.n_kv * sizeof(struct gguf_kv));

        for (uint32_t i = 0; i < ctx->header.n_kv; ++i) {
            struct gguf_kv * kv = &ctx->kv[i];

            //fprintf(stderr, "%s: reading kv %d\n", __func__, i);

            ok = ok && gguf_fread_str(file, &kv->key,                    &offset);
            ok = ok && gguf_fread_el (file, &kv->type, sizeof(kv->type), &offset);

            //fprintf(stderr, "%s: reading kv with key %s\n", __func__, kv->key.data);

            switch (kv->type) {
                case GGUF_TYPE_UINT8:   ok = ok && gguf_fread_el (file, &kv->value.uint8,   sizeof(kv->value.uint8),   &offset); break;
                case GGUF_TYPE_INT8:    ok = ok && gguf_fread_el (file, &kv->value.int8,    sizeof(kv->value.int8),    &offset); break;
                case GGUF_TYPE_UINT16:  ok = ok && gguf_fread_el (file, &kv->value.uint16,  sizeof(kv->value.uint16),  &offset); break;
                case GGUF_TYPE_INT16:   ok = ok && gguf_fread_el (file, &kv->value.int16,   sizeof(kv->value.int16),   &offset); break;
                case GGUF_TYPE_UINT32:  ok = ok && gguf_fread_el (file, &kv->value.uint32,  sizeof(kv->value.uint32),  &offset); break;
                case GGUF_TYPE_INT32:   ok = ok && gguf_fread_el (file, &kv->value.int32,   sizeof(kv->value.int32),   &offset); break;
                case GGUF_TYPE_FLOAT32: ok = ok && gguf_fread_el (file, &kv->value.float32, sizeof(kv->value.float32), &offset); break;
                case GGUF_TYPE_UINT64:  ok = ok && gguf_fread_el (file, &kv->value.uint64,  sizeof(kv->value.uint64),  &offset); break;
                case GGUF_TYPE_INT64:   ok = ok && gguf_fread_el (file, &kv->value.int64,   sizeof(kv->value.int64),   &offset); break;
                case GGUF_TYPE_FLOAT64: ok = ok && gguf_fread_el (file, &kv->value.float64, sizeof(kv->value.float64), &offset); break;
                case GGUF_TYPE_BOOL:    ok = ok && gguf_fread_el (file, &kv->value.bool_,   sizeof(kv->value.bool_),   &offset); break;
                case GGUF_TYPE_STRING:  ok = ok && gguf_fread_str(file, &kv->value.str,                                &offset); break;
                case GGUF_TYPE_ARRAY:
                    {
                        ok = ok && gguf_fread_el(file, &kv->value.arr.type, sizeof(kv->value.arr.type), &offset);
                        ok = ok && gguf_fread_el(file, &kv->value.arr.n,    sizeof(kv->value.arr.n), &offset);

                        switch (kv->value.arr.type) {
                            case GGUF_TYPE_UINT8:
                            case GGUF_TYPE_INT8:
                            case GGUF_TYPE_UINT16:
                            case GGUF_TYPE_INT16:
                            case GGUF_TYPE_UINT32:
                            case GGUF_TYPE_INT32:
                            case GGUF_TYPE_FLOAT32:
                            case GGUF_TYPE_UINT64:
                            case GGUF_TYPE_INT64:
                            case GGUF_TYPE_FLOAT64:
                            case GGUF_TYPE_BOOL:
                                {
                                    kv->value.arr.data = malloc(kv->value.arr.n * GGUF_TYPE_SIZE[kv->value.arr.type]);
                                    ok = ok && gguf_fread_el(file, kv->value.arr.data, kv->value.arr.n * GGUF_TYPE_SIZE[kv->value.arr.type], &offset);
                                } break;
                            case GGUF_TYPE_STRING:
                                {
                                    kv->value.arr.data = malloc(kv->value.arr.n * sizeof(struct gguf_str));
                                    for (uint32_t j = 0; j < kv->value.arr.n; ++j) {
                                        ok = ok && gguf_fread_str(file, &((struct gguf_str *) kv->value.arr.data)[j], &offset);
                                    }
                                } break;
                            case GGUF_TYPE_ARRAY:
                            case GGUF_TYPE_COUNT: GGML_ASSERT(false && "invalid type"); break;
                        }
                    } break;
                case GGUF_TYPE_COUNT: GGML_ASSERT(false && "invalid type");
            }

            if (!ok) {
                break;
            }
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read key-value pairs\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }
    }

    // read the tensor infos
    {
        ctx->infos = malloc(ctx->header.n_tensors * sizeof(struct gguf_tensor_info));

        for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];

            for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                info->ne[j] = 1;
            }

            ok = ok && gguf_fread_str(file, &info->name,                          &offset);
            ok = ok && gguf_fread_el (file, &info->n_dims, sizeof(info->n_dims),  &offset);
            for (uint32_t j = 0; j < info->n_dims; ++j) {
                ok = ok && gguf_fread_el(file, &info->ne[j], sizeof(info->ne[j]), &offset);
            }
            ok = ok && gguf_fread_el (file, &info->type,   sizeof(info->type),    &offset);
            ok = ok && gguf_fread_el (file, &info->offset, sizeof(info->offset),  &offset);

            if (!ok) {
                fprintf(stderr, "%s: failed to read tensor info\n", __func__);
                fclose(file);
                gguf_free(ctx);
                return NULL;
            }
        }
    }

    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    int alignment_idx = gguf_find_key(ctx, "general.alignment");
    if (alignment_idx != -1) {
        ctx->alignment = gguf_get_val_u32(ctx, alignment_idx);
    }

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset_pad = offset % ctx->alignment;

        if (offset_pad != 0) {
            offset += ctx->alignment - offset_pad;
            fseek(file, offset, SEEK_SET);
        }
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = offset;

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];

            const int64_t ne =
                (int64_t) info->ne[0] *
                (int64_t) info->ne[1] *
                (int64_t) info->ne[2] *
                (int64_t) info->ne[3];

            if (ggml_blck_size(info->type) == 0 || ne % ggml_blck_size(info->type) != 0) {
                printf("%d\n", info->type);
                fprintf(stderr, "%s: tensor '%s' number of elements (%" PRId64 ") is not a multiple of block size (%d). %d\n",
                        __func__, info->name.data, ne, ggml_blck_size(info->type));
                fclose(file);
                gguf_free(ctx);
                return NULL;
            }

            const size_t size_cur = (ne*ggml_type_size(info->type))/ggml_blck_size(info->type);

            ctx->size += GGML_PAD(size_cur, ctx->alignment);
        }
    }

#if 0
    // load the tensor data only if requested
    if (params.ctx != NULL) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        // the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        const size_t mem_size =
            params.no_alloc ?
            (ctx->header.n_tensors    )*ggml_tensor_overhead() :
            (ctx->header.n_tensors + 1)*ggml_tensor_overhead() + ctx->size;

        struct ggml_init_params pdata = {
            .mem_size   = mem_size,
            .mem_buffer = NULL,
            .no_alloc   = params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);

        struct ggml_context * ctx_data = *params.ctx;

        struct ggml_tensor * data = NULL;

        if (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != NULL;

            // read the binary blob with the tensor data
            ok = ok && gguf_fread_el(file, data->data, ctx->size, &offset);

            if (!ok) {
                fprintf(stderr, "%s: failed to read tensor data\n", __func__);
                fclose(file);
                ggml_free(ctx_data);
                gguf_free(ctx);
                return NULL;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, true);

        // create the tensors
        for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
            const int64_t ne[GGML_MAX_DIMS] = {
                ctx->infos[i].ne[0],
                ctx->infos[i].ne[1],
                ctx->infos[i].ne[2],
                ctx->infos[i].ne[3],
            };

            struct ggml_tensor * cur = ggml_new_tensor(ctx_data, ctx->infos[i].type, ctx->infos[i].n_dims, ne);

            ok = ok && cur != NULL;

            ggml_set_name(cur, ctx->infos[i].name.data);

            if (!ok) {
                break;
            }

            // point the data member to the appropriate location in the binary blob using the tensor infos
            if (!params.no_alloc) {
              //cur->data = (char *) data->data + ctx->infos[i].offset - ctx->offset; // offset from start of file
                cur->data = (char *) data->data + ctx->infos[i].offset;               // offset from data
            }
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read the tensor data\n", __func__);
            fclose(file);
            ggml_free(ctx_data);
            gguf_free(ctx);
            return NULL;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }
#endif

    fclose(file);

    return ctx;
}

void gguf_free(struct gguf_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    if (ctx->kv) {
        // free string memory - not great..
        for (uint32_t i = 0; i < ctx->header.n_kv; ++i) {
            struct gguf_kv * kv = &ctx->kv[i];

            if (kv->key.data) {
                free(kv->key.data);
            }

            if (kv->type == GGUF_TYPE_STRING) {
                if (kv->value.str.data) {
                    free(kv->value.str.data);
                }
            }

            if (kv->type == GGUF_TYPE_ARRAY) {
                if (kv->value.arr.data) {
                    if (kv->value.arr.type == GGUF_TYPE_STRING) {
                        for (uint32_t j = 0; j < kv->value.arr.n; ++j) {
                            struct gguf_str * str = &((struct gguf_str *) kv->value.arr.data)[j];
                            if (str->data) {
                                free(str->data);
                            }
                        }
                    }
                    free(kv->value.arr.data);
                }
            }
        }

        free(ctx->kv);
    }

    if (ctx->infos) {
        for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];

            if (info->name.data) {
                free(info->name.data);
            }
        }

        free(ctx->infos);
    }

    GGML_ALIGNED_FREE(ctx);
}

const char * gguf_type_name(enum gguf_type type) {
    return GGUF_TYPE_NAME[type];
}

int gguf_get_version(const struct gguf_context * ctx) {
    return ctx->header.version;
}

size_t gguf_get_alignment(const struct gguf_context * ctx) {
    return ctx->alignment;
}

size_t gguf_get_data_offset(const struct gguf_context * ctx) {
    return ctx->offset;
}

void * gguf_get_data(const struct gguf_context * ctx) {
    return ctx->data;
}

int gguf_get_n_kv(const struct gguf_context * ctx) {
    return ctx->header.n_kv;
}

int gguf_find_key(const struct gguf_context * ctx, const char * key) {
    // return -1 if key not found
    int keyfound = -1;

    const int n_kv = gguf_get_n_kv(ctx);

    for (int i = 0; i < n_kv; ++i) {
        if (strcmp(key, gguf_get_key(ctx, i)) == 0) {
            keyfound = i;
            break;
        }
    }

    return keyfound;
}

const char * gguf_get_key(const struct gguf_context * ctx, int key_id) {
    return ctx->kv[key_id].key.data;
}

enum gguf_type gguf_get_kv_type(const struct gguf_context * ctx, int key_id) {
    return ctx->kv[key_id].type;
}

enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.type;
}

const void * gguf_get_arr_data(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.data;
}

const char * gguf_get_arr_str(const struct gguf_context * ctx, int key_id, int i) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    struct gguf_kv * kv = &ctx->kv[key_id];
    struct gguf_str * str = &((struct gguf_str *) kv->value.arr.data)[i];
    return str->data;
}

int gguf_get_arr_n(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.n;
}

uint8_t gguf_get_val_u8(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT8);
    return ctx->kv[key_id].value.uint8;
}

int8_t gguf_get_val_i8(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_INT8);
    return ctx->kv[key_id].value.int8;
}

uint16_t gguf_get_val_u16(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT16);
    return ctx->kv[key_id].value.uint16;
}

int16_t gguf_get_val_i16(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_INT16);
    return ctx->kv[key_id].value.int16;
}

uint32_t gguf_get_val_u32(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT32);
    return ctx->kv[key_id].value.uint32;
}

int32_t gguf_get_val_i32(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_INT32);
    return ctx->kv[key_id].value.int32;
}

float gguf_get_val_f32(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_FLOAT32);
    return ctx->kv[key_id].value.float32;
}

uint64_t gguf_get_val_u64(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT64);
    return ctx->kv[key_id].value.uint64;
}

int64_t gguf_get_val_i64(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_INT64);
    return ctx->kv[key_id].value.int64;
}

double gguf_get_val_f64(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_FLOAT64);
    return ctx->kv[key_id].value.float64;
}

bool gguf_get_val_bool(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_BOOL);
    return ctx->kv[key_id].value.bool_;
}

const char * gguf_get_val_str(const struct gguf_context * ctx, int key_id) {
    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_STRING);
    return ctx->kv[key_id].value.str.data;
}

int gguf_get_n_tensors(const struct gguf_context * ctx) {
    return ctx->header.n_tensors;
}

enum ggml_sparse_deriv gguf_get_sparse_deriv(const struct gguf_context * ctx) {
    return ctx->sparse_deriv;
}

int gguf_find_tensor(const struct gguf_context * ctx, const char * name) {
    // return -1 if tensor not found
    int tensorfound = -1;

    const int n_tensors = gguf_get_n_tensors(ctx);

    for (int i = 0; i < n_tensors; ++i) {
        if (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensorfound = i;
            break;
        }
    }

    return tensorfound;
}

size_t gguf_get_tensor_offset(const struct gguf_context * ctx, int i) {
    return ctx->infos[i].offset;
}

char * gguf_get_tensor_name(const struct gguf_context * ctx, int i) {
    return ctx->infos[i].name.data;
}

// returns the index
static int gguf_get_or_add_key(struct gguf_context * ctx, const char * key) {
    const int idx = gguf_find_key(ctx, key);
    if (idx >= 0) {
        return idx;
    }

    const int n_kv = gguf_get_n_kv(ctx);

    ctx->kv = realloc(ctx->kv, (n_kv + 1) * sizeof(struct gguf_kv));
    ctx->kv[n_kv].key.n    = strlen(key);
    ctx->kv[n_kv].key.data = strdup(key);
    ctx->header.n_kv++;

    return n_kv;
}

void gguf_set_val_u8(struct gguf_context * ctx, const char * key, uint8_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = GGUF_TYPE_UINT8;
    ctx->kv[idx].value.uint8 = val;
}

void gguf_set_val_i8(struct gguf_context * ctx, const char * key, int8_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type       = GGUF_TYPE_INT8;
    ctx->kv[idx].value.int8 = val;
}

void gguf_set_val_u16(struct gguf_context * ctx, const char * key, uint16_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type         = GGUF_TYPE_UINT16;
    ctx->kv[idx].value.uint16 = val;
}

void gguf_set_val_i16(struct gguf_context * ctx, const char * key, int16_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = GGUF_TYPE_INT16;
    ctx->kv[idx].value.int16 = val;
}

void gguf_set_val_u32(struct gguf_context * ctx, const char * key, uint32_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type         = GGUF_TYPE_UINT32;
    ctx->kv[idx].value.uint32 = val;
}

void gguf_set_val_i32(struct gguf_context * ctx, const char * key, int32_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = GGUF_TYPE_INT32;
    ctx->kv[idx].value.int32 = val;
}

void gguf_set_val_f32(struct gguf_context * ctx, const char * key, float val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type          = GGUF_TYPE_FLOAT32;
    ctx->kv[idx].value.float32 = val;
}

void gguf_set_val_u64(struct gguf_context * ctx, const char * key, uint64_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type         = GGUF_TYPE_UINT64;
    ctx->kv[idx].value.uint64 = val;
}

void gguf_set_val_i64(struct gguf_context * ctx, const char * key, int64_t val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = GGUF_TYPE_INT64;
    ctx->kv[idx].value.int64 = val;
}

void gguf_set_val_f64(struct gguf_context * ctx, const char * key, double val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type          = GGUF_TYPE_FLOAT64;
    ctx->kv[idx].value.float64 = val;
}

void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = GGUF_TYPE_BOOL;
    ctx->kv[idx].value.bool_ = val;
}

void gguf_set_val_str(struct gguf_context * ctx, const char * key, const char * val) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type           = GGUF_TYPE_STRING;
    ctx->kv[idx].value.str.n    = strlen(val);
    ctx->kv[idx].value.str.data = strdup(val);
}

void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type           = GGUF_TYPE_ARRAY;
    ctx->kv[idx].value.arr.type = type;
    ctx->kv[idx].value.arr.n    = n;
    ctx->kv[idx].value.arr.data = malloc(n*GGUF_TYPE_SIZE[type]);
    memcpy(ctx->kv[idx].value.arr.data, data, n*GGUF_TYPE_SIZE[type]);
}

void gguf_set_arr_str(struct gguf_context * ctx, const char * key, const char ** data, int n) {
    const int idx = gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type           = GGUF_TYPE_ARRAY;
    ctx->kv[idx].value.arr.type = GGUF_TYPE_STRING;
    ctx->kv[idx].value.arr.n    = n;
    ctx->kv[idx].value.arr.data = malloc(n*sizeof(struct gguf_str));
    for (int i = 0; i < n; i++) {
        struct gguf_str * str = &((struct gguf_str *)ctx->kv[idx].value.arr.data)[i];
        str->n    = strlen(data[i]);
        str->data = strdup(data[i]);
    }
}

// set or add KV pairs from another context
void gguf_set_kv(struct gguf_context * ctx, struct gguf_context * src) {
    for (uint32_t i = 0; i < src->header.n_kv; i++) {
        switch (src->kv[i].type) {
            case GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, src->kv[i].key.data, src->kv[i].value.uint8);    break;
            case GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, src->kv[i].key.data, src->kv[i].value.int8);     break;
            case GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, src->kv[i].key.data, src->kv[i].value.uint16);   break;
            case GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, src->kv[i].key.data, src->kv[i].value.int16);    break;
            case GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, src->kv[i].key.data, src->kv[i].value.uint32);   break;
            case GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, src->kv[i].key.data, src->kv[i].value.int32);    break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, src->kv[i].key.data, src->kv[i].value.float32);  break;
            case GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, src->kv[i].key.data, src->kv[i].value.uint64);   break;
            case GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, src->kv[i].key.data, src->kv[i].value.int64);    break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, src->kv[i].key.data, src->kv[i].value.float64);  break;
            case GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, src->kv[i].key.data, src->kv[i].value.bool_);    break;
            case GGUF_TYPE_STRING:  gguf_set_val_str (ctx, src->kv[i].key.data, src->kv[i].value.str.data); break;
            case GGUF_TYPE_ARRAY:
                {
                    if (src->kv[i].value.arr.type == GGUF_TYPE_STRING) {
                        const char ** data = malloc(src->kv[i].value.arr.n*sizeof(char *));
                        for (uint32_t j = 0; j < src->kv[i].value.arr.n; j++) {
                            data[j] = ((struct gguf_str *)src->kv[i].value.arr.data)[j].data;
                        }
                        gguf_set_arr_str(ctx, src->kv[i].key.data, data, src->kv[i].value.arr.n);
                        free(data);
                    } else if (src->kv[i].value.arr.type == GGUF_TYPE_ARRAY) {
                        GGML_ASSERT(false && "nested arrays not supported");
                    } else {
                        gguf_set_arr_data(ctx, src->kv[i].key.data, src->kv[i].value.arr.type, src->kv[i].value.arr.data, src->kv[i].value.arr.n);
                    }
                } break;
            case GGUF_TYPE_COUNT:  GGML_ASSERT(false && "invalid type"); break;
        }
    }
}

void gguf_add_tensor(
             struct gguf_context * ctx,
        const struct ggml_tensor * tensor) {
    const int idx = ctx->header.n_tensors;
    ctx->infos = realloc(ctx->infos, (idx + 1)*sizeof(struct gguf_tensor_info));

    ctx->infos[idx].name.n    = strlen(tensor->name);
    ctx->infos[idx].name.data = strdup(tensor->name);

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        ctx->infos[idx].ne[i] = 1;
    }

    ctx->infos[idx].n_dims = tensor->n_dims;
    for (int i = 0; i < tensor->n_dims; i++) {
        ctx->infos[idx].ne[i] = tensor->ne[i];
    }

    ctx->infos[idx].type   = tensor->type;
    ctx->infos[idx].offset = 0;
    ctx->infos[idx].data   = tensor->data;
    ctx->infos[idx].size   = ggml_nbytes(tensor);

    if (ctx->header.n_tensors > 0) {
        ctx->infos[idx].offset = ctx->infos[idx - 1].offset + GGML_PAD(ctx->infos[idx - 1].size, ctx->alignment);
    }

    ctx->header.n_tensors++;
}

void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type) {
    const int idx = gguf_find_tensor(ctx, name);
    if (idx < 0) {
        GGML_ASSERT(false && "tensor not found");
    }

    ctx->infos[idx].type = type;
}

void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data, size_t size) {
    const int idx = gguf_find_tensor(ctx, name);
    if (idx < 0) {
        GGML_ASSERT(false && "tensor not found");
    }

    ctx->infos[idx].data = data;
    ctx->infos[idx].size = size;

    // update offsets
    for (uint32_t i = idx + 1; i < ctx->header.n_tensors; ++i) {
        ctx->infos[i].offset = ctx->infos[i - 1].offset + GGML_PAD(ctx->infos[i - 1].size, ctx->alignment);
    }
}

//static void gguf_fwrite_str(FILE * file, const struct gguf_str * val) {
//    fwrite(&val->n,   sizeof(val->n),    1, file);
//    fwrite(val->data, sizeof(char), val->n, file);
//}
//
//static void gguf_fwrite_el(FILE * file, const void * val, size_t size) {
//    fwrite(val, sizeof(char), size, file);
//}

struct gguf_buf {
    void * data;
    size_t size;
    size_t offset;
};

static struct gguf_buf gguf_buf_init(size_t size) {
    struct gguf_buf buf = {
        /*buf.data   =*/ size == 0 ? NULL : malloc(size),
        /*buf.size   =*/ size,
        /*buf.offset =*/ 0,
    };

    return buf;
}

static void gguf_buf_free(struct gguf_buf buf) {
    if (buf.data) {
        free(buf.data);
    }
}

static void gguf_buf_grow(struct gguf_buf * buf, size_t size) {
    if (buf->offset + size > buf->size) {
        buf->size = 1.5*(buf->offset + size);
        if (buf->data) {
            buf->data = realloc(buf->data, buf->size);
        }
    }
}

static void gguf_bwrite_str(struct gguf_buf * buf, const struct gguf_str * val) {
    gguf_buf_grow(buf, sizeof(val->n) + val->n);

    if (buf->data) {
        memcpy((char *) buf->data + buf->offset, &val->n, sizeof(val->n));
    }
    buf->offset += sizeof(val->n);

    if (buf->data) {
        memcpy((char *) buf->data + buf->offset, val->data, val->n);
    }
    buf->offset += val->n;
}

static void gguf_bwrite_el(struct gguf_buf * buf, const void * val, size_t el_size) {
    gguf_buf_grow(buf, el_size);

    if (buf->data) {
        memcpy((char *) buf->data + buf->offset, val, el_size);
    }
    buf->offset += el_size;
}

static void gguf_write_to_buf(const struct gguf_context * ctx, struct gguf_buf * buf, bool only_meta) {
    // write header
    gguf_bwrite_el(buf, &ctx->header.magic,     sizeof(ctx->header.magic));
    gguf_bwrite_el(buf, &ctx->header.version,   sizeof(ctx->header.version));
    gguf_bwrite_el(buf, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors));
    gguf_bwrite_el(buf, &ctx->header.n_kv,      sizeof(ctx->header.n_kv));

    // write key-value pairs
    for (uint32_t i = 0; i < ctx->header.n_kv; ++i) {
        struct gguf_kv * kv = &ctx->kv[i];

        gguf_bwrite_str(buf, &kv->key);
        gguf_bwrite_el (buf, &kv->type, sizeof(kv->type));

        switch (kv->type) {
            case GGUF_TYPE_UINT8:   gguf_bwrite_el( buf, &kv->value.uint8,   sizeof(kv->value.uint8)  ); break;
            case GGUF_TYPE_INT8:    gguf_bwrite_el (buf, &kv->value.int8,    sizeof(kv->value.int8)   ); break;
            case GGUF_TYPE_UINT16:  gguf_bwrite_el (buf, &kv->value.uint16,  sizeof(kv->value.uint16) ); break;
            case GGUF_TYPE_INT16:   gguf_bwrite_el (buf, &kv->value.int16,   sizeof(kv->value.int16)  ); break;
            case GGUF_TYPE_UINT32:  gguf_bwrite_el (buf, &kv->value.uint32,  sizeof(kv->value.uint32) ); break;
            case GGUF_TYPE_INT32:   gguf_bwrite_el (buf, &kv->value.int32,   sizeof(kv->value.int32)  ); break;
            case GGUF_TYPE_FLOAT32: gguf_bwrite_el (buf, &kv->value.float32, sizeof(kv->value.float32)); break;
            case GGUF_TYPE_UINT64:  gguf_bwrite_el (buf, &kv->value.uint64,  sizeof(kv->value.uint64) ); break;
            case GGUF_TYPE_INT64:   gguf_bwrite_el (buf, &kv->value.int64,   sizeof(kv->value.int64)  ); break;
            case GGUF_TYPE_FLOAT64: gguf_bwrite_el (buf, &kv->value.float64, sizeof(kv->value.float64)); break;
            case GGUF_TYPE_BOOL:    gguf_bwrite_el (buf, &kv->value.bool_,   sizeof(kv->value.bool_)  ); break;
            case GGUF_TYPE_STRING:  gguf_bwrite_str(buf, &kv->value.str                               ); break;
            case GGUF_TYPE_ARRAY:
                {
                    gguf_bwrite_el(buf, &kv->value.arr.type, sizeof(kv->value.arr.type));
                    gguf_bwrite_el(buf, &kv->value.arr.n,    sizeof(kv->value.arr.n)   );

                    switch (kv->value.arr.type) {
                        case GGUF_TYPE_UINT8:
                        case GGUF_TYPE_INT8:
                        case GGUF_TYPE_UINT16:
                        case GGUF_TYPE_INT16:
                        case GGUF_TYPE_UINT32:
                        case GGUF_TYPE_INT32:
                        case GGUF_TYPE_FLOAT32:
                        case GGUF_TYPE_UINT64:
                        case GGUF_TYPE_INT64:
                        case GGUF_TYPE_FLOAT64:
                        case GGUF_TYPE_BOOL:
                            {
                                gguf_bwrite_el(buf, kv->value.arr.data, kv->value.arr.n * GGUF_TYPE_SIZE[kv->value.arr.type]);
                            } break;
                        case GGUF_TYPE_STRING:
                            {
                                for (uint32_t j = 0; j < kv->value.arr.n; ++j) {
                                    gguf_bwrite_str(buf, &((struct gguf_str *) kv->value.arr.data)[j]);
                                }
                            } break;
                        case GGUF_TYPE_ARRAY:
                        case GGUF_TYPE_COUNT: GGML_ASSERT(false && "invalid type"); break;
                    }
                } break;
            case GGUF_TYPE_COUNT: GGML_ASSERT(false && "invalid type");
        }
    }

    // write tensor infos
    for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
        struct gguf_tensor_info * info = &ctx->infos[i];

        gguf_bwrite_str(buf, &info->name);
        gguf_bwrite_el (buf, &info->n_dims, sizeof(info->n_dims));
        for (uint32_t j = 0; j < info->n_dims; ++j) {
            gguf_bwrite_el(buf, &info->ne[j], sizeof(info->ne[j]));
        }
        gguf_bwrite_el(buf, &info->type,   sizeof(info->type));
        gguf_bwrite_el(buf, &info->offset, sizeof(info->offset));
    }

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset     = buf->offset;
        const size_t offset_pad = GGML_PAD(offset, ctx->alignment);

        if (offset_pad != offset) {
            uint8_t pad = 0;
            for (size_t i = 0; i < offset_pad - offset; ++i) {
                gguf_bwrite_el(buf, &pad, sizeof(pad));
            }
        }
    }

    if (only_meta) {
        return;
    }

    size_t offset = 0;

    // write tensor data
    for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
        struct gguf_tensor_info * info = &ctx->infos[i];

        const size_t size     = info->size;
        const size_t size_pad = GGML_PAD(size, ctx->alignment);

        gguf_bwrite_el(buf, info->data, size);

        if (size_pad != size) {
            uint8_t pad = 0;
            for (size_t j = 0; j < size_pad - size; ++j) {
                gguf_bwrite_el(buf, &pad, sizeof(pad));
            }
        }

        GGML_ASSERT(offset == info->offset);

        offset += size_pad;
    }
}

void gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta) {
    FILE * file = fopen(fname, "wb");
    if (!file) {
        GGML_ASSERT(false && "failed to open file for writing");
    }

    struct gguf_buf buf = gguf_buf_init(16*1024);

    gguf_write_to_buf(ctx, &buf, only_meta);

    fwrite(buf.data, 1, buf.offset, file);

    gguf_buf_free(buf);

    fclose(file);
}

size_t gguf_get_meta_size(const struct gguf_context * ctx) {
    // no allocs - only compute size
    struct gguf_buf buf = gguf_buf_init(0);

    gguf_write_to_buf(ctx, &buf, true);

    return buf.offset;
}

void gguf_get_meta_data(const struct gguf_context * ctx, void * data) {
    struct gguf_buf buf = gguf_buf_init(16*1024);

    gguf_write_to_buf(ctx, &buf, true);

    memcpy(data, buf.data, buf.offset);

    gguf_buf_free(buf);
}