/*
 * omni.h — the OMNI C ABI.
 *
 * The substrate every non-Rust binding is built on (docs/design/sdk.md §3).
 * Implemented by `omni-ffi`, which is the only crate in this repository that
 * uses `unsafe`: omni-core parses untrusted binary input and is
 * `#![forbid(unsafe_code)]`, so the pointer arithmetic a C ABI cannot avoid is
 * confined to a crate that does no parsing at all.
 *
 * Ownership
 * ---------
 * Handles are opaque. Every call that produces one has exactly one call that
 * releases it: omni_store_close, omni_model_free, omni_plan_free,
 * omni_tensor_release. Releasing NULL is defined and does nothing. The handles
 * are reference-counted internally, so a tensor outliving the store it came
 * from is fine — that is deliberate, because a Python object outliving a
 * borrow is the classic zero-copy-binding bug.
 *
 * Threads
 * -------
 * A handle must not be used from two threads at once. omni_last_error is
 * thread-local, so an error is always the error your thread caused.
 *
 * Errors
 * ------
 * Every fallible call returns an omni_status. omni_last_error() then holds a
 * sentence about it, valid until this thread's next call. No call unwinds: a
 * panic inside the implementation becomes OMNI_EINTERNAL.
 *
 * Linking
 * -------
 *   cargo build --release -p omni-ffi
 *   cc prog.c -I reference/omni-ffi/include \
 *      reference/target/release/libomni.a -lpthread -ldl -lm
 * or link -lomni against the cdylib.
 */

#ifndef OMNI_H
#define OMNI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- versioning */

/*
 * High 16 bits: incompatible changes. Low 16 bits: additive ones. A caller
 * whose compiled-in high half differs from omni_abi_version()'s must stop.
 */
#define OMNI_ABI_VERSION 0x00010000u

uint32_t    omni_abi_version(void);
const char *omni_spec_version(void);

/* ---------------------------------------------------------------- statuses */

/*
 * These are the CLI's exit codes (docs/design/cli.md §10.3), so a C caller and
 * `omni` agree about what happened, plus one the CLI has no need for: a caught
 * panic is a return value here and a crash there.
 *
 * OMNI_INDETERMINATE is the one that matters and the one most tools get wrong.
 * It means the file is fine and this build cannot fully handle it: an unknown
 * critical extension, a codec that is not compiled in, a dtype DLPack cannot
 * spell. Treating it as invalid is how an ecosystem fragments (§14.4).
 */
typedef int omni_status;
#define OMNI_OK             0
#define OMNI_EINVALID       1
#define OMNI_EUSAGE         2
#define OMNI_INDETERMINATE  3
#define OMNI_EPOLICY        4
#define OMNI_EINCOMPLETE    5
#define OMNI_EINFEASIBLE    6
#define OMNI_EINTERNAL      7

/* Never NULL; empty after a successful call. Valid until this thread's next
 * call into the library. */
const char *omni_last_error(void);
/* A stable name for a status code, e.g. "indeterminate". Static storage. */
const char *omni_status_name(omni_status status);

/* ----------------------------------------------------------------- handles */

typedef struct omni_store  omni_store;
typedef struct omni_model  omni_model;
typedef struct omni_plan   omni_plan;
typedef struct omni_tensor omni_tensor;

/* -------------------------------------------------------------------- store */

omni_status omni_store_open(const char *path, omni_store **out);
/* Copies `len` bytes; the caller may free them on return. */
omni_status omni_store_open_bytes(const uint8_t *bytes, size_t len,
                                  omni_store **out);
void        omni_store_close(omni_store *s);

uint64_t    omni_store_size(omni_store *s);
uint64_t    omni_store_object_count(omni_store *s);
/* The §03.5.1 name: "blake3-256" or "sha2-256". Static storage. */
omni_status omni_store_hash_name(omni_store *s, const char **out);
/* Copies the 32-byte root digest into `d32`. */
omni_status omni_store_root_digest(omni_store *s, uint8_t *d32);

/*
 * Raises the per-node materialization cap (§12.4), which defaults to 2^28
 * elements. A declared size is untrusted input, so loading something enormous
 * is a decision the caller makes rather than one a header makes for it.
 */
omni_status omni_store_set_max_elems(omni_store *s, uint64_t n);

typedef struct {
    uint64_t segments;
    uint64_t objects_verified;
    uint64_t bytes_verified;
    uint64_t reachable;
    uint64_t dangling;
    uint64_t mistyped;
    int      padding_ok;
    int      alignment_ok;
} omni_verify_report;

/*
 * Checks every object's digest, the framing, and reachability. OMNI_OK when
 * clean, OMNI_EINCOMPLETE when something referenced is absent, OMNI_EINVALID
 * when a rule is broken. `report` may be NULL.
 */
omni_status omni_store_verify(omni_store *s, omni_verify_report *report);

/* -------------------------------------------------------------------- model */

omni_status omni_store_root(omni_store *s, omni_model **out);
void        omni_model_free(omni_model *m);

/*
 * The manifest as JSON. Cached on the handle; valid until it is freed. `len`
 * may be NULL. The projection from CBOR is lossy in three stated ways: byte
 * strings become hex text, a tagged value becomes {"@tag":n,"value":…}, and a
 * ref carrying tag 1001 becomes {"@ref":{"t":…,"d":…}}. A bare ref stays the
 * [otype, "<hex>"] array §01.3 defines.
 */
omni_status omni_model_meta_json(omni_model *m, const char **json, size_t *len);

size_t      omni_model_tensor_count(omni_model *m);
/* Name `i` in §04.2 load order. Valid until the model is freed. */
omni_status omni_model_tensor_name(omni_model *m, size_t i, const char **out);
/*
 * Takes one tensor by name. Reads its description only — opening a model does
 * not touch its weights (§02.1).
 */
omni_status omni_model_tensor(omni_model *m, const char *name,
                              omni_tensor **out);

/* --------------------------------------------------------------------- plan */

typedef int omni_objective;
#define OMNI_OBJ_MIN_MEMORY     0
#define OMNI_OBJ_MAX_QUALITY    1
#define OMNI_OBJ_MIN_LOAD_TIME  2
#define OMNI_OBJ_MIN_LATENCY    3
#define OMNI_OBJ_BALANCED       4

/*
 * Negotiates a plan (§10.5). `caps_json` may be NULL, meaning the C0 baseline
 * — the floor every conforming reader meets, which deliberately does not
 * include the expression feature, so a model that needs it comes back
 * OMNI_EINFEASIBLE rather than silently reduced.
 *
 * The plan is produced either way: on OMNI_EINFEASIBLE, `*out` is still set and
 * omni_plan_json names what was unmet and what would fix it.
 */
omni_status omni_model_resolve(omni_model *m, const char *caps_json,
                               omni_objective objective, omni_plan **out);

int         omni_plan_feasible(omni_plan *p);
uint64_t    omni_plan_resident_bytes(omni_plan *p);
uint64_t    omni_plan_read_bytes(omni_plan *p);
omni_status omni_plan_json(omni_plan *p, const char **json, size_t *len);
void        omni_plan_free(omni_plan *p);

/* ------------------------------------------------------------------- tensor */

typedef struct {
    const char     *dtype;       /* §04.3 label: "bf16", "i4", "q8.8" */
    uint32_t        dtype_bits;  /* bits per element, rounded up */
    const char     *layout;      /* §04.4 kind: "strided", "packed", … */
    const char     *value_op;    /* §04.7 root node: "literal", "dequantize" */
    uint32_t        ndim;
    const uint64_t *shape;       /* ndim extents, or NULL if symbolic */
    uint64_t        numel;       /* 0 if symbolic */
} omni_tensor_info;

/*
 * Returns OMNI_INDETERMINATE — with everything but `shape` and `numel` filled
 * in — when a dimension is symbolic (§04.7.3). The extent is genuinely unknown
 * until the model's `dims` are bound, and 0 would be a lie that looks like a
 * value.
 */
omni_status omni_tensor_get_info(omni_tensor *t, omni_tensor_info *out);

/* The name the model's table gives it. Valid until the tensor is released. */
omni_status omni_tensor_name(omni_tensor *t, const char **out);

/*
 * The stored bytes, laid out exactly as §04.3.5 says. Valid until the tensor is
 * released. OMNI_INDETERMINATE for a tensor whose value is computed rather than
 * stored: a `dequantize` has no bytes of its own, and handing back its operand
 * would be the wrong array with the right length.
 */
omni_status omni_tensor_bytes(omni_tensor *t, const void **ptr, size_t *len);

/*
 * 1 when the pointer from omni_tensor_bytes points into the container itself
 * rather than a copy — the Bytes::Mapped / Bytes::Owned distinction of the SDK
 * design, made observable so a caller can tell whether it got zero copy.
 */
int omni_tensor_mapped(omni_tensor *t);

/*
 * Evaluates to double elements whatever the value expression is: a literal is
 * decoded through its dtype and layout, a `dequantize` is computed. Cached on
 * the handle. This is the C1 path and it costs 8 bytes an element, which is why
 * it is a separate call rather than the only way to read.
 */
omni_status omni_tensor_values(omni_tensor *t, const double **ptr, size_t *len);

void omni_tensor_release(omni_tensor *t);

/* ------------------------------------------------------------------- DLPack */

/*
 * DLPack is the interop lingua franca: PyTorch, JAX, CuPy, TensorFlow, MLX and
 * NumPy all consume it. These declarations match dlpack.h; a program that
 * already includes dlpack.h should define OMNI_NO_DLPACK_TYPES before this
 * header.
 */
#ifndef OMNI_NO_DLPACK_TYPES
typedef struct { int32_t device_type; int32_t device_id; } DLDevice;
typedef struct { uint8_t code; uint8_t bits; uint16_t lanes; } DLDataType;
typedef struct {
    void      *data;
    DLDevice   device;
    int32_t    ndim;
    DLDataType dtype;
    int64_t   *shape;
    int64_t   *strides;
    uint64_t   byte_offset;
} DLTensor;
typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void    *manager_ctx;
    void   (*deleter)(struct DLManagedTensor *self);
} DLManagedTensor;
#endif

#define OMNI_DLPACK_CPU 1

/*
 * Hands the tensor to a DLPack consumer, without a copy when the bytes were
 * mappable. The caller owns the result and must call its deleter; after this
 * call the omni_tensor may be released, because the DLPack object keeps what it
 * needs alive.
 *
 * OMNI_INDETERMINATE, naming the reason, when the dtype has no DLPack spelling
 * (i4, ternary, codebook, fixed point — DLPack describes whole-byte lanes) or
 * when the layout is not dense row-major, since strides == NULL means exactly
 * that and a packed buffer described as dense would be read wrongly.
 */
omni_status omni_tensor_dlpack(omni_tensor *t, DLManagedTensor **out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* OMNI_H */
