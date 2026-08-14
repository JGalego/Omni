/*
 * write.c — building a container from C, through the C ABI.
 *
 * read.c is the evidence that a C program can consume a model. This is the
 * other half of docs/design/sdk.md §3, and it is the half that decides whether
 * a binding is a *binding* or a viewer: a language that can only read has to go
 * back through Rust to publish anything.
 *
 * The program builds a small two-tensor model, sets the metadata §06 asks for,
 * hands one array over as DLPack (the same struct PyTorch, JAX and NumPy speak,
 * so this is the path an array from any of them takes), writes the container,
 * and then opens what it wrote through the read path to check it says the same
 * thing back. It also checks the property §01.10 makes a requirement: the same
 * calls produce the same bytes.
 *
 *   cc -Wall -Wextra -Werror -std=c11 write.c -I ../include \
 *      ../../target/release/libomni.a -lpthread -ldl -lm -o write
 *   ./write out.omni
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omni.h"

static int fail(const char *what, omni_status st) {
    fprintf(stderr, "%s: %s (%d): %s\n", what, omni_status_name(st), st,
            omni_last_error());
    return 1;
}

/* bf16 is the top 16 bits of the float32 with the same value. */
static uint16_t bf16_of(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof bits);
    return (uint16_t)(bits >> 16);
}

/* Fills a builder with exactly the same calls every time, so two of them can
 * be compared byte for byte. */
static omni_status populate(omni_builder *b, uint16_t *w, size_t w_elems,
                            float *bias, int64_t *bias_shape) {
    omni_status st;
    if ((st = omni_builder_set_license(b, "Apache-2.0")) != OMNI_OK) return st;
    if ((st = omni_builder_set_arch(b, "transformer.decoder",
                                    "{\"n_layers\": 1, \"d_model\": 8}")) != OMNI_OK)
        return st;
    if ((st = omni_builder_add_meta(b, "description",
                                    "\"written from C\"")) != OMNI_OK)
        return st;
    if ((st = omni_builder_set_chunk_size(b, 4096)) != OMNI_OK) return st;

    uint64_t shape[2] = {4, 8};
    if ((st = omni_builder_add_tensor(b, "model.layers.0.attn.q_proj.weight",
                                      "bf16", shape, 2, w,
                                      w_elems * sizeof *w)) != OMNI_OK)
        return st;
    if ((st = omni_builder_set_tensor_axes(b, "model.layers.0.attn.q_proj.weight",
                                           "out_features,in_features")) != OMNI_OK)
        return st;
    if ((st = omni_builder_set_tensor_semantic(b,
                                               "model.layers.0.attn.q_proj.weight",
                                               "weight")) != OMNI_OK)
        return st;

    /* The second tensor arrives as DLPack — the way an array from PyTorch,
     * JAX, NumPy or MLX arrives, with no copy on the caller's side. */
    DLTensor dl;
    memset(&dl, 0, sizeof dl);
    dl.data = bias;
    dl.device.device_type = OMNI_DLPACK_CPU;
    dl.device.device_id = 0;
    dl.ndim = 1;
    dl.dtype.code = 2;  /* kDLFloat */
    dl.dtype.bits = 32;
    dl.dtype.lanes = 1;
    dl.shape = bias_shape;
    dl.strides = NULL;  /* dense row-major */
    dl.byte_offset = 0;
    if ((st = omni_builder_add_dlpack(b, "model.layers.0.attn.q_proj.bias",
                                      &dl)) != OMNI_OK)
        return st;
    return omni_builder_set_tensor_semantic(b, "model.layers.0.attn.q_proj.bias",
                                            "bias");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: write <out.omni>\n");
        return 2;
    }
    if ((omni_abi_version() >> 16) != (OMNI_ABI_VERSION >> 16)) {
        fprintf(stderr, "ABI major mismatch: header %u, library %u\n",
                OMNI_ABI_VERSION >> 16, omni_abi_version() >> 16);
        return 1;
    }

    enum { ROWS = 4, COLS = 8, W_ELEMS = ROWS * COLS };
    uint16_t w[W_ELEMS];
    for (int i = 0; i < W_ELEMS; i++) w[i] = bf16_of((float)i * 0.25f - 4.0f);
    float bias[ROWS];
    for (int i = 0; i < ROWS; i++) bias[i] = (float)i;
    int64_t bias_shape[1] = {ROWS};

    omni_builder *b = NULL;
    omni_status st = omni_builder_new("acme/from-c", &b);
    if (st != OMNI_OK) return fail("builder_new", st);
    if ((st = populate(b, w, W_ELEMS, bias, bias_shape)) != OMNI_OK)
        return fail("populate", st);
    printf("built %zu tensors\n", omni_builder_tensor_count(b));

    /* The root digest is known before a byte is written: identity is a
     * function of content (§01.2), and packing only decides placement. */
    uint8_t predicted[32];
    if ((st = omni_builder_root_digest(b, predicted)) != OMNI_OK)
        return fail("root_digest", st);

    if ((st = omni_builder_write(b, argv[1])) != OMNI_OK)
        return fail("write", st);

    const uint8_t *bytes = NULL;
    size_t len = 0;
    if ((st = omni_builder_write_bytes(b, &bytes, &len)) != OMNI_OK)
        return fail("write_bytes", st);
    printf("wrote %s, %zu bytes\n", argv[1], len);

    /* §01.10 writer rule W1: same calls, same bytes. A second builder, given
     * the same calls, must produce a byte-identical container. */
    omni_builder *b2 = NULL;
    if ((st = omni_builder_new("acme/from-c", &b2)) != OMNI_OK)
        return fail("builder_new(2)", st);
    if ((st = populate(b2, w, W_ELEMS, bias, bias_shape)) != OMNI_OK)
        return fail("populate(2)", st);
    const uint8_t *bytes2 = NULL;
    size_t len2 = 0;
    if ((st = omni_builder_write_bytes(b2, &bytes2, &len2)) != OMNI_OK)
        return fail("write_bytes(2)", st);
    if (len != len2 || memcmp(bytes, bytes2, len) != 0) {
        fprintf(stderr, "the same calls produced different bytes\n");
        return 1;
    }
    printf("reproducible=1  (%zu bytes, twice)\n", len);
    omni_builder_free(b2);

    /* A wrong byte count is caught by the call that made it, not by the
     * write — R-T02 is about the tensor, so it is checked where the tensor is
     * described. */
    uint64_t bad_shape[2] = {4, 4};
    st = omni_builder_add_tensor(b, "wrong", "f32", bad_shape, 2, w, 8);
    if (st != OMNI_EINVALID) {
        fprintf(stderr, "a short buffer was accepted (%d)\n", st);
        return 1;
    }
    printf("short buffer: %s\n", omni_last_error());

    /* A codec in §03.7.1 this build cannot produce is indeterminate, and the
     * container is never quietly written uncompressed instead. */
    st = omni_builder_set_codec(b, "brotli");
    if (st != OMNI_INDETERMINATE) {
        fprintf(stderr, "an unimplemented codec was not indeterminate (%d)\n", st);
        return 1;
    }
    printf("unimplemented codec: %s\n", omni_last_error());

    /* Now read back what was written, through the other half of this ABI. */
    omni_store *s = NULL;
    if ((st = omni_store_open(argv[1], &s)) != OMNI_OK) return fail("open", st);
    uint8_t root[32];
    if ((st = omni_store_root_digest(s, root)) != OMNI_OK)
        return fail("root_digest(store)", st);
    if (memcmp(root, predicted, sizeof root) != 0) {
        fprintf(stderr, "the predicted root digest is not the one written\n");
        return 1;
    }
    omni_verify_report rep;
    memset(&rep, 0, sizeof rep);
    if ((st = omni_store_verify(s, &rep)) != OMNI_OK) return fail("verify", st);
    printf("verified: %llu objects, %llu bytes, %llu reachable, %llu dangling\n",
           (unsigned long long)rep.objects_verified,
           (unsigned long long)rep.bytes_verified,
           (unsigned long long)rep.reachable,
           (unsigned long long)rep.dangling);

    omni_model *m = NULL;
    if ((st = omni_store_root(s, &m)) != OMNI_OK) return fail("root", st);
    size_t n = omni_model_tensor_count(m);
    printf("tensors: %zu\n", n);
    for (size_t i = 0; i < n; i++) {
        const char *name = NULL;
        if ((st = omni_model_tensor_name(m, i, &name)) != OMNI_OK)
            return fail("tensor_name", st);
        omni_tensor *t = NULL;
        if ((st = omni_model_tensor(m, name, &t)) != OMNI_OK)
            return fail("tensor", st);
        omni_tensor_info info;
        memset(&info, 0, sizeof info);
        if ((st = omni_tensor_get_info(t, &info)) != OMNI_OK)
            return fail("tensor_info", st);
        printf("  %s  %s  numel=%llu\n", name, info.dtype,
               (unsigned long long)info.numel);
        omni_tensor_release(t);
    }

    /* The values must be the ones handed in, decoded back through the dtype.
     * Byte identity is not the claim; the claim is that the tensor means what
     * the caller said it meant. */
    omni_tensor *t = NULL;
    if ((st = omni_model_tensor(m, "model.layers.0.attn.q_proj.bias", &t)) != OMNI_OK)
        return fail("tensor(bias)", st);
    const double *vals = NULL;
    size_t vn = 0;
    if ((st = omni_tensor_values(t, &vals, &vn)) != OMNI_OK)
        return fail("values", st);
    if (vn != ROWS) {
        fprintf(stderr, "bias came back with %zu values\n", vn);
        return 1;
    }
    for (size_t i = 0; i < vn; i++) {
        if (vals[i] != (double)bias[i]) {
            fprintf(stderr, "bias[%zu] = %f, expected %f\n", i, vals[i],
                    (double)bias[i]);
            return 1;
        }
    }
    printf("dlpack_roundtrip=1  (%zu values, exact)\n", vn);
    omni_tensor_release(t);
    omni_model_free(m);
    omni_store_close(s);
    omni_builder_free(b);

    printf("the C ABI wrote a container end to end\n");
    return 0;
}
