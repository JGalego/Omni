/*
 * read.c — the whole read path through the C ABI, in C.
 *
 * This is the evidence for docs/design/sdk.md §3: not that the header compiles,
 * but that a C program with no knowledge of Rust can open a container, verify
 * it, walk its tensors, get zero-copy bytes, negotiate a plan, and hand a
 * tensor to a DLPack consumer. Every step checks its status, because a binding
 * that ignores OMNI_INDETERMINATE is exactly the bug §14.4 is about.
 *
 *   cc -Wall -Wextra -Werror -std=c11 read.c -I ../include \
 *      ../../target/release/libomni.a -lpthread -ldl -lm -o read
 *   ./read model.omni
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

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: read <file.omni>\n");
        return 2;
    }

    if ((omni_abi_version() >> 16) != (OMNI_ABI_VERSION >> 16)) {
        fprintf(stderr, "ABI major mismatch: header %u, library %u\n",
                OMNI_ABI_VERSION >> 16, omni_abi_version() >> 16);
        return 1;
    }
    printf("abi %08x  spec %s\n", omni_abi_version(), omni_spec_version());

    omni_store *store = NULL;
    omni_status st = omni_store_open(argv[1], &store);
    if (st != OMNI_OK) return fail("open", st);

    const char *hash = NULL;
    st = omni_store_hash_name(store, &hash);
    if (st != OMNI_OK) return fail("hash", st);

    uint8_t root[32];
    st = omni_store_root_digest(store, root);
    if (st != OMNI_OK) return fail("root digest", st);

    printf("opened %s: %llu bytes, %llu objects, %s, root %02x%02x%02x%02x…\n",
           argv[1], (unsigned long long)omni_store_size(store),
           (unsigned long long)omni_store_object_count(store), hash,
           root[0], root[1], root[2], root[3]);

    omni_verify_report rep;
    memset(&rep, 0, sizeof rep);
    st = omni_store_verify(store, &rep);
    if (st != OMNI_OK) return fail("verify", st);
    printf("verified %llu object(s), %llu byte(s), %llu reachable, "
           "%llu dangling\n",
           (unsigned long long)rep.objects_verified,
           (unsigned long long)rep.bytes_verified,
           (unsigned long long)rep.reachable,
           (unsigned long long)rep.dangling);

    omni_model *model = NULL;
    st = omni_store_root(store, &model);
    if (st != OMNI_OK) return fail("root", st);

    const char *meta = NULL;
    size_t meta_len = 0;
    st = omni_model_meta_json(model, &meta, &meta_len);
    if (st != OMNI_OK) return fail("meta", st);
    printf("manifest json: %zu bytes\n", meta_len);
    if (strstr(meta, "\"t\":\"omni.core/manifest\"") == NULL) {
        fprintf(stderr, "the manifest json is not a manifest\n");
        return 1;
    }

    size_t n = omni_model_tensor_count(model);
    printf("%zu tensor(s)\n", n);
    if (n == 0) {
        fprintf(stderr, "a model with no tensors is not a useful test\n");
        return 1;
    }

    /* Walk every tensor: description, then bytes. A computed value reports
     * OMNI_INDETERMINATE rather than handing back its operand, and that is as
     * much the behaviour worth exercising as the successful reads. */
    int mapped_seen = 0, indeterminate_seen = 0;
    int dlpack_ok = 0, dlpack_refused = 0;
    for (size_t i = 0; i < n; i++) {
        const char *name = NULL;
        st = omni_model_tensor_name(model, i, &name);
        if (st != OMNI_OK) return fail("tensor name", st);

        omni_tensor *t = NULL;
        st = omni_model_tensor(model, name, &t);
        if (st != OMNI_OK) return fail(name, st);

        /* The handle knows its own name; a binding that keeps tensors in a map
         * should not have to keep the key beside them. */
        const char *own = NULL;
        if (omni_tensor_name(t, &own) != OMNI_OK || strcmp(own, name) != 0) {
            fprintf(stderr, "tensor %s does not know its own name\n", name);
            return 1;
        }

        omni_tensor_info info;
        memset(&info, 0, sizeof info);
        st = omni_tensor_get_info(t, &info);
        if (st != OMNI_OK && st != OMNI_INDETERMINATE) return fail("info", st);

        printf("  %-44s %-10s %2u-bit %-9s %s [",
               name, info.dtype, info.dtype_bits, info.layout, info.value_op);
        for (uint32_t d = 0; d < info.ndim; d++) {
            printf("%s%llu", d ? ", " : "",
                   info.shape ? (unsigned long long)info.shape[d] : 0ULL);
        }
        printf("%s]", info.shape ? "" : " symbolic");

        const void *ptr = NULL;
        size_t len = 0;
        st = omni_tensor_bytes(t, &ptr, &len);
        if (st == OMNI_OK) {
            printf("  %zu stored bytes%s\n", len,
                   omni_tensor_mapped(t) ? " (mapped)" : " (copied)");
            mapped_seen |= omni_tensor_mapped(t);
        } else if (st == OMNI_INDETERMINATE) {
            printf("  computed: %s\n", omni_last_error());
            indeterminate_seen = 1;
        } else {
            printf("\n");
            return fail("bytes", st);
        }

        /* DLPack on every tensor. The refusals are as much the point as the
         * successes: a u4 packed weight has no honest DLPack spelling, and
         * handing it over as uint8 would be wrong about both the width and the
         * layout. */
        DLManagedTensor *dl = NULL;
        st = omni_tensor_dlpack(t, &dl);
        if (st == OMNI_OK) {
            if (dlpack_ok++ == 0) {
                printf("    dlpack: device %d, code %u, %u bits, %d dim, "
                       "strides %s\n",
                       dl->dl_tensor.device.device_type, dl->dl_tensor.dtype.code,
                       dl->dl_tensor.dtype.bits, dl->dl_tensor.ndim,
                       dl->dl_tensor.strides ? "explicit" : "null (dense)");
            }
            if (dl->dl_tensor.device.device_type != OMNI_DLPACK_CPU) {
                fprintf(stderr, "dlpack device is not the CPU\n");
                return 1;
            }
            if (dl->deleter == NULL) {
                fprintf(stderr, "dlpack tensor has no deleter\n");
                return 1;
            }
            /* Release the OMNI handle first: the DLPack object outliving every
             * OMNI handle is the design, not an accident of ordering. */
            omni_tensor_release(t);
            t = NULL;
            dl->deleter(dl);
        } else if (st == OMNI_INDETERMINATE) {
            if (dlpack_refused++ == 0)
                printf("    dlpack refused: %s\n", omni_last_error());
        } else {
            return fail("dlpack", st);
        }
        omni_tensor_release(t);
    }

    /* Negotiation. NULL capabilities means C0, the floor; a model that needs
     * the expression feature is infeasible there and says so. */
    omni_plan *plan = NULL;
    st = omni_model_resolve(model, NULL, OMNI_OBJ_MIN_MEMORY, &plan);
    if (st != OMNI_OK && st != OMNI_EINFEASIBLE) return fail("resolve", st);
    const char *pj = NULL;
    size_t pj_len = 0;
    if (omni_plan_json(plan, &pj, &pj_len) != OMNI_OK) return fail("plan json", st);
    printf("plan against C0: %s, %llu resident, %llu read, %zu bytes of json\n",
           omni_plan_feasible(plan) ? "feasible" : "infeasible",
           (unsigned long long)omni_plan_resident_bytes(plan),
           (unsigned long long)omni_plan_read_bytes(plan), pj_len);
    omni_plan_free(plan);

    /* Errors are statuses, not crashes. */
    omni_tensor *missing = NULL;
    st = omni_model_tensor(model, "no.such.tensor", &missing);
    if (st != OMNI_EUSAGE) {
        fprintf(stderr, "an unknown tensor name should be a usage error, got %d\n", st);
        return 1;
    }
    printf("unknown name: %s\n", omni_last_error());

    omni_model_free(model);
    omni_store_close(store);

    printf("mapped=%d computed=%d dlpack_ok=%d dlpack_refused=%d\n",
           mapped_seen, indeterminate_seen, dlpack_ok, dlpack_refused);
    printf("the C ABI read a container end to end\n");
    return 0;
}
