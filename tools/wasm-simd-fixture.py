#!/usr/bin/env python3
"""Generate WebAssembly SIMD test modules and what a real engine makes of them.

The plugin host in `wasm.rs` implements §11.6's fixed-width SIMD subset — some
230 instructions, each with exact lane semantics. A table that size cannot be
checked by reading it: the saturating forms, the two different NaN rules for
`min`/`max` against `pmin`/`pmax`, the rounding Q15 multiply, the narrowing
saturations and the `trunc_sat` clamps are all places where a plausible
implementation is wrong in a way its own tests would not notice.

So the check is differential against an independent engine. For each case this
writes a `.wasm` module that computes the operation and stores the 16-byte result
at address 0, runs it under `wasmtime`, and records what came out. `wasm.rs`'s
vector test then runs the same module and must produce the same bytes.

    tools/wasm-simd-fixture.py <out-dir>

Writes `<name>.wasm` per case and `manifest.txt` listing each case with the
expected result bytes in hex. Requires the `wasmtime` package; like the rest of
`tools/` it lives outside the crate's zero-dependency boundary.
"""

import os
import sys

# Two vectors of interesting bytes: zeros, ones, signs, saturation edges, and
# values that differ per lane so a lane-order mistake cannot cancel out.
A = "0 1 127 128 129 255 2 254 64 192 63 191 32 224 16 240"
B = "255 1 1 255 127 128 3 3 200 100 1 2 15 240 128 127"
# Float lanes chosen for the rules that are easy to get wrong: both zeros, both
# infinities, a NaN, and ordinary values.
FA32 = "1.5 -0.0 nan 3.25"
FB32 = "-1.5 0.0 2.0 inf"
FA64 = "1.5 -0.0"
FB64 = "-1.5 nan"

# (name, wat body). Each body leaves a v128 on the stack; the wrapper stores it.
UNARY_I = [
    ("i8x16_abs", "i8x16.abs"), ("i8x16_neg", "i8x16.neg"),
    ("i8x16_popcnt", "i8x16.popcnt"),
    ("i16x8_abs", "i16x8.abs"), ("i16x8_neg", "i16x8.neg"),
    ("i32x4_abs", "i32x4.abs"), ("i32x4_neg", "i32x4.neg"),
    ("i64x2_abs", "i64x2.abs"), ("i64x2_neg", "i64x2.neg"),
    ("v128_not", "v128.not"),
    ("i16x8_extadd_pairwise_i8x16_s", "i16x8.extadd_pairwise_i8x16_s"),
    ("i16x8_extadd_pairwise_i8x16_u", "i16x8.extadd_pairwise_i8x16_u"),
    ("i32x4_extadd_pairwise_i16x8_s", "i32x4.extadd_pairwise_i16x8_s"),
    ("i32x4_extadd_pairwise_i16x8_u", "i32x4.extadd_pairwise_i16x8_u"),
    ("i16x8_extend_low_i8x16_s", "i16x8.extend_low_i8x16_s"),
    ("i16x8_extend_high_i8x16_s", "i16x8.extend_high_i8x16_s"),
    ("i16x8_extend_low_i8x16_u", "i16x8.extend_low_i8x16_u"),
    ("i16x8_extend_high_i8x16_u", "i16x8.extend_high_i8x16_u"),
    ("i32x4_extend_low_i16x8_s", "i32x4.extend_low_i16x8_s"),
    ("i32x4_extend_high_i16x8_s", "i32x4.extend_high_i16x8_s"),
    ("i32x4_extend_low_i16x8_u", "i32x4.extend_low_i16x8_u"),
    ("i32x4_extend_high_i16x8_u", "i32x4.extend_high_i16x8_u"),
    ("i64x2_extend_low_i32x4_s", "i64x2.extend_low_i32x4_s"),
    ("i64x2_extend_high_i32x4_s", "i64x2.extend_high_i32x4_s"),
    ("i64x2_extend_low_i32x4_u", "i64x2.extend_low_i32x4_u"),
    ("i64x2_extend_high_i32x4_u", "i64x2.extend_high_i32x4_u"),
]

BINARY_I = [
    ("i8x16_eq", "i8x16.eq"), ("i8x16_ne", "i8x16.ne"),
    ("i8x16_lt_s", "i8x16.lt_s"), ("i8x16_lt_u", "i8x16.lt_u"),
    ("i8x16_gt_s", "i8x16.gt_s"), ("i8x16_gt_u", "i8x16.gt_u"),
    ("i8x16_le_s", "i8x16.le_s"), ("i8x16_le_u", "i8x16.le_u"),
    ("i8x16_ge_s", "i8x16.ge_s"), ("i8x16_ge_u", "i8x16.ge_u"),
    ("i16x8_eq", "i16x8.eq"), ("i16x8_ne", "i16x8.ne"),
    ("i16x8_lt_s", "i16x8.lt_s"), ("i16x8_lt_u", "i16x8.lt_u"),
    ("i16x8_gt_s", "i16x8.gt_s"), ("i16x8_gt_u", "i16x8.gt_u"),
    ("i16x8_le_s", "i16x8.le_s"), ("i16x8_le_u", "i16x8.le_u"),
    ("i16x8_ge_s", "i16x8.ge_s"), ("i16x8_ge_u", "i16x8.ge_u"),
    ("i32x4_eq", "i32x4.eq"), ("i32x4_ne", "i32x4.ne"),
    ("i32x4_lt_s", "i32x4.lt_s"), ("i32x4_lt_u", "i32x4.lt_u"),
    ("i32x4_gt_s", "i32x4.gt_s"), ("i32x4_gt_u", "i32x4.gt_u"),
    ("i32x4_le_s", "i32x4.le_s"), ("i32x4_le_u", "i32x4.le_u"),
    ("i32x4_ge_s", "i32x4.ge_s"), ("i32x4_ge_u", "i32x4.ge_u"),
    ("i64x2_eq", "i64x2.eq"), ("i64x2_ne", "i64x2.ne"),
    ("i64x2_lt_s", "i64x2.lt_s"), ("i64x2_gt_s", "i64x2.gt_s"),
    ("i64x2_le_s", "i64x2.le_s"), ("i64x2_ge_s", "i64x2.ge_s"),
    ("v128_and", "v128.and"), ("v128_andnot", "v128.andnot"),
    ("v128_or", "v128.or"), ("v128_xor", "v128.xor"),
    ("i8x16_narrow_i16x8_s", "i8x16.narrow_i16x8_s"),
    ("i8x16_narrow_i16x8_u", "i8x16.narrow_i16x8_u"),
    ("i16x8_narrow_i32x4_s", "i16x8.narrow_i32x4_s"),
    ("i16x8_narrow_i32x4_u", "i16x8.narrow_i32x4_u"),
    ("i8x16_add", "i8x16.add"), ("i8x16_add_sat_s", "i8x16.add_sat_s"),
    ("i8x16_add_sat_u", "i8x16.add_sat_u"), ("i8x16_sub", "i8x16.sub"),
    ("i8x16_sub_sat_s", "i8x16.sub_sat_s"), ("i8x16_sub_sat_u", "i8x16.sub_sat_u"),
    ("i8x16_min_s", "i8x16.min_s"), ("i8x16_min_u", "i8x16.min_u"),
    ("i8x16_max_s", "i8x16.max_s"), ("i8x16_max_u", "i8x16.max_u"),
    ("i8x16_avgr_u", "i8x16.avgr_u"),
    ("i16x8_add", "i16x8.add"), ("i16x8_add_sat_s", "i16x8.add_sat_s"),
    ("i16x8_add_sat_u", "i16x8.add_sat_u"), ("i16x8_sub", "i16x8.sub"),
    ("i16x8_sub_sat_s", "i16x8.sub_sat_s"), ("i16x8_sub_sat_u", "i16x8.sub_sat_u"),
    ("i16x8_mul", "i16x8.mul"),
    ("i16x8_min_s", "i16x8.min_s"), ("i16x8_min_u", "i16x8.min_u"),
    ("i16x8_max_s", "i16x8.max_s"), ("i16x8_max_u", "i16x8.max_u"),
    ("i16x8_avgr_u", "i16x8.avgr_u"),
    ("i16x8_q15mulr_sat_s", "i16x8.q15mulr_sat_s"),
    ("i32x4_add", "i32x4.add"), ("i32x4_sub", "i32x4.sub"),
    ("i32x4_mul", "i32x4.mul"),
    ("i32x4_min_s", "i32x4.min_s"), ("i32x4_min_u", "i32x4.min_u"),
    ("i32x4_max_s", "i32x4.max_s"), ("i32x4_max_u", "i32x4.max_u"),
    ("i32x4_dot_i16x8_s", "i32x4.dot_i16x8_s"),
    ("i64x2_add", "i64x2.add"), ("i64x2_sub", "i64x2.sub"),
    ("i64x2_mul", "i64x2.mul"),
    ("i16x8_extmul_low_i8x16_s", "i16x8.extmul_low_i8x16_s"),
    ("i16x8_extmul_high_i8x16_s", "i16x8.extmul_high_i8x16_s"),
    ("i16x8_extmul_low_i8x16_u", "i16x8.extmul_low_i8x16_u"),
    ("i16x8_extmul_high_i8x16_u", "i16x8.extmul_high_i8x16_u"),
    ("i32x4_extmul_low_i16x8_s", "i32x4.extmul_low_i16x8_s"),
    ("i32x4_extmul_high_i16x8_s", "i32x4.extmul_high_i16x8_s"),
    ("i32x4_extmul_low_i16x8_u", "i32x4.extmul_low_i16x8_u"),
    ("i32x4_extmul_high_i16x8_u", "i32x4.extmul_high_i16x8_u"),
    ("i64x2_extmul_low_i32x4_s", "i64x2.extmul_low_i32x4_s"),
    ("i64x2_extmul_high_i32x4_s", "i64x2.extmul_high_i32x4_s"),
    ("i64x2_extmul_low_i32x4_u", "i64x2.extmul_low_i32x4_u"),
    ("i64x2_extmul_high_i32x4_u", "i64x2.extmul_high_i32x4_u"),
    ("i8x16_swizzle", "i8x16.swizzle"),
]

# Shifts take a vector and an i32; the count is exercised past the lane width so
# the modulo rule is checked rather than assumed.
SHIFTS = [
    ("i8x16_shl", "i8x16.shl", 3), ("i8x16_shr_s", "i8x16.shr_s", 3),
    ("i8x16_shr_u", "i8x16.shr_u", 3), ("i8x16_shl_wrap", "i8x16.shl", 11),
    ("i16x8_shl", "i16x8.shl", 5), ("i16x8_shr_s", "i16x8.shr_s", 5),
    ("i16x8_shr_u", "i16x8.shr_u", 5), ("i16x8_shr_s_wrap", "i16x8.shr_s", 21),
    ("i32x4_shl", "i32x4.shl", 9), ("i32x4_shr_s", "i32x4.shr_s", 9),
    ("i32x4_shr_u", "i32x4.shr_u", 9), ("i32x4_shl_wrap", "i32x4.shl", 41),
    ("i64x2_shl", "i64x2.shl", 17), ("i64x2_shr_s", "i64x2.shr_s", 17),
    ("i64x2_shr_u", "i64x2.shr_u", 17), ("i64x2_shr_u_wrap", "i64x2.shr_u", 81),
]

UNARY_F32 = [
    ("f32x4_abs", "f32x4.abs"), ("f32x4_neg", "f32x4.neg"),
    ("f32x4_sqrt", "f32x4.sqrt"), ("f32x4_ceil", "f32x4.ceil"),
    ("f32x4_floor", "f32x4.floor"), ("f32x4_trunc", "f32x4.trunc"),
    ("f32x4_nearest", "f32x4.nearest"),
    ("i32x4_trunc_sat_f32x4_s", "i32x4.trunc_sat_f32x4_s"),
    ("i32x4_trunc_sat_f32x4_u", "i32x4.trunc_sat_f32x4_u"),
    ("f64x2_promote_low_f32x4", "f64x2.promote_low_f32x4"),
]
BINARY_F32 = [
    ("f32x4_eq", "f32x4.eq"), ("f32x4_ne", "f32x4.ne"),
    ("f32x4_lt", "f32x4.lt"), ("f32x4_gt", "f32x4.gt"),
    ("f32x4_le", "f32x4.le"), ("f32x4_ge", "f32x4.ge"),
    ("f32x4_add", "f32x4.add"), ("f32x4_sub", "f32x4.sub"),
    ("f32x4_mul", "f32x4.mul"), ("f32x4_div", "f32x4.div"),
    ("f32x4_min", "f32x4.min"), ("f32x4_max", "f32x4.max"),
    ("f32x4_pmin", "f32x4.pmin"), ("f32x4_pmax", "f32x4.pmax"),
]
UNARY_F64 = [
    ("f64x2_abs", "f64x2.abs"), ("f64x2_neg", "f64x2.neg"),
    ("f64x2_sqrt", "f64x2.sqrt"), ("f64x2_ceil", "f64x2.ceil"),
    ("f64x2_floor", "f64x2.floor"), ("f64x2_trunc", "f64x2.trunc"),
    ("f64x2_nearest", "f64x2.nearest"),
    ("f32x4_demote_f64x2_zero", "f32x4.demote_f64x2_zero"),
    ("i32x4_trunc_sat_f64x2_s_zero", "i32x4.trunc_sat_f64x2_s_zero"),
    ("i32x4_trunc_sat_f64x2_u_zero", "i32x4.trunc_sat_f64x2_u_zero"),
]
BINARY_F64 = [
    ("f64x2_eq", "f64x2.eq"), ("f64x2_ne", "f64x2.ne"),
    ("f64x2_lt", "f64x2.lt"), ("f64x2_gt", "f64x2.gt"),
    ("f64x2_le", "f64x2.le"), ("f64x2_ge", "f64x2.ge"),
    ("f64x2_add", "f64x2.add"), ("f64x2_sub", "f64x2.sub"),
    ("f64x2_mul", "f64x2.mul"), ("f64x2_div", "f64x2.div"),
    ("f64x2_min", "f64x2.min"), ("f64x2_max", "f64x2.max"),
    ("f64x2_pmin", "f64x2.pmin"), ("f64x2_pmax", "f64x2.pmax"),
]
# Integer→float conversions read the integer vector.
CONV_I = [
    ("f32x4_convert_i32x4_s", "f32x4.convert_i32x4_s"),
    ("f32x4_convert_i32x4_u", "f32x4.convert_i32x4_u"),
    ("f64x2_convert_low_i32x4_s", "f64x2.convert_low_i32x4_s"),
    ("f64x2_convert_low_i32x4_u", "f64x2.convert_low_i32x4_u"),
]


def module(body):
    """Wrap a body that leaves one v128 on the stack into a storing module."""
    return f"""(module
  (memory (export "mem") 1)
  (func (export "run")
    i32.const 0
    {body}
    v128.store))
"""


def cases():
    out = {}
    ia = f"v128.const i8x16 {A}"
    ib = f"v128.const i8x16 {B}"
    fa32 = f"v128.const f32x4 {FA32}"
    fb32 = f"v128.const f32x4 {FB32}"
    fa64 = f"v128.const f64x2 {FA64}"
    fb64 = f"v128.const f64x2 {FB64}"

    for name, op in UNARY_I:
        out[name] = module(f"{ia}\n    {op}")
    for name, op in BINARY_I:
        out[name] = module(f"{ia}\n    {ib}\n    {op}")
    for name, op, n in SHIFTS:
        out[name] = module(f"{ia}\n    i32.const {n}\n    {op}")
    for name, op in UNARY_F32:
        out[name] = module(f"{fa32}\n    {op}")
    for name, op in BINARY_F32:
        out[name] = module(f"{fa32}\n    {fb32}\n    {op}")
    for name, op in UNARY_F64:
        out[name] = module(f"{fa64}\n    {op}")
    for name, op in BINARY_F64:
        out[name] = module(f"{fa64}\n    {fb64}\n    {op}")
    for name, op in CONV_I:
        out[name] = module(f"{ia}\n    {op}")

    # bitselect, shuffle and the splats, which have shapes of their own.
    out["v128_bitselect"] = module(f"{ia}\n    {ib}\n    "
                                   f"v128.const i8x16 255 0 255 0 255 0 255 0 "
                                   f"15 240 15 240 1 2 3 4\n    v128.bitselect")
    out["i8x16_shuffle"] = module(
        f"{ia}\n    {ib}\n    i8x16.shuffle "
        "0 17 2 19 4 21 6 23 8 25 10 27 12 29 14 31")
    out["i8x16_splat"] = module("i32.const 0x9f\n    i8x16.splat")
    out["i16x8_splat"] = module("i32.const 0xbeef\n    i16x8.splat")
    out["i32x4_splat"] = module("i32.const 0xdeadbeef\n    i32x4.splat")
    out["i64x2_splat"] = module("i64.const 0x0123456789abcdef\n    i64x2.splat")
    out["f32x4_splat"] = module("f32.const -2.5\n    f32x4.splat")
    out["f64x2_splat"] = module("f64.const 1.25\n    f64x2.splat")

    # replace_lane, which writes into a vector rather than reading from one.
    out["i8x16_replace_lane"] = module(f"{ia}\n    i32.const 7\n    "
                                       "i8x16.replace_lane 5")
    out["i16x8_replace_lane"] = module(f"{ia}\n    i32.const 4660\n    "
                                       "i16x8.replace_lane 3")
    out["i32x4_replace_lane"] = module(f"{ia}\n    i32.const -2\n    "
                                       "i32x4.replace_lane 2")
    out["i64x2_replace_lane"] = module(f"{ia}\n    i64.const -3\n    "
                                       "i64x2.replace_lane 1")
    out["f32x4_replace_lane"] = module(f"{fa32}\n    f32.const 9.5\n    "
                                       "f32x4.replace_lane 0")
    out["f64x2_replace_lane"] = module(f"{fa64}\n    f64.const -8.25\n    "
                                       "f64x2.replace_lane 1")

    # The loads and stores, against bytes the module puts in memory itself. The
    # data segment is at 64 so a store to 0 cannot overwrite the source.
    def loader(op, extra=""):
        return f"""(module
  (memory (export "mem") 1)
  (data (i32.const 64) "\\00\\01\\7f\\80\\81\\ff\\02\\fe\\40\\c0\\3f\\bf\\20\\e0\\10\\f0")
  (func (export "run")
    i32.const 0
    i32.const 64
    {op}{extra}
    v128.store))
"""
    for name, op in [
        ("v128_load", "v128.load"),
        ("v128_load8x8_s", "v128.load8x8_s"), ("v128_load8x8_u", "v128.load8x8_u"),
        ("v128_load16x4_s", "v128.load16x4_s"), ("v128_load16x4_u", "v128.load16x4_u"),
        ("v128_load32x2_s", "v128.load32x2_s"), ("v128_load32x2_u", "v128.load32x2_u"),
        ("v128_load8_splat", "v128.load8_splat"),
        ("v128_load16_splat", "v128.load16_splat"),
        ("v128_load32_splat", "v128.load32_splat"),
        ("v128_load64_splat", "v128.load64_splat"),
        ("v128_load32_zero", "v128.load32_zero"),
        ("v128_load64_zero", "v128.load64_zero"),
    ]:
        out[name] = loader(op)
    # load_lane merges into an existing vector.
    for name, op in [
        ("v128_load8_lane", "v128.load8_lane 3"),
        ("v128_load16_lane", "v128.load16_lane 2"),
        ("v128_load32_lane", "v128.load32_lane 1"),
        ("v128_load64_lane", "v128.load64_lane 1"),
    ]:
        out[name] = f"""(module
  (memory (export "mem") 1)
  (data (i32.const 64) "\\00\\01\\7f\\80\\81\\ff\\02\\fe\\40\\c0\\3f\\bf\\20\\e0\\10\\f0")
  (func (export "run")
    i32.const 0
    i32.const 64
    v128.const i8x16 {A}
    {op}
    v128.store))
"""
    # store_lane writes one lane and leaves the rest of memory alone, so the
    # whole 16 bytes at 0 are the answer.
    for name, op in [
        ("v128_store8_lane", "v128.store8_lane 5"),
        ("v128_store16_lane", "v128.store16_lane 3"),
        ("v128_store32_lane", "v128.store32_lane 2"),
        ("v128_store64_lane", "v128.store64_lane 1"),
    ]:
        out[name] = f"""(module
  (memory (export "mem") 1)
  (func (export "run")
    i32.const 0
    v128.const i8x16 {B}
    v128.store
    i32.const 0
    v128.const i8x16 {A}
    {op}))
"""
    # The boolean reductions return an i32 rather than a vector, so they are
    # stored as one i32 lane and compared the same way.
    for name, op in [
        ("v128_any_true", "v128.any_true"),
        ("i8x16_all_true", "i8x16.all_true"),
        ("i16x8_all_true", "i16x8.all_true"),
        ("i32x4_all_true", "i32x4.all_true"),
        ("i64x2_all_true", "i64x2.all_true"),
        ("i8x16_bitmask", "i8x16.bitmask"),
        ("i16x8_bitmask", "i16x8.bitmask"),
        ("i32x4_bitmask", "i32x4.bitmask"),
        ("i64x2_bitmask", "i64x2.bitmask"),
    ]:
        out[name] = f"""(module
  (memory (export "mem") 1)
  (func (export "run")
    i32.const 0
    v128.const i8x16 {A}
    {op}
    i32.store))
"""
    # extract_lane likewise.
    for name, op, st in [
        ("i8x16_extract_lane_s", "i8x16.extract_lane_s 3", "i32.store"),
        ("i8x16_extract_lane_u", "i8x16.extract_lane_u 3", "i32.store"),
        ("i16x8_extract_lane_s", "i16x8.extract_lane_s 2", "i32.store"),
        ("i16x8_extract_lane_u", "i16x8.extract_lane_u 2", "i32.store"),
        ("i32x4_extract_lane", "i32x4.extract_lane 1", "i32.store"),
        ("i64x2_extract_lane", "i64x2.extract_lane 1", "i64.store"),
    ]:
        out[name] = f"""(module
  (memory (export "mem") 1)
  (func (export "run")
    i32.const 0
    v128.const i8x16 {A}
    {op}
    {st}))
"""
    out["f32x4_extract_lane"] = f"""(module
  (memory (export "mem") 1)
  (func (export "run")
    i32.const 0
    v128.const f32x4 {FA32}
    f32x4.extract_lane 2
    f32.store))
"""
    out["f64x2_extract_lane"] = f"""(module
  (memory (export "mem") 1)
  (func (export "run")
    i32.const 0
    v128.const f64x2 {FA64}
    f64x2.extract_lane 1
    f64.store))
"""
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = sys.argv[1]
    try:
        import wasmtime
    except ImportError:
        print("needs the `wasmtime` package", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)

    engine = wasmtime.Engine()
    lines = []
    for name, wat in sorted(cases().items()):
        try:
            binary = wasmtime.wat2wasm(wat)
        except Exception as e:  # noqa: BLE001 - report which case, then stop
            print(f"{name}: wat2wasm failed: {e}", file=sys.stderr)
            return 1
        store = wasmtime.Store(engine)
        module_ = wasmtime.Module(engine, wat)
        inst = wasmtime.Instance(store, module_, [])
        inst.exports(store)["run"](store)
        mem = inst.exports(store)["mem"]
        got = bytes(mem.read(store, 0, 16))
        open(os.path.join(out, f"{name}.wasm"), "wb").write(binary)
        lines.append(f"{name}\t{got.hex()}")

    open(os.path.join(out, "manifest.txt"), "w").write("\n".join(lines) + "\n")
    print(f"{out}: {len(lines)} SIMD cases from wasmtime "
          f"{getattr(wasmtime, '__version__', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
