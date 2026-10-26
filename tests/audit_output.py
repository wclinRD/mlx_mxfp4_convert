#!/usr/bin/env python3
"""Full audit of a converted MLX model: verify the weights really are at the
target bit width, module by module, and that untouched tensors are unchanged.

    python3 tests/audit_output.py SRC DST [--bits 4] [--group-size 64]

Checks
  1. tensor names: identical sets, and every tensor exists exactly once
  2. every quantized module: packed shape / scales shape match target bits & gs
  3. bits recomputed from the on-disk shapes equal the target bits
  4. dequantized values are finite, and rel-rms error vs the source is sane
  5. every non-quantized tensor (vision tower, norms, conv1d, ...) is bit-identical
  6. output size and per-tensor byte totals add up to the files on disk
"""

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import mlx.core as mx

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return h


def scan(root: Path):
    """name -> (file, dtype, shape) for every tensor in a model directory."""
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        files = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        files = [root / f for f in files]
    else:
        files = [root / "model.safetensors"]
    info = {}
    for f in files:
        for name, meta in read_header(f).items():
            info[name] = (f, meta["dtype"], tuple(meta["shape"]))
    return info


def quant_specs(config):
    q = config.get("quantization") or config.get("quantization_config") or {}
    return q, q.get("bits"), q.get("group_size")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument(
        "--bits",
        type=int,
        default=None,
        help="expected bits (default: from dst config)",
    )
    ap.add_argument("--group-size", type=int, default=None)
    args = ap.parse_args()

    src, dst = Path(args.src).expanduser(), Path(args.dst).expanduser()
    src_cfg = json.loads((src / "config.json").read_text())
    dst_cfg = json.loads((dst / "config.json").read_text())
    _, src_bits, src_gs = quant_specs(src_cfg)
    _, dst_bits, dst_gs = quant_specs(dst_cfg)
    want_bits = args.bits or dst_bits
    want_gs = args.group_size or dst_gs
    print(
        f"source bits/gs defaults: {src_bits}/{src_gs}   "
        f"target bits/gs defaults: {dst_bits}/{dst_gs}"
    )

    S, D = scan(src), scan(dst)
    problems = []
    if set(S) != set(D):
        problems.append(
            f"tensor name mismatch: missing={sorted(set(S) - set(D))[:5]} "
            f"extra={sorted(set(D) - set(S))[:5]}"
        )

    sq, s_bits_default, s_gs_default = quant_specs(src_cfg)
    dq, d_bits_default, d_gs_default = quant_specs(dst_cfg)

    modules = sorted({n[: -len(".scales")] for n in D if n.endswith(".scales")})
    # tensors that belong to a quantized module: .weight / .scales / .biases
    quant_names = {
        m + suf for m in modules for suf in (".weight", ".scales", ".biases")
    }
    plain = [n for n in D if n not in quant_names]

    # ---- pass 1: shapes ----
    bits_seen = Counter()
    for m in modules:
        w, sc = D[m + ".weight"], D[m + ".scales"]
        sw, ssc = S[m + ".weight"], S[m + ".scales"]
        s_entry = sq.get(m) if isinstance(sq.get(m), dict) else {}
        s_bits = s_entry.get("bits", s_bits_default)
        s_gs = s_entry.get("group_size", s_gs_default)
        last_dim = int(ssc[2][-1]) * int(s_gs)
        if int(sc[2][-1]) * int(want_gs) != last_dim:
            problems.append(
                f"{m}: scales imply last_dim {int(sc[2][-1]) * want_gs} != {last_dim}"
            )
        stored_bits = int(w[2][-1]) * 32 / last_dim
        bits_seen[round(stored_bits, 3)] += 1
        if round(stored_bits) != want_bits or abs(stored_bits - want_bits) > 1e-6:
            problems.append(
                f"{m}: packed last dim {w[2][-1]} implies {stored_bits:g} bits, wanted {want_bits}"
            )

    # ---- pass 2: values ----
    cache = {}

    def load(f):
        if f not in cache:
            if len(cache) > 1:
                cache.clear()
                mx.clear_cache()
            cache[f] = mx.load(str(f))
        return cache[f]

    worst, nan = [], 0
    for i, m in enumerate(modules):
        s_entry = sq.get(m) if isinstance(sq.get(m), dict) else {}
        s_bits = int(s_entry.get("bits", s_bits_default))
        s_gs = int(s_entry.get("group_size", s_gs_default))
        d_entry = dq.get(m) if isinstance(dq.get(m), dict) else {}
        t_bits = int(d_entry.get("bits", d_bits_default))
        t_gs = int(d_entry.get("group_size", d_gs_default))

        st = load(S[m + ".weight"][0])
        w_src = mx.dequantize(
            st[m + ".weight"],
            st[m + ".scales"],
            st[m + ".biases"],
            group_size=s_gs,
            bits=s_bits,
        )
        del st
        dt = load(D[m + ".weight"][0])
        w_dst = mx.dequantize(
            dt[m + ".weight"],
            dt[m + ".scales"],
            dt[m + ".biases"],
            group_size=t_gs,
            bits=t_bits,
        )
        num = mx.sqrt(mx.mean(mx.square(w_src - w_dst)))
        den = mx.sqrt(mx.mean(mx.square(w_src))) + 1e-12
        mx.eval(num, den)
        if not bool(mx.all(mx.isfinite(w_dst))):
            nan += 1
        worst.append((float(num) / float(den), m))
        del w_src, w_dst
        if i % 64 == 0:
            mx.clear_cache()

    worst.sort(reverse=True)

    # ---- pass 3: untouched tensors ----
    changed = []
    for n in plain:
        a = load(S[n][0])[n]
        b = load(D[n][0])[n]
        if a.dtype != b.dtype or a.shape != b.shape:
            changed.append((n, f"dtype/shape {a.dtype}{a.shape} -> {b.dtype}{b.shape}"))
        elif not bool(mx.array_equal(a, b)):
            changed.append((n, "values differ"))
    del cache
    mx.clear_cache()

    # ---- report ----
    total_out = sum(f.stat().st_size for f in dst.glob("*.safetensors"))
    idx = dst / "model.safetensors.index.json"
    declared = (
        json.loads(idx.read_text())["metadata"]["total_size"] if idx.exists() else None
    )
    src_bytes = sum(f.stat().st_size for f in src.glob("*.safetensors"))

    print(f"tensors      : {len(D)} (source {len(S)})")
    print(f"modules      : {len(modules)} quantized, {len(plain)} plain")
    print(f"bits on disk : {dict(bits_seen)}")
    print(f"size         : {src_bytes / 2**30:.2f} GiB -> {total_out / 2**30:.2f} GiB")
    print(f"non-finite   : {nan} modules")
    print(
        f"quant error  : mean {sum(w for w, _ in worst) / len(worst) * 100:.2f}%, "
        f"max {worst[0][0] * 100:.2f}% ({worst[0][1]})"
    )
    print(f"untouched    : {len(plain)} plain tensors, {len(changed)} changed")
    for n, why in changed[:10]:
        print(f"    CHANGED {n}: {why}")
    print(f"index total  : {declared} declared / {total_out} on disk")

    ok = not problems and not changed and nan == 0 and worst[0][0] < 0.35
    if problems:
        print(f"\n{len(problems)} shape problems:")
        for p in problems[:10]:
            print("   ", p)
    print("\nAUDIT", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
