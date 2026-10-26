#!/usr/bin/env python3
"""
conv.py - re-quantize MLX models (e.g. 6-bit -> 4-bit) at the safetensors level,
keeping the multimodal / vision tower untouched.

Why this works on an already-quantized model
--------------------------------------------
An MLX quantized module is stored as three tensors:

    <module>.weight   uint32  packed integers
    <module>.scales   bf16    one scale per group of weights
    <module>.biases   bf16    one bias per group of weights

plus `config.json["quantization"]` describing bits/group_size per module.
Going 6-bit -> 4-bit is therefore: dequantize the module back to bf16 using its
own original (bits, group_size), then quantize it again at the target
(bits, group_size). Every tensor that is not quantized -- in particular the whole
`vision_tower` -- is copied verbatim with the same dtype and the same bytes, so
image/video inputs behave exactly as before.

Usage
-----
    python conv.py SRC DST                     # 6-bit -> 4-bit, group size 64
    python conv.py SRC DST --bits 4 --group-size 64
    python conv.py SRC DST --vision-bits 8     # also quantize the vision tower
    python conv.py SRC DST --dry-run           # print the plan, convert nothing

SRC is a directory holding config.json and either model.safetensors or shards
plus model.safetensors.index.json. DST must not exist (unless --overwrite).
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from collections import Counter, OrderedDict
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

W_SUF, S_SUF, B_SUF = ".weight", ".scales", ".biases"

# written by us, not copied from the source
REGENERATED = {"config.json", "model.safetensors.index.json", ".DS_Store"}

VISION_PREFIXES = ("vision_tower.", "vision_model.", "vision.", "visual.")


# --------------------------------------------------------------------------- #
# safetensors helpers
# --------------------------------------------------------------------------- #
def read_header(path: Path) -> "OrderedDict[str, dict]":
    """Read only the JSON header of a safetensors file."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return OrderedDict(header)


def nbytes(dtype: str, shape) -> int:
    total = 1
    for d in shape:
        total *= int(d)
    return total * DTYPE_BYTES[dtype]


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f}{unit}"
        n /= 1024.0
    return f"{n:,.1f}PB"


def parse_size(text) -> int:
    text = str(text).strip().upper()
    mult = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if text and text[-1] in mult:
        return int(float(text[:-1]) * mult[text[-1]])
    return int(text)


def shard_files(model_dir: Path):
    """Return the ordered list of safetensors shards of a model directory."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        files = list(OrderedDict.fromkeys(index["weight_map"].values()))
        return [model_dir / f for f in files]
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors found in {model_dir}")
    return shards


def is_vision(name: str) -> bool:
    return name.startswith(VISION_PREFIXES)


def pick_group_size(last_dim: int, bits: int, preferred: int):
    """Largest usable group size for `last_dim` (MLX needs group_size*bits % 32 == 0)."""
    for gs in (preferred, 64, 32, 16, 8):
        if gs and last_dim % gs == 0 and (gs * bits) % 32 == 0:
            return gs
    return None


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def build_plan(
    model_dir: Path,
    bits: int,
    group_size: int,
    vision_bits,
    donor=None,
    donor_shapes=None,
):
    """Read every header and decide, for each tensor, what the output looks like.

    When `donor` is set, any vision tensor is copied verbatim from the donor
    model instead of being re-quantized (used to repair a corrupted vision tower).
    `donor_shapes` maps full tensor name -> shape in the donor file; it is used to
    verify that each donor vision tensor matches the source tensor's shape.
    """
    config = json.loads((model_dir / "config.json").read_text())
    files = shard_files(model_dir)

    where = {}  # tensor name -> (shard path, dtype, shape)
    order = []  # [(shard path, [tensor names in file order])]
    for f in files:
        header = read_header(f)
        names = []
        for name, meta in header.items():
            where[name] = (f, meta["dtype"], tuple(int(d) for d in meta["shape"]))
            names.append(name)
        order.append((f, names))

    q = config.get("quantization") or config.get("quantization_config") or {}
    default_bits, default_gs = q.get("bits"), q.get("group_size")

    # source spec of every quantized module (config entry, else inferred)
    src_spec = {}
    for name in where:
        if not name.endswith(S_SUF):
            continue
        module = name[: -len(S_SUF)]
        w_name, b_name = module + W_SUF, module + B_SUF
        if w_name not in where:
            continue
        entry = q.get(module) if isinstance(q.get(module), dict) else {}
        s_bits = entry.get("bits", default_bits)
        s_gs = entry.get("group_size", default_gs) or 32
        s_shape = where[name][2]
        last_dim = int(s_shape[-1]) * int(s_gs)
        if s_bits is None:  # infer from packed size
            packed_bits = where[w_name][2][-1] * DTYPE_BYTES[where[w_name][1]] * 8
            s_bits = max(1, round(packed_bits / last_dim))
        src_spec[module] = {
            "bits": int(s_bits),
            "group_size": int(s_gs),
            "last_dim": last_dim,
            "weight": w_name,
            "scales": name,
            "biases": b_name if b_name in where else None,
        }

    units, stats = [], Counter()
    handled = set()
    for f, names in order:
        file_units = []
        for name in names:
            if name in handled:
                continue
            module = None
            for suffix in (S_SUF, W_SUF, B_SUF):
                if name.endswith(suffix):
                    candidate = name[: -len(suffix)]
                    if candidate in src_spec:
                        module = candidate
                    break
            if module in src_spec:
                spec = src_spec[module]
                src_names = [spec["weight"], spec["scales"]] + (
                    [spec["biases"]] if spec["biases"] else []
                )
                for n in src_names:
                    handled.add(n)

                # If a donor is supplied, vision tensors come from the donor verbatim
                # to repair a corrupted own vision tower. Verify each donor tensor's
                # shape matches the source so we never emit wrong-shaped weights.
                use_donor = donor is not None and is_vision(module)
                if use_donor:
                    for n in src_names:
                        donor_shape = tuple(donor_shapes[n])
                        src_shape = where[n][2]
                        if donor_shape != src_shape:
                            raise SystemExit(
                                f"vision tensor {n} shape mismatch between "
                                f"source {src_shape} and donor {donor_shape}"
                            )

                want = bits
                if vision_bits is not None and is_vision(module):
                    want = vision_bits
                elif vision_bits is None and is_vision(module):
                    want = spec["bits"]  # leave vision alone
                gs = pick_group_size(spec["last_dim"], want, group_size)
                if gs is None:  # shape cannot take that group size
                    want, gs = spec["bits"], spec["group_size"]
                copy = want == spec["bits"] and gs == spec["group_size"]
                if copy:
                    stats["kept as is"] += 1
                else:
                    stats[
                        f"{spec['bits']}b/g{spec['group_size']} -> {want}b/g{gs}"
                    ] += 1
                    if is_vision(module):
                        stats["  (vision tower)"] += 1

                w_shape = list(where[spec["weight"]][2])
                if copy or use_donor:
                    out = OrderedDict(
                        (n, (where[n][1], list(where[n][2]))) for n in src_names
                    )
                else:
                    packed_last = spec["last_dim"] * want // 32
                    pw = w_shape[:-1] + [packed_last]
                    ps = [spec["last_dim"] // gs]
                    out = OrderedDict(
                        [
                            (spec["weight"], ("U32", pw)),
                            (spec["scales"], ("BF16", pw[:-1] + ps)),
                        ]
                    )
                    if spec["biases"]:
                        out[spec["biases"]] = ("BF16", pw[:-1] + ps)
                file_units.append(
                    {
                        "kind": "quant",
                        "file": f,
                        "module": module,
                        "src": src_names,
                        "out": out,
                        "spec": spec,
                        "target": {"bits": want, "group_size": gs},
                        "copy": copy or use_donor,
                        # when set, these src names are read from the donor file
                        # (repair a corrupted vision tower) instead of the source
                        "donor_src": src_names if use_donor else None,
                        "bytes": sum(nbytes(d, s) for d, s in out.values()),
                    }
                )
            else:
                handled.add(name)
                dtype, shape = where[name][1], list(where[name][2])
                file_units.append(
                    {
                        "kind": "plain",
                        "file": f,
                        "src": [name],
                        "out": OrderedDict([(name, (dtype, shape))]),
                        "bytes": nbytes(dtype, shape),
                    }
                )
        units.append((f, file_units))

    return config, units, src_spec, stats, where


def assign_shards(units, cap: int):
    shard_of, sizes, current = {}, [], 0
    for _, file_units in units:
        for unit in file_units:
            if not sizes:
                sizes.append(0)
            if current and current + unit["bytes"] > cap:
                sizes.append(0)
                current = 0
            sizes[-1] += unit["bytes"]
            current += unit["bytes"]
            for n in unit["out"]:
                shard_of[n] = len(sizes) - 1
    return shard_of, sizes


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-quantize an MLX model (e.g. 6-bit -> 4-bit); vision-safe."
    )
    ap.add_argument("src", help="source MLX model directory")
    ap.add_argument("dst", help="destination directory (must not exist)")
    ap.add_argument("--bits", type=int, default=4, help="target bits (default: 4)")
    ap.add_argument(
        "--group-size", type=int, default=64, help="target group size (default: 64)"
    )
    ap.add_argument(
        "--vision-bits",
        type=int,
        default=None,
        help="also quantize the vision tower to this bit width "
        "(default: copy vision tensors unchanged)",
    )
    ap.add_argument(
        "--vision-from",
        default=None,
        help="donor MLX model directory; copy vision tensors from it verbatim "
        "to repair a corrupted vision tower (default: use the source's own)",
    )
    ap.add_argument(
        "--shard-size", default="4.5G", help="max output shard size (default: 4.5G)"
    )
    ap.add_argument(
        "--verify-sample",
        type=int,
        default=24,
        help="numerically compare this many modules (0 disables)",
    )
    ap.add_argument("--overwrite", action="store_true", help="replace dst if it exists")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    ap.add_argument(
        "--cache-limit", default="1.5G", help="MLX cache limit (default: 1.5G)"
    )
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not (src / "config.json").exists():
        raise SystemExit(f"config.json not found in {src}")
    # Optional donor for vision tensors: repair a corrupted own vision tower.
    donor = None
    donor_shapes = {}
    if args.vision_from is not None:
        donor = Path(args.vision_from).expanduser().resolve()
        if not (donor / "config.json").exists():
            raise SystemExit(f"vision-from config.json not found in {donor}")
        for f in shard_files(donor):
            loaded = mx.load(str(f))
            for n, arr in loaded.items():
                if is_vision(n):
                    donor_shapes[n] = tuple(int(d) for d in arr.shape)
    if dst.exists() and not args.dry_run:
        if not args.overwrite:
            raise SystemExit(f"destination already exists: {dst} (use --overwrite)")
        print(f"removing existing {dst}")
        shutil.rmtree(dst)

    mx.set_cache_limit(parse_size(args.cache_limit))

    config, units, src_spec, stats, where = build_plan(
        src,
        args.bits,
        args.group_size,
        args.vision_bits,
        donor=donor,
        donor_shapes=donor_shapes,
    )
    shard_of, shard_sizes = assign_shards(units, parse_size(args.shard_size))
    n_shards = len(shard_sizes)
    plain = sum(1 for _, fu in units for u in fu if u["kind"] == "plain")
    src_total = sum(p.stat().st_size for p in shard_files(src))
    out_total = sum(shard_sizes)

    print(f"source  : {src}")
    print(f"target  : {dst}")
    print(
        f"quant   : {args.bits}-bit / group {args.group_size}  "
        + (
            f"| vision tower -> {args.vision_bits}-bit"
            if args.vision_bits
            else ("| vision tower copied from " + donor.name)
            if donor is not None
            else "| vision tower copied unchanged"
        )
    )
    print(f"tensors : {len(src_spec)} quantized modules, {plain} plain tensors")
    for k, v in sorted(stats.items()):
        print(f"          {k}: {v}")
    print(f"shards  : {n_shards} (max ~{human(max(shard_sizes))})")
    print(
        f"size    : {human(src_total)} -> {human(out_total)} "
        f"(x{out_total / src_total:.3f})"
    )
    if args.dry_run:
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    if n_shards == 1:
        out_paths = [dst / "model.safetensors"]
    else:
        out_paths = [
            dst / f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
            for i in range(n_shards)
        ]

    # sample modules for the numeric sanity check
    quant_conv = [
        u for _, fu in units for u in fu if u["kind"] == "quant" and not u["copy"]
    ]
    sample = set()
    if args.verify_sample > 0 and quant_conv:
        stride = max(1, len(quant_conv) // args.verify_sample)
        sample = {u["module"] for u in quant_conv[::stride]}

    errors, copied_vision = [], []
    weight_map, bucket, bucket_bytes = OrderedDict(), {}, 0
    cur_shard, done_bytes, t0 = 0, 0, time.time()

    # Load donor vision tensors ONCE into memory (small: ~1GB) so the write loop
    # can copy them verbatim instead of re-quantizing a corrupted own tower.
    donor_tensors = {}
    if donor is not None:
        for f in shard_files(donor):
            loaded = mx.load(str(f))
            for n, arr in loaded.items():
                if is_vision(n) and n in donor_shapes:
                    donor_tensors[n] = arr

    def flush(idx: int):
        nonlocal bucket, bucket_bytes
        if not bucket:
            return
        mx.save_safetensors(str(out_paths[idx]), bucket)
        for n in bucket:
            weight_map[n] = out_paths[idx].name
        print(f"          wrote {out_paths[idx].name} ({human(bucket_bytes)})")
        bucket, bucket_bytes = {}, 0

    for src_file, file_units in units:
        tensors = mx.load(str(src_file))
        print(
            f"[{done_bytes / src_total * 100:5.1f}%] {src_file.name}: {len(file_units)} units"
        )
        for unit in file_units:
            first = unit["src"][0]
            shard = shard_of[first]
            if shard != cur_shard:
                flush(cur_shard)
                cur_shard = shard

            if unit["kind"] == "plain":
                # When a donor is supplied, vision plain tensors come from the
                # donor verbatim (repair a corrupted own vision tower); other
                # plain tensors still come from source.
                src_map = (
                    donor_tensors
                    if (donor is not None and is_vision(first))
                    else tensors
                )
                arr = src_map[first]
                if is_vision(first):
                    copied_vision.append(first)
                bucket[first] = arr
                bucket_bytes += unit["bytes"]
                continue

            spec, tgt = unit["spec"], unit["target"]
            if unit["copy"]:
                # vision tensors from the donor are copied verbatim (repair a
                # corrupted own vision tower); everything else comes from source.
                src_map = donor_tensors if unit.get("donor_src") else tensors
                for n in unit["src"]:
                    bucket[n] = src_map[n]
                    if is_vision(n):
                        copied_vision.append(n)
            else:
                w = mx.dequantize(
                    tensors[spec["weight"]],
                    tensors[spec["scales"]],
                    tensors[spec["biases"]] if spec["biases"] else None,
                    group_size=spec["group_size"],
                    bits=spec["bits"],
                )
                wq, sc, bi = mx.quantize(
                    w, group_size=tgt["group_size"], bits=tgt["bits"]
                )
                mx.eval(wq, sc, bi)
                if unit["module"] in sample:
                    back = mx.dequantize(
                        wq, sc, bi, group_size=tgt["group_size"], bits=tgt["bits"]
                    )
                    num = mx.sqrt(mx.mean(mx.square(w - back)))
                    den = mx.sqrt(mx.mean(mx.square(w))) + 1e-12
                    mx.eval(num, den)
                    errors.append((unit["module"], float(num) / float(den)))
                    del back
                del w
                bucket[spec["weight"]] = wq
                bucket[spec["scales"]] = sc
                if spec["biases"]:
                    bucket[spec["biases"]] = bi
            bucket_bytes += unit["bytes"]
            mx.clear_cache()

        done_bytes += src_file.stat().st_size
        del tensors
        mx.clear_cache()

    flush(cur_shard)

    # ---------------- config.json + auxiliary files ---------------- #
    new_q = OrderedDict()
    new_q["group_size"] = args.group_size
    new_q["bits"] = args.bits
    new_q["mode"] = (config.get("quantization") or {}).get("mode", "affine")
    for _, file_units in units:
        for unit in file_units:
            if unit["kind"] != "quant":
                continue
            tgt = unit["target"]
            if (tgt["bits"], tgt["group_size"]) != (args.bits, args.group_size):
                new_q[unit["module"]] = {
                    "group_size": tgt["group_size"],
                    "bits": tgt["bits"],
                }
    config["quantization"] = new_q
    if "quantization_config" in config:
        config["quantization_config"] = json.loads(json.dumps(new_q))
    (dst / "config.json").write_text(json.dumps(config, indent=4) + "\n")

    copied = []
    for entry in sorted(src.iterdir()):
        if entry.name in REGENERATED or entry.suffix == ".safetensors":
            continue
        if entry.is_dir():
            shutil.copytree(entry, dst / entry.name, dirs_exist_ok=True)
            copied.append(entry.name + "/")
        elif entry.is_file():
            shutil.copy2(entry, dst / entry.name)
            copied.append(entry.name)

    if n_shards > 1:
        (dst / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": out_total}, "weight_map": weight_map},
                indent=2,
            )
            + "\n"
        )

    # ---------------- report ---------------- #
    print()
    print(f"copied {len(copied)} auxiliary entries: {', '.join(copied)}")
    print(f"finished in {time.time() - t0:.1f}s -> {dst}")
    if errors:
        errors.sort(key=lambda kv: kv[1], reverse=True)
        mean = sum(e for _, e in errors) / len(errors)
        print(
            f"quantization error ({len(errors)} sampled modules): "
            f"mean rel-rms {mean * 100:.2f}%, worst {errors[0][1] * 100:.2f}%"
        )
        for name, err in errors[:5]:
            print(f"    {err * 100:6.2f}%  {name}")
    if args.vision_bits is None and copied_vision:
        print(
            f"vision tower: {len(copied_vision)} tensors copied verbatim "
            f"(unchanged dtype and bytes)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
