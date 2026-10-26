#!/usr/bin/env python3
"""End-to-end test for conv.py on a tiny synthetic MLX model.

Builds a two-shard fake model whose layout mimics the real Qwen3.5-MoE VLM
(biases/scales/weight ordering, a MoE 3-D tensor, an already-4-bit module, a
bf16 vision tower) and checks the converted output.

    python3 tests/test_conv.py
"""

import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

HERE = Path(__file__).resolve().parent
CONV = HERE.parent / "conv.py"

BITS, GS = 4, 64


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    header.pop("__metadata__", None)
    return header


def build_model(root: Path, shard_size_bytes: int = 0):
    """A fake 6-bit (mixed 4-bit) MLX model, optionally split into shards."""
    mx.random.seed(0)
    hidden, vocab, experts, moe_hidden = 128, 512, 4, 64

    quant_spec = {}
    tensors = {}

    def add_quant(name, shape, bits, gs):
        w = mx.random.normal(shape).astype(mx.bfloat16) * 0.02
        wq, sc, bi = mx.quantize(w, group_size=gs, bits=bits)
        # stored order in the wild: biases, scales, weight
        tensors[name + ".biases"] = bi
        tensors[name + ".scales"] = sc
        tensors[name + ".weight"] = wq
        quant_spec[name] = {"group_size": gs, "bits": bits}

    add_quant("language_model.model.embed_tokens", (vocab, hidden), 6, 32)
    add_quant("language_model.lm_head", (vocab, hidden), 6, 32)
    add_quant("language_model.model.layers.0.self_attn.q_proj", (hidden, hidden), 6, 32)
    add_quant(
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj",
        (experts, moe_hidden, hidden),
        6,
        32,
    )
    add_quant(
        "language_model.model.layers.0.mlp.switch_mlp.down_proj",
        (experts, hidden, moe_hidden),
        6,
        32,
    )
    # already at the target precision -> must be copied byte for byte
    add_quant("language_model.model.layers.1.self_attn.q_proj", (hidden, hidden), 4, 64)

    tensors["language_model.model.layers.0.input_layernorm.weight"] = mx.ones(
        (hidden,), dtype=mx.bfloat16
    )
    tensors["language_model.model.layers.0.linear_attn.conv1d.weight"] = (
        mx.random.normal((hidden, 4, 1)).astype(mx.bfloat16)
    )
    # vision tower: never quantized, must survive byte for byte
    tensors["vision_tower.blocks.0.attn.qkv.weight"] = mx.random.normal(
        (3 * 64, 64)
    ).astype(mx.bfloat16)
    tensors["vision_tower.blocks.0.attn.qkv.bias"] = mx.random.normal((3 * 64,)).astype(
        mx.bfloat16
    )
    tensors["vision_tower.patch_embed.proj.weight"] = mx.random.normal(
        (64, 3, 2, 16, 16)
    ).astype(mx.bfloat16)

    config = {
        "architectures": ["FakeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "dtype": "bfloat16",
        "vision_config": {"depth": 1, "hidden_size": 64},
        "quantization": dict(
            {"group_size": 32, "bits": 6, "mode": "affine"}, **quant_spec
        ),
    }
    config["quantization_config"] = json.loads(json.dumps(config["quantization"]))

    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config, indent=4))
    (root / "tokenizer.json").write_text('{"fake": true}')
    (root / "preprocessor_config.json").write_text('{"fake": true}')

    names = list(tensors)
    if shard_size_bytes:
        shards, cur, cur_bytes = [], {}, 0
        for n in names:
            nb = tensors[n].nbytes
            if cur and cur_bytes + nb > shard_size_bytes:
                shards.append(cur)
                cur, cur_bytes = {}, 0
            cur[n] = tensors[n]
            cur_bytes += nb
        if cur:
            shards.append(cur)
    else:
        shards = [tensors]

    if len(shards) == 1:
        mx.save_safetensors(str(root / "model.safetensors"), shards[0])
        return

    weight_map = {}
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(root / fname), shard)
        for n in shard:
            weight_map[n] = fname
    total = sum(t.nbytes for t in tensors.values())
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2
        )
    )


def run(src, dst, *extra):
    cmd = [sys.executable, str(CONV), str(src), str(dst), *extra]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise AssertionError(f"conv.py failed\n{out.stdout}\n{out.stderr}")
    return out.stdout


def load_model(root: Path):
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        weight_map = json.loads(idx.read_text())["weight_map"]
        tensors = {}
        for f in sorted(set(weight_map.values())):
            tensors.update(mx.load(str(root / f)))
        return tensors, json.loads(idx.read_text())
    return mx.load(str(root / "model.safetensors")), None


def check(src, dst, *, shards=False):
    src_t, _ = load_model(src)
    out_t, out_idx = load_model(dst)
    cfg = json.loads((dst / "config.json").read_text())
    src_cfg = json.loads((src / "config.json").read_text())

    assert set(src_t) == set(out_t), "tensor name set changed"
    assert out_idx is None or set(out_idx["weight_map"]) == set(out_t)
    if shards:
        assert out_idx is not None, "expected an index for a multi-shard output"

    src_q = {k: v for k, v in src_cfg["quantization"].items() if isinstance(v, dict)}
    assert cfg["quantization"]["bits"] == BITS
    assert cfg["quantization"]["group_size"] == GS
    assert "quantization_config" in cfg, "quantization_config mirror lost"

    worst = 0.0
    for module, spec in src_q.items():
        w_src = mx.dequantize(
            src_t[module + ".weight"],
            src_t[module + ".scales"],
            src_t[module + ".biases"],
            group_size=spec["group_size"],
            bits=spec["bits"],
        )
        tgt = cfg["quantization"].get(module, {"bits": BITS, "group_size": GS})
        packed = out_t[module + ".weight"]
        assert packed.dtype == mx.uint32
        assert packed.shape[-1] == w_src.shape[-1] * tgt["bits"] // 32, module
        assert (
            out_t[module + ".scales"].shape[-1] == w_src.shape[-1] // tgt["group_size"]
        )
        w_out = mx.dequantize(
            packed,
            out_t[module + ".scales"],
            out_t[module + ".biases"],
            group_size=tgt["group_size"],
            bits=tgt["bits"],
        )
        err = float(
            mx.sqrt(mx.mean(mx.square(w_src - w_out)))
            / (mx.sqrt(mx.mean(mx.square(w_src))) + 1e-12)
        )
        worst = max(worst, err)
    assert worst < 0.20, f"quantization error too large: {worst:.3f}"

    # non-quantized tensors must be identical (vision tower above all)
    plain = [
        n
        for n in src_t
        if not any(n == m + s for m in src_q for s in (".weight", ".scales", ".biases"))
    ]
    for n in plain:
        assert src_t[n].dtype == out_t[n].dtype, n
        assert src_t[n].shape == out_t[n].shape, n
        assert bool(mx.array_equal(src_t[n], out_t[n])), f"tensor changed: {n}"
    assert any(n.startswith("vision_tower") for n in plain)
    return worst


def main():
    tmp = Path(tempfile.mkdtemp(prefix="conv-test-"))
    try:
        # 1) single file -> single file
        src = tmp / "src1"
        build_model(src)
        dst = tmp / "dst1"
        print(run(src, dst))
        err = check(src, dst)
        assert (dst / "model.safetensors").exists()
        assert not (dst / "model.safetensors.index.json").exists()
        print(f"ok: single shard, worst rel-rms {err * 100:.2f}%")

        # 2) multi shard -> multi shard with forced small shards
        src2 = tmp / "src2"
        build_model(src2, shard_size_bytes=200_000)
        n_src = len(list(src2.glob("*.safetensors")))
        assert n_src > 1, "test setup: expected several source shards"
        dst2 = tmp / "dst2"
        print(run(src2, dst2, "--shard-size", "120K"))
        err = check(src2, dst2, shards=True)
        n_out = len(list(dst2.glob("model-*.safetensors")))
        assert n_out > 1, "expected several output shards"
        print(
            f"ok: {n_src} source shards -> {n_out} output shards, worst {err * 100:.2f}%"
        )

        # 3) refusing to overwrite
        r = subprocess.run(
            [sys.executable, str(CONV), str(src), str(dst)],
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0 and "exist" in (r.stderr + r.stdout)
        print("ok: refuses to overwrite without --overwrite")

        # 4) another target precision keeps every module valid
        dst3 = tmp / "dst3"
        print(run(src, dst3, "--bits", "8", "--group-size", "32"))
        cfg3 = json.loads((dst3 / "config.json").read_text())
        assert cfg3["quantization"]["bits"] == 8
        assert cfg3["quantization"]["group_size"] == 32
        out3, _ = load_model(dst3)
        # lm_head is 512x128 -> 8-bit packing is half the elements of 6-bit? no: 128*8/32
        assert out3["language_model.lm_head.weight"].shape == (512, 128 * 8 // 32)
        assert out3["language_model.lm_head.scales"].shape == (512, 128 // 32)
        print("ok: 8-bit target produces 8-bit packed weights")

        print("\nALL TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
