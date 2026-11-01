"""The DFlash2 gate's fixture: the dflash2/* sections from the checkpoint, the fixed inputs, and the
NumPy reference (the pipeline's math with the kernel's bf16 roundings).

Usage: python3 make_fixture.py <fixture_dir> <dflash2_source_dir> <target.basalt>
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
from convert.model_sections import DFlash2  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
import ggufq  # noqa: E402


# ---------------------------------------------------------------- the bf16 rounding

def bf16(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 (round-to-nearest-even) -> f32, the kernel's activation rounding."""
    u = x.astype(np.float32).view(np.uint32)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u & 0xFFFF0000).view(np.float32)


def gemm(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """C [N, M] = W [M, K] @ x [N, K], f32 accumulation, one bf16 rounding (the mma's contract)."""
    return bf16(x @ w.T)


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    inv = 1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + eps)
    return bf16(x * inv * w)


def per_head_rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """x [N, NH, 128]: each 128-dim head normalized with the [128] weight, f32, one rounding."""
    inv = 1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + eps)
    return bf16(x * inv * w)


def rope(x: np.ndarray, pos: np.ndarray, theta: float) -> np.ndarray:
    """x [N, NH, 128]: the pair (j, j+64) rotates by pos * theta^(-2j/128), one rounding."""
    n, nh, d = x.shape
    j = np.arange(d // 2, dtype=np.float64)
    ang = pos[:, None].astype(np.float64) * theta ** (-2.0 * j / d)
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., : d // 2].astype(np.float64), x[..., d // 2 :].astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    out[..., : d // 2] = a * c - b * s
    out[..., d // 2 :] = a * s + b * c
    return bf16(out.astype(np.float32))


def conv_in(x: np.ndarray, d: np.ndarray, base: np.ndarray) -> tuple:
    """prepared [W, h] + the finish deltas [W, 2, G] (the kernel's rounding: each tap product is
    rounded to bf16, the sum in f32, one final rounding).  base [2, 2, h]."""
    wsize, h = x.shape
    G = h // 16
    prepared = np.empty_like(x)
    fd = np.empty((wsize, 2, G), dtype=np.float32)
    for i in range(wsize):
        v = bf16((base[0, 0] + np.repeat(d[i, :G], 16)) * x[i]).astype(np.float32)
        if i > 0:
            v = v + bf16((base[0, 1] + np.repeat(d[i, G : 2 * G], 16)) * x[i - 1]).astype(np.float32)
        prepared[i] = bf16(v)
        fd[i, 0] = d[i, 2 * G : 3 * G]
        fd[i, 1] = d[i, 3 * G : 4 * G]
    return prepared, fd


def conv_out(residual: np.ndarray, z: np.ndarray, base: np.ndarray, fd: np.ndarray) -> np.ndarray:
    """residual += the output conv (the kernel: base + delta rounded to bf16, the fma in f32, one
    final rounding)."""
    wsize, h = residual.shape
    G = h // 16
    out = residual.copy()
    for i in range(wsize):
        d0 = bf16((base[1, 0] + np.repeat(fd[i, 0], 16)).astype(np.float32))
        v = (d0.astype(np.float64) * z[i].astype(np.float64))
        if i > 0:
            d1 = bf16((base[1, 1] + np.repeat(fd[i, 1], 16)).astype(np.float32))
            v = v + d1.astype(np.float64) * z[i - 1].astype(np.float64)
        out[i] = bf16(out[i].astype(np.float32) + v.astype(np.float32))
    return out


def swa(q, k, v, bpos, kc, vc, pa, window, nhq, nhk, d):
    """The symmetric sliding-window attention: the context keys [max(0, pi-window+1), pa) + every block
    key; per-head softmax in f32, one bf16 rounding.  q/k/v [W, NH, d]."""
    wsize = q.shape[0]
    if v.ndim == 2:
        v = v.reshape(wsize, nhk, d)
    out = np.empty_like(q, dtype=np.float32)
    scale = 1.0 / np.sqrt(d)
    for i in range(wsize):
        lo = max(0, int(bpos[i]) - window + 1)
        keys = np.concatenate([kc[lo:pa].reshape(pa - lo, nhk, d), k], axis=0)
        vals = np.concatenate([vc[lo:pa].reshape(pa - lo, nhk, d), v], axis=0)
        for hq in range(nhq):
            hk = hq // (nhq // nhk)
            s = (q[i, hq].astype(np.float64)[None, :] @ keys[:, hk].astype(np.float64).T) * scale
            m = s.max()
            wts = np.exp(s - m)
            wts /= wts.sum()
            out[i, hq] = bf16((wts @ vals[:, hk].astype(np.float64)).astype(np.float32))
    return out


# ---------------------------------------------------------------- the reference

def reference(pool, emb, head, tokens, features, geom, anchor, n_ctx):
    """The whole pipeline: the context append + the block forward + the greedy walk."""
    h = geom["hidden_size"]
    nhq, nhk, d = geom["num_attention_heads"], geom["num_key_value_heads"], geom["head_dim"]
    ntap = len(geom["target_layer_ids"])
    eps = geom["rms_norm_eps"]
    theta = geom["rope_theta"]
    window = geom["sliding_window"]
    wsize = geom["block_size"]
    K = wsize - 1

    def t(name):
        return pool[name].astype(np.float32)

    # ---- the context append
    ctx = bf16(features)  # [n, ntap*h] (the host conversion)
    ctx = gemm(t("fc.weight"), ctx)
    ctx = rmsnorm(ctx, t("hidden_norm.weight"), eps)
    kc, vc = {}, {}
    pos = np.arange(n_ctx, dtype=np.int64)
    for l in range(geom["n_layers"]):
        k = gemm(t(f"layers.{l}.self_attn.k_proj.weight"), ctx).reshape(n_ctx, nhk, d)
        k = per_head_rmsnorm(k, t(f"layers.{l}.self_attn.k_norm.weight"), eps)
        k = rope(k, pos, theta)
        v = gemm(t(f"layers.{l}.self_attn.v_proj.weight"), ctx)
        kc[l] = k.reshape(n_ctx, nhk * d)
        vc[l] = v

    # ---- the block
    ids = np.array([anchor] + [geom["mask_token_id"]] * K, dtype=np.int64)
    pa = n_ctx
    bpos = pa + np.arange(wsize, dtype=np.int64)
    residual = bf16(emb[ids])  # [W, h]
    for l in range(geom["n_layers"]):
        n1 = rmsnorm(residual, t(f"layers.{l}.input_layernorm.weight"), eps)
        d1 = gemm(t(f"layers.{l}.attention_conv.kernel_projection.weight"), n1)
        prepared, fd = conv_in(n1, d1, t(f"layers.{l}.attention_conv.base_kernel").reshape(2, 2, h))
        q = gemm(t(f"layers.{l}.self_attn.q_proj.weight"), prepared).reshape(wsize, nhq, d)
        k = gemm(t(f"layers.{l}.self_attn.k_proj.weight"), prepared).reshape(wsize, nhk, d)
        v = gemm(t(f"layers.{l}.self_attn.v_proj.weight"), prepared).reshape(wsize, nhk, d)
        q = per_head_rmsnorm(q, t(f"layers.{l}.self_attn.q_norm.weight"), eps)
        k = per_head_rmsnorm(k, t(f"layers.{l}.self_attn.k_norm.weight"), eps)
        q = rope(q, bpos, theta)
        k = rope(k, bpos, theta)
        attn = swa(q, k, v, bpos, kc[l], vc[l], pa, window, nhq, nhk, d)
        attn_out = gemm(t(f"layers.{l}.self_attn.o_proj.weight"), attn.reshape(wsize, nhq * d))
        residual = conv_out(residual, attn_out, t(f"layers.{l}.attention_conv.base_kernel").reshape(2, 2, h), fd)
        n2 = rmsnorm(residual, t(f"layers.{l}.post_attention_layernorm.weight"), eps)
        d2 = gemm(t(f"layers.{l}.mlp_conv.kernel_projection.weight"), n2)
        prepared2, fd2 = conv_in(n2, d2, t(f"layers.{l}.mlp_conv.base_kernel").reshape(2, 2, h))
        gate = gemm(t(f"layers.{l}.mlp.gate_proj.weight"), prepared2)
        up = gemm(t(f"layers.{l}.mlp.up_proj.weight"), prepared2)
        act = bf16(gate / (1.0 + np.exp(-gate)) * up)
        mlp_out = gemm(t(f"layers.{l}.mlp.down_proj.weight"), act)
        residual = conv_out(residual, mlp_out, t(f"layers.{l}.mlp_conv.base_kernel").reshape(2, 2, h), fd2)

    hidden_full = rmsnorm(residual[1:], t("norm.weight"), eps)
    hidden_r = gemm(t("candidate_selector.hidden_projection.weight"), hidden_full)
    head_bf = bf16(head)  # the kernel's dequant rounding
    logits = bf16(hidden_full @ head_bf.T)

    top16 = np.empty((K, 16), dtype=np.int64)
    unary = np.empty((K, 16), dtype=np.float32)
    for i in range(K):
        order = np.argsort(-logits[i], kind="stable")[:16]
        top16[i] = order
        unary[i] = logits[i][order]
    pred = t("candidate_selector.predecessor_codebook")
    succ = t("candidate_selector.successor_codebook")
    drafts = np.empty(K, dtype=np.int64)
    pred_tok = anchor
    for step in range(K):
        product = pred[pred_tok] * hidden_r[step]
        edges = unary[step] + np.array(
            [float(np.dot(product, succ[top16[step, c]])) for c in range(16)]
        )
        pred_tok = int(top16[step, int(np.argmax(edges))])
        drafts[step] = pred_tok
    return {
        "hidden_full": hidden_full,
        "hidden_r": hidden_r,
        "logits": logits,
        "top16": top16,
        "unary": unary,
        "drafts": drafts,
    }


# ---------------------------------------------------------------- the main

def q8_rows(data: bytes, shape: list) -> np.ndarray:
    a = ggufq.dequantize(np.frombuffer(data, dtype=np.uint8).copy(), ggufq.GGMLQuantizationType.Q8_0)
    return a.reshape(shape[1], shape[0]).astype(np.float32)  # [rows, h] (the GGUF's [h, rows])


def main():
    fixture = pathlib.Path(sys.argv[1])
    src = pathlib.Path(sys.argv[2])
    target = pathlib.Path(sys.argv[3])
    (fixture / "dflash2").mkdir(parents=True, exist_ok=True)

    d = DFlash2(str(src))
    with open(fixture / "dflash2" / "dense", "wb") as f:
        for chunk in d.dense_chunks():
            f.write(chunk)
    (fixture / "dflash2" / "dense.index").write_bytes(d.dense_index())
    (fixture / "meta.json").write_text(json.dumps({"dflash2": d.provenance()}))

    # the pool's tensors for the reference, in the pool's form (norms F32, the rest BF16-as-f32; the
    # bf16 -> f32 widening is exact)
    pool = {}
    for name in sorted(d.tensors):
        shape = d.tensors[name]["shape"]
        raw = d._tensor_bytes(name)
        u16 = raw.view("<u2")
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        pool[name] = f32 if len(shape) == 1 else f32.reshape(shape[0], -1)

    # the target's GGUF tensors: the meta's table + the `gguf` section's data
    import model_file as mf

    m = mf.ModelFile(str(target))
    gmeta = json.loads(m.read("meta").decode())
    goff, _ = m.section("gguf")
    with open(target, "rb") as f:
        for name in ("token_embd.weight", "output.weight"):
            t = gmeta["tensors"][name]
            f.seek(goff + t["offset"])
            raw = f.read(t["bytes"])
            if name == "token_embd.weight":
                emb = q8_rows(raw, t["shape"])
            else:
                head = q8_rows(raw, t["shape"])

    geom = {
        "hidden_size": d.config.get("hidden_size"),
        "num_attention_heads": d.config.get("num_attention_heads"),
        "num_key_value_heads": d.config.get("num_key_value_heads"),
        "head_dim": d.config.get("head_dim"),
        "intermediate_size": d.config.get("intermediate_size"),
        "vocab_size": d.config.get("vocab_size"),
        "n_layers": d.config.get("num_hidden_layers"),
        "rms_norm_eps": d.config.get("rms_norm_eps"),
        "rope_theta": d.config.get("rope_parameters", {}).get("rope_theta", d.config.get("rope_theta")),
        "sliding_window": d.config.get("sliding_window"),
        "block_size": d.config["dflash_config"]["block_size"],
        "conv_group_size": d.config["dflash_config"]["conv_group_size"],
        "selector_rank": d.config["dflash_config"]["selector_rank"],
        "target_layer_ids": d.config["dflash_config"]["target_layer_ids"],
        "mask_token_id": d.config["dflash_config"]["mask_token_id"],
    }
    rng = np.random.default_rng(42)
    n_ctx = 64
    tokens = rng.integers(0, geom["vocab_size"], size=n_ctx + 1).astype(np.int64)
    anchor = int(tokens[-1])
    features = (
        rng.standard_normal((n_ctx, len(geom["target_layer_ids"]) * geom["hidden_size"])) * 0.02
    ).astype(np.float32)

    ref = reference(pool, emb, head, tokens, features, geom, anchor, n_ctx)

    # the arrays as raw files + a manifest (the gate's reader is a few lines)
    arrays = {
        "tokens": (tokens, "i8"),
        "features": (features, "f4"),
        "ref_hidden_full": (ref["hidden_full"], "f4"),
        "ref_hidden_r": (ref["hidden_r"], "f4"),
        "ref_logits": (ref["logits"], "f4"),
        "ref_top16": (ref["top16"], "i8"),
        "ref_drafts": (ref["drafts"], "i8"),
    }
    manifest = {}
    dts = {"i8": np.int64, "f4": np.float32}
    for name, (arr, dt) in arrays.items():
        a = arr.astype(dts[dt]).tobytes()
        (fixture / f"{name}.bin").write_bytes(a)
        manifest[name] = {"shape": list(arr.shape), "dtype": dt}
    (fixture / "manifest.json").write_text(json.dumps(manifest))
    print(f"fixture: {fixture} (pool {d.dense_bytes} B, {len(pool)} tensors, {n_ctx} ctx tokens, anchor {anchor})")


if __name__ == "__main__":
    main()
