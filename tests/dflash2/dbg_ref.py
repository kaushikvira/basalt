#!/usr/bin/env python3
"""Dump the reference's per-layer intermediates so the C++ forward can be compared stage by stage.

Usage: dbg_ref.py <fixture_dir> <src_checkpoint> <target.basalt> <out_dir>
"""
import json
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
from convert.model_sections import DFlash2  # noqa: E402
import ggufq  # noqa: E402
import model_file as mfm  # noqa: E402
import make_fixture as mf  # noqa: E402


def q8_rows(data, shape):
    a = ggufq.dequantize(np.frombuffer(data, dtype=np.uint8).copy(), ggufq.GGMLQuantizationType.Q8_0)
    return a.reshape(shape[1], shape[0]).astype(np.float32)


def main():
    fixture = pathlib.Path(sys.argv[1])
    src = pathlib.Path(sys.argv[2])
    target = pathlib.Path(sys.argv[3])
    out = pathlib.Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)

    # load the pool directly from the fixture's dense/index (bf16 or f32 per the index's dtype)
    data = open(fixture / "dflash2" / "dense", "rb").read()
    pool = {}
    with open(fixture / "dflash2" / "dense.index") as f:
        for line in f:
            p = line.split()
            name, dtype, rows, cols, off, nbytes = p[0], p[1], int(p[2]), int(p[3]), int(p[4]), int(p[5])
            if dtype == "f32":
                f32 = np.frombuffer(data[off:off + nbytes], dtype=np.float32).copy()
            else:
                f32 = (np.frombuffer(data[off:off + nbytes], dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            pool[name] = f32 if rows == 1 else f32.reshape(rows, -1)
    # the geometry comes from the real checkpoint's config
    import json as _json
    cfg = _json.load(open("/home/kv/work/models/dflash2-fn/config.json"))

    m = mfm.ModelFile(str(target))
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
        "hidden_size": cfg.get("hidden_size"),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "num_key_value_heads": cfg.get("num_key_value_heads"),
        "head_dim": cfg.get("head_dim"),
        "intermediate_size": cfg.get("intermediate_size"),
        "vocab_size": cfg.get("vocab_size"),
        "n_layers": cfg.get("num_hidden_layers"),
        "rms_norm_eps": cfg.get("rms_norm_eps"),
        "rope_theta": cfg.get("rope_parameters", {}).get("rope_theta", cfg.get("rope_theta")),
        "sliding_window": cfg.get("sliding_window"),
        "block_size": cfg["dflash_config"]["block_size"],
        "conv_group_size": cfg["dflash_config"]["conv_group_size"],
        "selector_rank": cfg["dflash_config"]["selector_rank"],
        "target_layer_ids": cfg["dflash_config"]["target_layer_ids"],
        "mask_token_id": cfg["dflash_config"]["mask_token_id"],
    }
    # read the fixture's fixed inputs
    manifest = json.loads((fixture / "manifest.json").read_text())
    tokens = np.fromfile(fixture / "tokens.bin", dtype=np.int64)
    features = np.fromfile(fixture / "features.bin", dtype=np.float32).reshape(manifest["features"]["shape"])
    n_ctx = len(tokens) - 1
    anchor = int(tokens[-1])

    h = geom["hidden_size"]
    G = h // 16
    nhq, nhk, d_ = geom["num_attention_heads"], geom["num_key_value_heads"], geom["head_dim"]
    ntap = len(geom["target_layer_ids"])
    eps = geom["rms_norm_eps"]
    theta = geom["rope_theta"]
    window = geom["sliding_window"]
    wsize = geom["block_size"]
    K = wsize - 1

    def t(name):
        return pool[name].astype(np.float32)

    def bf16(x):
        return mf.bf16(x)

    # ---- context append
    ctx = bf16(features)
    ctx = mf.gemm(t("fc.weight"), ctx)
    ctx = mf.rmsnorm(ctx, t("hidden_norm.weight"), eps)
    kc, vc = {}, {}
    pos = np.arange(n_ctx, dtype=np.int64)
    for l in range(geom["n_layers"]):
        k = mf.gemm(t(f"layers.{l}.self_attn.k_proj.weight"), ctx).reshape(n_ctx, nhk, d_)
        k = mf.per_head_rmsnorm(k, t(f"layers.{l}.self_attn.k_norm.weight"), eps)
        k = mf.rope(k, pos, theta)
        v = mf.gemm(t(f"layers.{l}.self_attn.v_proj.weight"), ctx)
        kc[l] = k.reshape(n_ctx, nhk * d_)
        vc[l] = v
    for l in range(geom["n_layers"]):
        np.save(out / f"ctx_k_{l}.npy", kc[l])
        np.save(out / f"ctx_v_{l}.npy", vc[l])

    # ---- block
    ids = np.array([anchor] + [geom["mask_token_id"]] * K, dtype=np.int64)
    pa = n_ctx
    bpos = pa + np.arange(wsize, dtype=np.int64)
    residual = bf16(emb[ids])
    np.save(out / "residual_0.npy", residual)
    for l in range(geom["n_layers"]):
        n1 = mf.rmsnorm(residual, t(f"layers.{l}.input_layernorm.weight"), eps)
        d1 = mf.gemm(t(f"layers.{l}.attention_conv.kernel_projection.weight"), n1)
        if l == 0:
            np.save(out / "l0_d1.npy", d1)
        prepared, fd = mf.conv_in(n1, d1, t(f"layers.{l}.attention_conv.base_kernel").reshape(2, 2, h))
        if l == 0:
            np.save(out / "l0_fd.npy", fd.reshape(wsize, 2 * G).astype(np.float32))
        q = mf.gemm(t(f"layers.{l}.self_attn.q_proj.weight"), prepared).reshape(wsize, nhq, d_)
        k = mf.gemm(t(f"layers.{l}.self_attn.k_proj.weight"), prepared).reshape(wsize, nhk, d_)
        v = mf.gemm(t(f"layers.{l}.self_attn.v_proj.weight"), prepared).reshape(wsize, nhk, d_)
        q = mf.per_head_rmsnorm(q, t(f"layers.{l}.self_attn.q_norm.weight"), eps)
        k = mf.per_head_rmsnorm(k, t(f"layers.{l}.self_attn.k_norm.weight"), eps)
        q = mf.rope(q, bpos, theta)
        k = mf.rope(k, bpos, theta)
        attn = mf.swa(q, k, v, bpos, kc[l], vc[l], pa, window, nhq, nhk, d_)
        attn_out = mf.gemm(t(f"layers.{l}.self_attn.o_proj.weight"), attn.reshape(wsize, nhq * d_))
        if l == 0:
            np.save(out / "l0_o_raw.npy", attn_out)
        residual = mf.conv_out(residual, attn_out, t(f"layers.{l}.attention_conv.base_kernel").reshape(2, 2, h), fd)
        if l == 0:
            np.save(out / "l0_after_attn.npy", residual)
        n2 = mf.rmsnorm(residual, t(f"layers.{l}.post_attention_layernorm.weight"), eps)
        d2 = mf.gemm(t(f"layers.{l}.mlp_conv.kernel_projection.weight"), n2)
        prepared2, fd2 = mf.conv_in(n2, d2, t(f"layers.{l}.mlp_conv.base_kernel").reshape(2, 2, h))
        if l == 0:
            np.save(out / "l0_prepared2.npy", prepared2)
        gate = mf.gemm(t(f"layers.{l}.mlp.gate_proj.weight"), prepared2)
        up = mf.gemm(t(f"layers.{l}.mlp.up_proj.weight"), prepared2)
        act = bf16(gate / (1.0 + np.exp(-gate)) * up)
        if l == 0:
            np.save(out / "l0_act.npy", act)
        mlp_out = mf.gemm(t(f"layers.{l}.mlp.down_proj.weight"), act)
        if l == 0:
            np.save(out / "l0_down_raw.npy", mlp_out)
        residual = mf.conv_out(residual, mlp_out, t(f"layers.{l}.mlp_conv.base_kernel").reshape(2, 2, h), fd2)
        np.save(out / f"residual_{l + 1}.npy", residual)
        if l == 0:
            np.save(out / "l0_q.npy", q)
            np.save(out / "l0_k.npy", k)
            np.save(out / "l0_v.npy", v)
            np.save(out / "l0_attn.npy", attn)
            np.save(out / "l0_prepared.npy", prepared)
            np.save(out / "l0_n1.npy", n1)

    hidden_full = mf.rmsnorm(residual[1:], t("norm.weight"), eps)
    np.save(out / "hidden_full.npy", hidden_full)
    # the logits: the Q8 head GEMM over the full hidden
    logits = head @ hidden_full.T  # [V, K]
    np.save(out / "logits.npy", logits.astype(np.float32).T)  # [K, V]
    print(f"dumped {out} (n_ctx={n_ctx}, anchor={anchor})")


if __name__ == "__main__":
    main()
