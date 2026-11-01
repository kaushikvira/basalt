// Eager stage-by-stage debug of the DFlash2 forward: runs the context append + block forward without the
// CUDA graph, dumping each intermediate so it can be compared to dbg_ref.py's dump.
#include "src/core/dflash2/forward_kernels.h"
#include "basalt/core/dflash2.hpp"
#include "basalt/core/model_file.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/core/weights.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace basalt::core;
namespace fk = basalt::core::dflash2_detail;

[[noreturn]] void die(const std::string& m) { std::fprintf(stderr, "dbg: %s\n", m.c_str()); std::exit(1); }
std::vector<uint8_t> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) die("open " + p);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void save_f32(const std::string& p, const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost);
    std::ofstream o(p, std::ios::binary);
    o.write((const char*) h.data(), (std::streamsize) (n * 4));
}
void save_bf16_as_f32(const std::string& p, const uint16_t* d, size_t n) {
    std::vector<uint16_t> h(n);
    cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost);
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) { uint32_t u = (uint32_t) h[i] << 16; memcpy(&f[i], &u, 4); }
    std::ofstream o(p, std::ios::binary);
    o.write((const char*) f.data(), (std::streamsize) (n * 4));
}

int main(int argc, char** argv) {
    if (argc != 4) die("usage: dbg <fixture> <target.basalt> <out_dir>");
    const std::string fixture = argv[1], target = argv[2], out = argv[3];
    std::string err;

    ModelFile mf;
    if (!mf.open(target, err)) die(err);
    if (!mf.load_meta(err)) die(err);
    if (!mf.register_gguf_view(err)) die(err);
    ModelSources src;
    model_sources(mf, src);
    NativeEmbed emb;
    if (!emb.load({target}, 2560, 248320, err)) die("emb: " + err);
    set_native_embed(&emb);
    NativeHead head;
    if (!head.load({target}, 2560, 248320, err)) die("head: " + err);

    PackFiles pack = PackFiles::from(src);
    std::set<std::string> skip;
    {
        std::ifstream f(src.dense_index.path);
        f.seekg(src.dense_index.offset);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            char name[256] = {0};
            int file = 0, kind = 0, code_bits = 0;
            unsigned long long s_off = 0, s_bytes = 0, d_off = 0, d_bytes = 0;
            if (std::sscanf(line.c_str(), "%255s %d %d %llu %llu %llu %llu", name, &file, &kind, &s_off, &s_bytes,
                             &d_off, &d_bytes) != 7)
                continue;
            long long ne0 = 0, ne1 = 0;
            int cb = 0;
            if (std::sscanf(line.c_str(), "%*s %*d %*d %*llu %*llu %*llu %*llu %lld %lld %d", &ne0, &ne1, &cb) == 3 &&
                cb != 0 && d_bytes == 0)
                skip.insert(name);
        }
    }
    uint64_t pool = 0;
    if (!WeightTable::pool_bytes(pack, pool, err, &skip)) die(err);
    void* arena = nullptr;
    if (cudaMalloc(&arena, pool) != cudaSuccess) die("wt arena");
    WeightTable wt;
    if (!wt.load(pack, arena, pool, err, &skip)) die("wt: " + err);

    DFlash2Files files;
    {
        const auto dense = read_file(fixture + "/dflash2/dense");
        const auto index = read_file(fixture + "/dflash2/dense.index");
        files.dense = FileRange(fixture + "/dflash2/dense", 0, dense.size());
        files.index = FileRange(fixture + "/dflash2/dense.index", 0, index.size());
    }
    const auto meta_raw = read_file(fixture + "/meta.json");
    Json meta;
    if (!parse_json((const char*) meta_raw.data(), meta_raw.size(), meta, err)) die(err);
    DFlash2Drafter drafter;
    if (!drafter.load(files, meta, err)) die("load: " + err);
    if (!drafter.bind(wt, &head, nullptr, err)) die("bind: " + err);

    const DFlash2Geometry& g = drafter.geom();
    const DFlash2Weights& W = drafter.weights();
    const int Ws = g.block_size, K = Ws - 1, H = g.hidden_size, Q = g.q_width(), KV = g.kv_width();
    const int I = g.intermediate_size, G = H / g.conv_group_size, R = g.selector_rank;
    const int ntap = (int) g.target_layer_ids.size();
    const float scale = 1.0f / sqrtf((float) g.head_dim);

    // working buffers
    auto alloc = [](size_t n) { void* p; if (cudaMalloc(&p, n * 2) != cudaSuccess) die("alloc"); return (uint16_t*) p; };
    uint16_t* residual = alloc((size_t) Ws * H);
    uint16_t* normalized = alloc((size_t) Ws * H);
    uint16_t* prepared = alloc((size_t) Ws * H);
    uint16_t* q = alloc((size_t) Ws * Q);
    uint16_t* k = alloc((size_t) Ws * KV);
    uint16_t* v = alloc((size_t) Ws * KV);
    uint16_t* attn = alloc((size_t) Ws * Q);
    uint16_t* mlp = alloc((size_t) Ws * H);
    uint16_t* kproj = alloc((size_t) Ws * 4 * G);
    uint16_t* fdelta = alloc((size_t) Ws * 2 * G);
    uint16_t* prepared2 = alloc((size_t) Ws * H);
    uint16_t* kproj2 = alloc((size_t) Ws * 4 * G);
    uint16_t* fdelta2 = alloc((size_t) Ws * 2 * G);
    uint16_t* gate = alloc((size_t) Ws * I);
    uint16_t* up = alloc((size_t) Ws * I);
    uint16_t* act = alloc((size_t) Ws * I);
    uint16_t* ctx = alloc(512 * (size_t) H);
    uint16_t* ctx_k = alloc(512 * (size_t) KV);
    uint16_t* ctx_v = alloc(512 * (size_t) KV);
    uint16_t* ctx_x = alloc(512 * (size_t) ntap * H);
    // the cyclic K/V (one window per layer)
    std::vector<uint16_t*> kvk(g.n_layers), kvv(g.n_layers);
    for (int l = 0; l < g.n_layers; ++l) {
        kvk[l] = alloc((size_t) g.sliding_window * KV);
        kvv[l] = alloc((size_t) g.sliding_window * KV);
        cudaMemset(kvk[l], 0, (size_t) g.sliding_window * KV * 2);
        cudaMemset(kvv[l], 0, (size_t) g.sliding_window * KV * 2);
    }
    int* pos_dev = nullptr;
    int* ids_dev = nullptr;
    int* frontier_dev = nullptr;
    if (cudaMalloc(&pos_dev, Ws * 4) != cudaSuccess) die("pos");
    if (cudaMalloc(&ids_dev, Ws * 4) != cudaSuccess) die("ids");
    if (cudaMalloc(&frontier_dev, 4) != cudaSuccess) die("frontier");
    cudaMemset(frontier_dev, 0, 4);
    float* embed_f32 = (float*) alloc((size_t) Ws * H);

    // fixed inputs
    const auto tokens_raw = read_file(fixture + "/tokens.bin");
    const auto features_raw = read_file(fixture + "/features.bin");
    const int64_t* tokens = (const int64_t*) tokens_raw.data();
    const float* features = (const float*) features_raw.data();
    const int n_ctx = (int) (tokens_raw.size() / 8) - 1;
    const int anchor = (int) tokens[n_ctx];

    // ---- context append (eager)
    {
        std::vector<uint16_t> xh((size_t) n_ctx * ntap * H);
        auto to_bf16 = [](float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); };
        for (size_t i = 0; i < xh.size(); ++i) xh[i] = to_bf16(features[i]);
        cudaMemcpy(ctx_x, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
        int pos_h[512];
        for (int i = 0; i < n_ctx; ++i) pos_h[i] = i;
        int* ctx_pos = nullptr;
        if (cudaMalloc(&ctx_pos, n_ctx * 4) != cudaSuccess) die("ctx_pos");
        cudaMemcpy(ctx_pos, pos_h, n_ctx * 4, cudaMemcpyHostToDevice);
        fk::gemm_bf16_wide((const fk::bf16*) W.feature_projection.dev, (const fk::bf16*) ctx_x, (fk::bf16*) ctx, H, ntap * H, n_ctx, 0);
        fk::rmsnorm((const fk::bf16*) ctx, (const float*) W.context_norm.dev, (fk::bf16*) ctx, n_ctx, H, g.rms_norm_eps, 0);
        for (int l = 0; l < g.n_layers; ++l) {
            const DFlash2LayerW& L = W.layers[l];
            fk::gemm_bf16_wide((const fk::bf16*) L.k.dev, (const fk::bf16*) ctx, (fk::bf16*) ctx_k, KV, H, n_ctx, 0);
            fk::rmsnorm_rope(nullptr, (fk::bf16*) ctx_k, ctx_pos, n_ctx, 0, g.num_key_value_heads, nullptr, (const float*) L.k_norm.dev, g.rope_theta, g.rms_norm_eps, 0);
            fk::gemm_bf16_wide((const fk::bf16*) L.v.dev, (const fk::bf16*) ctx, (fk::bf16*) ctx_v, KV, H, n_ctx, 0);
            fk::ctx_store((const fk::bf16*) ctx_k, (const fk::bf16*) ctx_v, 0, n_ctx, g.sliding_window, KV, (fk::bf16*) kvk[l], (fk::bf16*) kvv[l], 0);
            save_bf16_as_f32(out + "/ctx_k_" + std::to_string(l) + ".bin", ctx_k, (size_t) n_ctx * KV);
            save_bf16_as_f32(out + "/ctx_v_" + std::to_string(l) + ".bin", ctx_v, (size_t) n_ctx * KV);
        }
        int fp = n_ctx;
        cudaMemcpy(frontier_dev, &fp, 4, cudaMemcpyHostToDevice);
    }

    // ---- block (eager)
    {
        int ids_h[8], pos_h[8];
        ids_h[0] = anchor;
        for (int i = 1; i < Ws; ++i) ids_h[i] = g.mask_token_id;
        for (int i = 0; i < Ws; ++i) pos_h[i] = n_ctx + i;
        cudaMemcpy(ids_dev, ids_h, Ws * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(pos_dev, pos_h, Ws * 4, cudaMemcpyHostToDevice);
        emb.gather_dev(ids_dev, Ws, embed_f32, 0);
        fk::f32_to_bf16(embed_f32, (fk::bf16*) residual, (int64_t) Ws * H, 0);
        save_bf16_as_f32(out + "/residual_0.bin", residual, (size_t) Ws * H);
        for (int l = 0; l < g.n_layers; ++l) {
            const DFlash2LayerW& L = W.layers[l];
            fk::rmsnorm((const fk::bf16*) residual, (const float*) L.input_norm.dev, (fk::bf16*) normalized, Ws, H, g.rms_norm_eps, 0);
            if (l == 0) save_bf16_as_f32(out + "/l0_n1.bin", normalized, (size_t) Ws * H);
            fk::gemm_bf16((const fk::bf16*) L.attn_conv_kernel.dev, (fk::bf16*) normalized, (fk::bf16*) kproj, 4 * G, H, Ws, 0);
            if (l == 0) save_bf16_as_f32(out + "/l0_d1.bin", kproj, (size_t) Ws * 4 * G);
            fk::conv_in((const fk::bf16*) normalized, (fk::bf16*) kproj, (const fk::bf16*) L.attn_conv_base.dev, (fk::bf16*) prepared, (fk::bf16*) fdelta, Ws, H, G, 0);
            if (l == 0) save_bf16_as_f32(out + "/l0_prepared.bin", prepared, (size_t) Ws * H);
            if (l == 0) save_bf16_as_f32(out + "/l0_fd.bin", fdelta, (size_t) Ws * 2 * G);
            fk::gemm_bf16((const fk::bf16*) L.q.dev, (fk::bf16*) prepared, (fk::bf16*) q, Q, H, Ws, 0);
            if (l == 0) {
                // run the same q GEMM into a fresh buffer to compare
                uint16_t* q2 = alloc((size_t) Ws * Q);
                fk::gemm_bf16((const fk::bf16*) L.q.dev, (fk::bf16*) prepared, (fk::bf16*) q2, Q, H, Ws, 0);
                save_bf16_as_f32(out + "/l0_q_alt.bin", q2, (size_t) Ws * Q);
            }
            fk::gemm_bf16((const fk::bf16*) L.k.dev, (fk::bf16*) prepared, (fk::bf16*) k, KV, H, Ws, 0);
            fk::gemm_bf16((const fk::bf16*) L.v.dev, (fk::bf16*) prepared, (fk::bf16*) v, KV, H, Ws, 0);
            if (l == 0) {
                save_bf16_as_f32(out + "/l0_q_raw.bin", q, (size_t) Ws * Q);
                save_bf16_as_f32(out + "/l0_k_raw.bin", k, (size_t) Ws * KV);
                save_bf16_as_f32(out + "/l0_v_raw.bin", v, (size_t) Ws * KV);
            }
            fk::rmsnorm_rope((fk::bf16*) q, (fk::bf16*) k, pos_dev, Ws, g.num_attention_heads, g.num_key_value_heads, (const float*) L.q_norm.dev, (const float*) L.k_norm.dev, g.rope_theta, g.rms_norm_eps, 0);
            if (l == 0) {
                save_bf16_as_f32(out + "/l0_q.bin", q, (size_t) Ws * Q);
                save_bf16_as_f32(out + "/l0_k.bin", k, (size_t) Ws * KV);
                save_bf16_as_f32(out + "/l0_v.bin", v, (size_t) Ws * KV);
            }
            fk::swa((fk::bf16*) q, (fk::bf16*) k, (fk::bf16*) v, pos_dev, (const fk::bf16*) kvk[l], (const fk::bf16*) kvv[l], Ws, g.num_attention_heads, g.num_key_value_heads, g.head_dim, g.sliding_window, frontier_dev, scale, (fk::bf16*) attn, 0);
            if (l == 0) save_bf16_as_f32(out + "/l0_attn.bin", attn, (size_t) Ws * Q);
            fk::gemm_bf16((const fk::bf16*) L.o_proj.dev, (fk::bf16*) attn, (fk::bf16*) mlp, H, Q, Ws, 0);
            if (l == 0) save_bf16_as_f32(out + "/l0_o_raw.bin", mlp, (size_t) Ws * H);
            fk::conv_out((fk::bf16*) residual, (fk::bf16*) mlp, (const fk::bf16*) L.attn_conv_base.dev, (fk::bf16*) fdelta, Ws, H, G, 0);
            if (l == 0) save_bf16_as_f32(out + "/l0_after_attn.bin", residual, (size_t) Ws * H);
            fk::rmsnorm((const fk::bf16*) residual, (const float*) L.post_attention_norm.dev, (fk::bf16*) normalized, Ws, H, g.rms_norm_eps, 0);
            fk::gemm_bf16((const fk::bf16*) L.mlp_conv_kernel.dev, (fk::bf16*) normalized, (fk::bf16*) kproj2, 4 * G, H, Ws, 0);
            fk::conv_in((const fk::bf16*) normalized, (fk::bf16*) kproj2, (const fk::bf16*) L.mlp_conv_base.dev, (fk::bf16*) prepared2, (fk::bf16*) fdelta2, Ws, H, G, 0);
            if (l == 0) save_bf16_as_f32(out + "/l0_prepared2.bin", prepared2, (size_t) Ws * H);
            fk::gemm_bf16((const fk::bf16*) L.gate.dev, (fk::bf16*) prepared2, (fk::bf16*) gate, I, H, Ws, 0);
            fk::gemm_bf16((const fk::bf16*) L.up.dev, (fk::bf16*) prepared2, (fk::bf16*) up, I, H, Ws, 0);
            fk::swiglu((fk::bf16*) gate, (fk::bf16*) up, (fk::bf16*) act, Ws, I, 0);
            if (l == 0) save_bf16_as_f32(out + "/l0_act.bin", act, (size_t) Ws * I);
            fk::gemm_bf16((const fk::bf16*) L.down.dev, (fk::bf16*) act, (fk::bf16*) mlp, H, I, Ws, 0);
            if (l == 0) save_bf16_as_f32(out + "/l0_down_raw.bin", mlp, (size_t) Ws * H);
            fk::conv_out((fk::bf16*) residual, (fk::bf16*) mlp, (const fk::bf16*) L.mlp_conv_base.dev, (fk::bf16*) fdelta2, Ws, H, G, 0);
            save_bf16_as_f32(out + "/residual_" + std::to_string(l + 1) + ".bin", residual, (size_t) Ws * H);
        }
        uint16_t* hidden_full = alloc((size_t) K * H);
        fk::pack_tail((const fk::bf16*) residual, (const float*) W.final_norm.dev, (fk::bf16*) hidden_full, K, Ws, H, g.rms_norm_eps, 0);
        save_bf16_as_f32(out + "/hidden_full.bin", hidden_full, (size_t) K * H);
        // the logits: the Q8 head GEMM over the full hidden (the reduction is H)
        uint16_t* logits = alloc((size_t) K * 248320);
        fk::gemm_q8((const uint8_t*) head.weights(), (const fk::bf16*) hidden_full, (fk::bf16*) logits, 248320, H, K, 0);
        save_bf16_as_f32(out + "/logits.bin", logits, (size_t) K * 248320);
    }
    // dump the drafter's v_proj weight (first layer) to compare against the pool file
    {
        const auto* vp = (const uint16_t*) W.layers[0].v.dev;
        std::vector<uint16_t> h(1024 * 2560);
        cudaMemcpy(h.data(), vp, h.size() * 2, cudaMemcpyDeviceToHost);
        std::vector<float> f(h.size());
        for (size_t i = 0; i < h.size(); ++i) { uint32_t u = (uint32_t) h[i] << 16; memcpy(&f[i], &u, 4); }
        std::ofstream o(out + "/l0_vproj.bin", std::ios::binary);
        o.write((const char*) f.data(), (std::streamsize) (f.size() * 4));
    }
    {
        const auto* qp = (const uint16_t*) W.layers[0].q.dev;
        std::vector<uint16_t> h(4096 * 2560);
        cudaMemcpy(h.data(), qp, h.size() * 2, cudaMemcpyDeviceToHost);
        std::vector<float> f(h.size());
        for (size_t i = 0; i < h.size(); ++i) { uint32_t u = (uint32_t) h[i] << 16; memcpy(&f[i], &u, 4); }
        std::ofstream o(out + "/l0_qproj.bin", std::ios::binary);
        o.write((const char*) f.data(), (std::streamsize) (f.size() * 4));
    }
    cudaDeviceSynchronize();
    std::printf("dbg: done\n");
    return 0;
}
