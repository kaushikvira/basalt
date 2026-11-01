// DFlash2Drafter::prewarm / draft / append_context - the block-diffusion forward.
//
// The block forward (the anchor + K-1 masks through the 5 layers, the greedy selector walk) is captured
// once in prewarm() as a CUDA graph (every kernel reads the fixed arena pointers; the per-round values -
// the ids, the positions, the anchor, the frontier - cross in device buffers).  append_context() is eager
// (the token count varies): the feature projection over the tapped hidden states, each layer's K/V, the
// cyclic slots.
#include "forward_kernels.h"

#include "basalt/core/dflash2.hpp"
#include "basalt/core/native_head.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace basalt::core {
namespace {

constexpr int GGML_Q8_0 = 8;   // include/basalt/ggml/ggml.h

namespace fk = basalt::core::dflash2_detail;

/// The arena's slots, carved 256-aligned.
struct Carve {
    uint8_t* p = nullptr;
    uint64_t left = 0;
    void* take(uint64_t bytes) {
        bytes = (bytes + 255) & ~255ull;
        void* out = p;
        p += bytes;
        left -= bytes;
        return out;
    }
};

/// The arena's buffers are uint16_t*; the kernels take the bf16 alias.
inline fk::bf16* b16(void* p) { return (fk::bf16*) p; }
inline const fk::bf16* cb16(const void* p) { return (const fk::bf16*) p; }

}  // namespace

// ================================ the prewarm ================================

bool DFlash2Drafter::prewarm(std::string& err) {
    if (tensors_.empty() || !dense_) {
        err = "dflash2: prewarm before bind";
        return false;
    }
    if (!head_ || !head_->loaded()) {
        err = "dflash2: the target's output head is not loaded";
        return false;
    }
    if (head_->type() != GGML_Q8_0) {
        err = "dflash2: the head is GGUF type " + std::to_string(head_->type()) + ", not Q8_0";
        return false;
    }
    const DFlash2Geometry& g = g_;
    const int W = g.block_size, K = W - 1, H = g.hidden_size, Q = g.q_width(), KV = g.kv_width();
    const int I = g.intermediate_size, G = H / g.conv_group_size, R = g.selector_rank, V = g.vocab_size;
    const int ntap = (int) g.target_layer_ids.size();
    const int chunks = (V + 1023) / 1024;
    const uint64_t B2 = 2u;   // bf16

    // ---- the arena's slots
    Carve cv{(uint8_t*) arena_, arena_bytes_};
    wk_.residual = (uint16_t*) cv.take((uint64_t) W * H * B2);
    wk_.normalized = (uint16_t*) cv.take((uint64_t) W * H * B2);
    wk_.prepared = (uint16_t*) cv.take((uint64_t) W * H * B2);
    wk_.q = (uint16_t*) cv.take((uint64_t) W * Q * B2);
    wk_.k = (uint16_t*) cv.take((uint64_t) W * KV * B2);
    wk_.v = (uint16_t*) cv.take((uint64_t) W * KV * B2);
    wk_.attn = (uint16_t*) cv.take((uint64_t) W * Q * B2);
    wk_.gate = (uint16_t*) cv.take((uint64_t) W * I * B2);
    wk_.up = (uint16_t*) cv.take((uint64_t) W * I * B2);
    wk_.act = (uint16_t*) cv.take((uint64_t) W * I * B2);
    wk_.mlp = (uint16_t*) cv.take((uint64_t) W * H * B2);
    wk_.kproj = (uint16_t*) cv.take((uint64_t) W * 4u * G * B2);
    wk_.fdelta = (uint16_t*) cv.take((uint64_t) W * 2u * G * B2);
    wk_.prepared2 = (uint16_t*) cv.take((uint64_t) W * H * B2);
    wk_.kproj2 = (uint16_t*) cv.take((uint64_t) W * 4u * G * B2);
    wk_.fdelta2 = (uint16_t*) cv.take((uint64_t) W * 2u * G * B2);
    wk_.ctx = (uint16_t*) cv.take(512u * (uint64_t) H * B2);
    wk_.ctx_k = (uint16_t*) cv.take(512u * (uint64_t) KV * B2);
    wk_.ctx_v = (uint16_t*) cv.take(512u * (uint64_t) KV * B2);
    wk_.ctx_x = (uint16_t*) cv.take(512u * (uint64_t) ntap * (uint64_t) H * B2);
    // the named takes must fit the budgeted arena (a wrap of cv.left is silent; check the signed sum)
    {
        int64_t used = (int64_t) (arena_bytes_ - cv.left);
        if (used > (int64_t) arena_bytes_) {
            err = "dflash2: the working arena is under-budgeted (activations_bytes)";
            return false;
        }
    }
    wk_.logits = (uint16_t*) cv.take((uint64_t) K * (uint64_t) V * B2);
    wk_.hidden_full = (uint16_t*) cv.take((uint64_t) K * (uint64_t) H * B2);
    wk_.hidden_r = (uint16_t*) cv.take((uint64_t) K * (uint64_t) R * B2);
    if (cv.left < (uint64_t) W * H * 4u + (uint64_t) K * 16u * 8u + 2u * (uint64_t) chunks * K * 16u * 8u +
                      128u * 4u + 6u * 64u) {
        err = "dflash2: the working arena is too small for the forward's buffers";
        return false;
    }
    wk_.embed_f32 = (float*) cv.take((uint64_t) W * H * 4u);
    wk_.cand_ids = (int*) cv.take((uint64_t) K * 16u * 4u);
    wk_.cand_scores = (float*) cv.take((uint64_t) K * 16u * 4u);
    wk_.tk_s[0] = (float*) cv.take((uint64_t) chunks * K * 16u * 4u);
    wk_.tk_ids[0] = (int*) cv.take((uint64_t) chunks * K * 16u * 4u);
    wk_.tk_s[1] = (float*) cv.take((uint64_t) chunks * K * 16u * 4u);
    wk_.tk_ids[1] = (int*) cv.take((uint64_t) chunks * K * 16u * 4u);
    wk_.pos = (int*) cv.take(64u * 4u);
    wk_.ids = (int*) cv.take(64u * 4u);
    wk_.frontier = (int*) cv.take(4u);
    wk_.drafts_dev = (int*) cv.take(64u * 4u);
    wk_.ctx_pos = (int*) cv.take(512u * 4u);
    wk_.ptrs = cv.take(6u * 64u);   // the 64 B pointer slots: 5 x swa (per layer) + 1 x the walk

    if (cudaHostAlloc(&n_drafts_pin_, 4, cudaHostAllocDefault) != cudaSuccess) {
        err = "dflash2: the pinned draft count";
        return false;
    }
    *n_drafts_pin_ = 0;
    frontier_pos_ = 0;
    cudaMemsetAsync(wk_.frontier, 0, 4, cs_);

    // ---- the block forward, captured
    const float scale = 1.0f / std::sqrt((float) g.head_dim);
    const auto* selh = cb16(w_.sel_hidden.dev);
    const auto* pred = cb16(w_.sel_pred.dev);
    const auto* succ = cb16(w_.sel_succ.dev);
    const auto* fin = (const float*) w_.final_norm.dev;
    const auto* headw = (const uint8_t*) head_->weights();

    // the pointer slots (the driver's 64 B parameter limit): written once here - the pointers are fixed
    // for the drafter's lifetime, so the graph's kernels read the same slots at every replay
    for (int l = 0; l < g.n_layers; ++l)
        fk::swa_write_slot((uint8_t*) wk_.ptrs + (size_t) l * 64u, b16(wk_.q), b16(wk_.k), b16(wk_.v),
                           cb16(kv_k_[(size_t) l]), cb16(kv_v_[(size_t) l]), wk_.pos, wk_.frontier, b16(wk_.attn));
    fk::sel_write_slot((uint8_t*) wk_.ptrs + (size_t) g.n_layers * 64u, wk_.cand_ids, wk_.cand_scores,
                       cb16(wk_.hidden_r), wk_.ids, pred, succ, wk_.drafts_dev, n_drafts_pin_);

    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        err = "dflash2: the graph capture began";
        cudaGetLastError();
        return false;
    }
    // 1: the embedding (the target's, gathered in f32, then to bf16)
    {
        const auto* ne = native_embed();
        if (!ne) {
            cudaStreamEndCapture(cs_, nullptr);
            err = "dflash2: the target's embedding is not set";
            return false;
        }
        ne->gather_dev(wk_.ids, W, wk_.embed_f32, cs_);
        fk::f32_to_bf16(wk_.embed_f32, b16(wk_.residual), (int64_t) W * H, cs_);
    }
    // 2: the layers (the NInfer order: the conv prepare runs on the normalized input, the projections on
    // the prepared one, the output conv folds the branch out into the residual with the prepared deltas)
    for (int l = 0; l < g.n_layers; ++l) {
        const DFlash2LayerW& L = w_.layers[(size_t) l];
        // the attention branch
        fk::rmsnorm(b16(wk_.residual), (const float*) L.input_norm.dev, b16(wk_.normalized), W, H, g.rms_norm_eps,
                    cs_);
        fk::gemm_bf16(cb16(L.attn_conv_kernel.dev), b16(wk_.normalized), b16(wk_.kproj), 4 * G, H, W, cs_);
        fk::conv_in(b16(wk_.normalized), b16(wk_.kproj), cb16(L.attn_conv_base.dev), b16(wk_.prepared),
                    b16(wk_.fdelta), W, H, G, cs_);
        fk::gemm_bf16(cb16(L.q.dev), b16(wk_.prepared), b16(wk_.q), Q, H, W, cs_);
        fk::gemm_bf16(cb16(L.k.dev), b16(wk_.prepared), b16(wk_.k), KV, H, W, cs_);
        fk::gemm_bf16(cb16(L.v.dev), b16(wk_.prepared), b16(wk_.v), KV, H, W, cs_);
        fk::rmsnorm_rope(b16(wk_.q), b16(wk_.k), wk_.pos, W, g.num_attention_heads, g.num_key_value_heads,
                         (const float*) L.q_norm.dev, (const float*) L.k_norm.dev, g.rope_theta, g.rms_norm_eps, cs_);
        fk::swa(b16(wk_.q), b16(wk_.k), b16(wk_.v), wk_.pos, cb16(kv_k_[(size_t) l]), cb16(kv_v_[(size_t) l]), W,
                g.num_attention_heads, g.num_key_value_heads, g.head_dim, g.sliding_window, wk_.frontier, scale,
                b16(wk_.attn), cs_, (uint8_t*) wk_.ptrs + (size_t) l * 64u);
        fk::gemm_bf16(cb16(L.o_proj.dev), b16(wk_.attn), b16(wk_.mlp), H, Q, W, cs_);
        fk::conv_out(b16(wk_.residual), b16(wk_.mlp), cb16(L.attn_conv_base.dev), b16(wk_.fdelta), W, H, G, cs_);
        // the MLP branch
        fk::rmsnorm(b16(wk_.residual), (const float*) L.post_attention_norm.dev, b16(wk_.normalized), W, H,
                    g.rms_norm_eps, cs_);
        fk::gemm_bf16(cb16(L.mlp_conv_kernel.dev), b16(wk_.normalized), b16(wk_.kproj2), 4 * G, H, W, cs_);
        fk::conv_in(b16(wk_.normalized), b16(wk_.kproj2), cb16(L.mlp_conv_base.dev), b16(wk_.prepared2),
                    b16(wk_.fdelta2), W, H, G, cs_);
        fk::gemm_bf16(cb16(L.gate.dev), b16(wk_.prepared2), b16(wk_.gate), I, H, W, cs_);
        fk::gemm_bf16(cb16(L.up.dev), b16(wk_.prepared2), b16(wk_.up), I, H, W, cs_);
        fk::swiglu(b16(wk_.gate), b16(wk_.up), b16(wk_.act), W, I, cs_);
        fk::gemm_bf16(cb16(L.down.dev), b16(wk_.act), b16(wk_.mlp), H, I, W, cs_);
        fk::conv_out(b16(wk_.residual), b16(wk_.mlp), cb16(L.mlp_conv_base.dev), b16(wk_.fdelta2), W, H, G, cs_);
    }
    // 3: the pack tail (the final norm over the K draft rows, then the selector's rank projection)
    fk::pack_tail(b16(wk_.residual), fin, b16(wk_.hidden_full), K, W, H, g.rms_norm_eps, cs_);
    fk::gemm_bf16(selh, b16(wk_.hidden_full), b16(wk_.hidden_r), R, H, K, cs_);
    // 4: the head (the Q8_0 logits over the full H-dim hidden; the rank projection feeds the selector only)
    fk::gemm_q8(headw, cb16(wk_.hidden_full), b16(wk_.logits), V, H, K, cs_);
    // 5: the top-16 (the chunked pass + the merge tree)
    {
        float* cur_s = wk_.tk_s[0];
        int* cur_i = wk_.tk_ids[0];
        int C = chunks;
        fk::topk_local(cb16(wk_.logits), V, K, cur_i, cur_s, cs_);
        while (C > 1) {
            int C2 = (C + 1) / 2;
            float* nx_s = (cur_s == wk_.tk_s[0]) ? wk_.tk_s[1] : wk_.tk_s[0];
            int* nx_i = (cur_i == wk_.tk_ids[0]) ? wk_.tk_ids[1] : wk_.tk_ids[0];
            fk::topk_merge(cur_i, cur_s, C, K, nx_i, nx_s, cs_);
            cur_s = nx_s;
            cur_i = nx_i;
            C = C2;
        }
        if (cudaMemcpyAsync(wk_.cand_ids, cur_i, (size_t) K * 16 * 4, cudaMemcpyDeviceToDevice, cs_) !=
            cudaSuccess) {
            err = "dflash2: the graph capture (the candidate ids)";
            cudaStreamEndCapture(cs_, nullptr);
            return false;
        }
        if (cudaMemcpyAsync(wk_.cand_scores, cur_s, (size_t) K * 16 * 4, cudaMemcpyDeviceToDevice, cs_) !=
            cudaSuccess) {
            err = "dflash2: the graph capture (the candidate scores)";
            cudaStreamEndCapture(cs_, nullptr);
            return false;
        }
    }
    // 6: the greedy walk (the anchor from the device ids; the count to the pinned scalar)
    fk::selector_walk(wk_.cand_ids, wk_.cand_scores, cb16(wk_.hidden_r), wk_.ids, pred, succ, wk_.drafts_dev,
                      n_drafts_pin_, K, R, cs_, (uint8_t*) wk_.ptrs + (size_t) g.n_layers * 64u);
    if (cudaStreamEndCapture(cs_, &graph_) != cudaSuccess) {
        err = "dflash2: the graph capture ended";
        graph_ = nullptr;
        return false;
    }
    if (cudaGraphInstantiate(&graph_exec_, graph_, 0) != cudaSuccess) {
        err = "dflash2: the graph instantiate";
        cudaGetLastError();
        return false;
    }
    return true;
}

// ================================ the draft ================================

bool DFlash2Drafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                           float* probs, int* n_drafts) {
    if (!graph_) {
        err = "dflash2: prewarm before draft";
        return false;
    }
    const DFlash2Geometry& g = g_;
    const int W = g.block_size, K = W - 1;
    if (T < 2 || T - 1 > K) {
        err = "dflash2: the window T-1 exceeds the block's K";
        return false;
    }
    if (a < 0 || a >= T) {
        char buf[128];
        std::snprintf(buf, sizeof buf, "dflash2: the accepted count is out of the window (T=%d a=%d p=%lld)", T, a, (long long) p);
        err = buf;
        return false;
    }
    if (frontier_pos_ < p + a) {
        err = "dflash2: the local K/V lags the anchor (append_context first)";
        return false;
    }
    // the block's ids: the anchor then the masks; the positions: the anchor's position forward
    int ids_h[W], pos_h[W];
    ids_h[0] = tokens[a];
    for (int i = 1; i < W; ++i) ids_h[i] = g.mask_token_id;
    const int64_t pa = p + a;
    // the NInfer masked block: the positions clamp at the last visible context column (a sequence
    // shorter than the block piles the tail columns on it)
    const int64_t last = frontier_pos_ - 1;
    for (int i = 0; i < W; ++i) pos_h[i] = (int) (pa + (i < last ? i : last));
    if (cudaMemcpyAsync(wk_.ids, ids_h, (size_t) W * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
        err = "dflash2: the ids upload";
        return false;
    }
    if (cudaMemcpyAsync(wk_.pos, pos_h, (size_t) W * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
        err = "dflash2: the positions upload";
        return false;
    }
    if (cudaGraphLaunch(graph_exec_, cs_) != cudaSuccess) {
        err = "dflash2: the graph launch";
        return false;
    }
    if (cudaMemcpyAsync(drafts, wk_.drafts_dev, (size_t) K * 4, cudaMemcpyDeviceToHost, cs_) != cudaSuccess) {
        err = "dflash2: the drafts download";
        return false;
    }
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "dflash2: the draft stream";
        cudaGetLastError();
        return false;
    }
    const int n = *n_drafts_pin_;
    if (n < 0 || n > K) {
        err = "dflash2: the walk's count is out of range";
        return false;
    }
    if (probs)
        for (int i = 0; i < n; ++i) probs[i] = 1.0f;   // the greedy walk keeps no probabilities (yet)
    if (n_drafts) *n_drafts = n;
    return true;
}

// ================================ the context append ================================

bool DFlash2Drafter::append_context(const float* features, int n, int64_t pos0, std::string& err) {
    if (!dense_) {
        err = "dflash2: append_context before bind";
        return false;
    }
    if (n <= 0 || n > 4096) {
        err = "dflash2: the context length is out of range";
        return false;
    }
    const DFlash2Geometry& g = g_;
    const int H = g.hidden_size, KV = g.kv_width();
    const int ntap = (int) g.target_layer_ids.size();
    const auto* fc = cb16(w_.feature_projection.dev);
    const auto* ctxn = (const float*) w_.context_norm.dev;
    // the features cross to the fc input as bf16 (the host conversion is a few MiB at most)
    std::vector<uint16_t> xh((size_t) 512 * ntap * H);
    auto to_bf16 = [](float f) {
        uint32_t u;
        memcpy(&u, &f, 4);
        u += 0x7fffu + ((u >> 16) & 1u);
        return (uint16_t) (u >> 16);
    };
    for (int c0 = 0; c0 < n; c0 += 512) {
        const int nn = std::min(512, n - c0);
        for (int i = 0; i < nn * ntap * H; ++i) xh[(size_t) i] = to_bf16(features[(size_t) i]);
        if (cudaMemcpyAsync(wk_.ctx_x, xh.data(), (size_t) nn * ntap * H * 2, cudaMemcpyHostToDevice, cs_) !=
            cudaSuccess) {
            char buf[256];
            int dev = -1;
            cudaGetDevice(&dev);
            std::snprintf(buf, sizeof buf, "dflash2: the features upload: %s (ctx_x=%p arena=%p bytes=%llu nn=%d dev=%d)",
                          cudaGetErrorString(cudaGetLastError()), (void*) wk_.ctx_x, (void*) arena_,
                          (unsigned long long) arena_bytes_, nn, dev);
            err = buf;
            return false;
        }
        int pos_h[512];
        for (int i = 0; i < nn; ++i) pos_h[i] = (int) (pos0 + c0 + i);
        if (cudaMemcpyAsync(wk_.ctx_pos, pos_h, (size_t) nn * 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = std::string("dflash2: the positions upload: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        fk::gemm_bf16_wide(fc, cb16(wk_.ctx_x), b16(wk_.ctx), H, ntap * H, nn, cs_);
        fk::rmsnorm(cb16(wk_.ctx), ctxn, b16(wk_.ctx), nn, H, g.rms_norm_eps, cs_);
        for (int l = 0; l < g.n_layers; ++l) {
            const DFlash2LayerW& L = w_.layers[(size_t) l];
            fk::gemm_bf16_wide(cb16(L.k.dev), cb16(wk_.ctx), b16(wk_.ctx_k), KV, H, nn, cs_);
            fk::rmsnorm_rope(nullptr, b16(wk_.ctx_k), wk_.ctx_pos, nn, 0, g.num_key_value_heads, nullptr,
                             (const float*) L.k_norm.dev, g.rope_theta, g.rms_norm_eps, cs_);
            fk::gemm_bf16_wide(cb16(L.v.dev), cb16(wk_.ctx), b16(wk_.ctx_v), KV, H, nn, cs_);
            fk::ctx_store(cb16(wk_.ctx_k), cb16(wk_.ctx_v), pos0 + c0, nn, g.sliding_window, KV,
                          b16(kv_k_[(size_t) l]), b16(kv_v_[(size_t) l]), cs_);
        }
    }
    frontier_pos_ = pos0 + n;
    if (cudaMemcpyAsync(wk_.frontier, &frontier_pos_, 4, cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
        err = "dflash2: the frontier upload";
        return false;
    }
    if (cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "dflash2: the context stream";
        cudaGetLastError();
        return false;
    }
    return true;
}

const void* DFlash2Drafter::slot(const char* name) const {
    if (std::strcmp(name, "residual") == 0) return wk_.residual;
    if (std::strcmp(name, "logits") == 0) return wk_.logits;
    if (std::strcmp(name, "hidden_full") == 0) return wk_.hidden_full;
    if (std::strcmp(name, "hidden_r") == 0) return wk_.hidden_r;
    if (std::strcmp(name, "cand_ids") == 0) return wk_.cand_ids;
    if (std::strcmp(name, "kv_k_0") == 0) return kv_k_[0];
    if (std::strcmp(name, "kv_v_0") == 0) return kv_v_[0];
    return nullptr;
}

}  // namespace basalt::core
