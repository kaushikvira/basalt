// src/core/dflash2/forward_kernels.h - the DFlash2 forward's kernel entry points (forward.cu's), shared with
// the host orchestration (forward.cpp).  The activations are [N, dim] row-major BF16; the drafter's weights
// are the pool's [rows, cols] row-major BF16.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace basalt::core::dflash2_detail {

using bf16 = __nv_bfloat16;

/// C [N, M] = W [M, K] @ x [N, K], N <= 16 (the block's columns).
void gemm_bf16(const bf16* w, const bf16* x, bf16* c, int M, int K, int N, cudaStream_t s);
/// C [N, M] = W [M, K] @ x [N, K], N <= 4096 (the context append).
void gemm_bf16_wide(const bf16* w, const bf16* x, bf16* c, int M, int K, int N, cudaStream_t s);
/// C [N, M] = head [K, M] Q8_0 @ x [N, K], N <= 16 (the target's output head).
void gemm_q8(const uint8_t* w, const bf16* x, bf16* c, int M, int K, int N, cudaStream_t s);

/// rmsnorm: y [N, H] = x * rsqrt(mean(x^2) + eps) * w (w [H] F32).
void rmsnorm(const bf16* x, const float* w, bf16* y, int N, int H, float eps, cudaStream_t s);
/// The per-head (128-dim) rmsnorm + RoPE over the q and k heads (the NInfer rmsnorm_rope contract),
/// in place.  The weights are [128] BF16 each; `qw` may be null (k only).
void rmsnorm_rope(bf16* q, bf16* k, const int* pos, int N, int NHq, int NHk, const float* qw, const float* kw,
                  float theta, float eps, cudaStream_t s);
/// The input side of the dynamic conv (prepared [N, H] + the output deltas fd [N, 2G]).
void conv_in(const bf16* x, const bf16* d, const bf16* base, bf16* prepared, bf16* fd, int N, int H, int G,
             cudaStream_t s);
/// The output side: residual [N, H] += the conv of z.
void conv_out(bf16* residual, const bf16* z, const bf16* base, const bf16* fd, int N, int H, int G, cudaStream_t s);
/// SwiGLU: out [N, I] = silu(gate) * up.
void swiglu(const bf16* gate, const bf16* up, bf16* out, int N, int I, cudaStream_t s);
/// The pack tail: hidden [K, H] = rmsnorm(residual[:, 1:1+K], w).
void pack_tail(const bf16* residual, const float* w, bf16* hidden, int K, int W, int H, float eps, cudaStream_t s);
/// f32 -> bf16, elementwise.
void f32_to_bf16(const float* in, bf16* out, int64_t n, cudaStream_t s);

/// The symmetric sliding-window attention (the block's columns over the local cyclic K/V + the block's K/V).
/// The pointers cross in the 64 B device slot `ptrs_dev`, not the kernel params: the driver (615.71.09)
/// corrupts any kernel whose parameter buffer exceeds 64 B.  The slot is written once before the graph
/// capture (`swa_write_slot`); the pointers are fixed for the drafter's lifetime.
void swa(const bf16* q, const bf16* kb, const bf16* vb, const int* pos, const bf16* kc, const bf16* vc, int W,
         int NHq, int NHk, int D, int window, const int* frontier_dev, float scale, bf16* out, cudaStream_t s,
         void* ptrs_dev);
void swa_write_slot(void* slot, const bf16* q, const bf16* kb, const bf16* vb, const bf16* kc, const bf16* vc,
                    const int* pos, const int* frontier_dev, bf16* out);

/// The top-k's local pass: the V/1024 chunks' top-16 (ids, scores), [chunks, N, 16] each.
void topk_local(const bf16* logits, int V, int N, int* part_ids, float* part_scores, cudaStream_t s);
/// One top-k merge level: in [C, N, 16] -> out [ceil(C/2), N, 16].
void topk_merge(const int* in_i, const float* in_s, int C, int N, int* out_i, float* out_s, cudaStream_t s);

/// The greedy candidate-path walk: the drafts [K] + the count (device, the graph writes it).  The pointers
/// cross in the 64 B device slot `ptrs_dev` (written once before the capture via `sel_write_slot`), not the
/// kernel params.
void selector_walk(const int* ids, const float* unary, const bf16* hidden, const int* anchor_dev, const bf16* pred,
                   const bf16* succ, int* drafts, int* n_drafts, int steps, int R, cudaStream_t s, void* ptrs_dev);
void sel_write_slot(void* slot, const int* ids, const float* unary, const bf16* hidden, const int* anchor_dev,
                    const bf16* pred, const bf16* succ, int* drafts, int* n_drafts);

/// The appended K/V into the cyclic slots (slot (pos0 + i) % window).
void ctx_store(const bf16* k, const bf16* v, int64_t pos0, int n, int window, int KV, bf16* kc, bf16* vc,
               cudaStream_t s);

}  // namespace basalt::core::dflash2_detail
