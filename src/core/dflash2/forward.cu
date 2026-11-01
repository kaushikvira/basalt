// src/core/dflash2/forward.cu - the DFlash2 block-diffusion drafter's kernels (the port of NInfer's
// qwen3_5 DFlash2 forward to the Flash Next geometry, self-contained BF16 GEMMs).
//
// Ported from ~/work/ninfer:
//   execution/draft.cpp            the block-diffusion forward (propose_dflash2_batch) + context append
//   ops/dynamic_grouped_conv/*     the per-layer dynamic convs (generalized: H/G from the geometry)
//   ops/softmax_attention/...      the symmetric sliding-window attention over the local cyclic K/V
//   ops/normalization/rmsnorm      the rmsnorm / pack-tail
//   ops/sampling/candidate_path    the greedy candidate-path selector walk
//
// NInfer's shape-specialized kernels (H=5120, G=320, Q8 projections, the fused 5-layer materialize) are
// re-derived here for Flash Next (H=2560, G=160, BF16 projections; the 5 layers run through the same
// kernels).  The math is the NInfer contract (see the op headers there):
//
//   context append (positions [pos0, pos0+n)):
//     ctx = rmsnorm(fc @ features, context_norm)
//     per layer: K = rope(rmsnorm(k_proj @ ctx, k_norm)), V = v_proj @ ctx -> the cyclic slot p % window
//
//   block (W = K + 1 columns: the anchor then K masks), per layer:
//     n  = rmsnorm(residual, norm)
//     d  = kernel_proj @ n                                  (the dynamic coefficients, 4*G rows)
//     x  = (base[0,0] + d[0,0,g]) * n[:,i] + I(i) (base[0,1] + d[0,1,g]) * n[:,i-1]      (the input conv)
//     ... attention / MLP over x ...
//     residual += (base[1,0] + d[1,0,g]) * z[:,i] + I(i) (base[1,1] + d[1,1,g]) * z[:,i-1]  (the output conv)
//
//   selector (greedy): edge[c] = unary[c,i] + sum_r pred_cb[r,pred] * hidden[r,i] * succ_cb[r,cand[c,i]];
//   the draft is the lowest candidate rank attaining max(edge).
//
// Layouts: the activations are [N, dim] row-major BF16 (N the token count, dim the hidden width); the
// drafter's weights are the pool's [rows, cols] row-major BF16; the target's embedding / head are GGUF
// Q8_0 - a row [K] is K/32 blocks of {half d; int8 qs[32]} (34 B), the rows contiguous (2720 B at
// K=2560, 2-byte aligned).
#include "forward_kernels.h"

#include "basalt/core/dflash2.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace basalt::core {
namespace dflash2_detail {

using bf16 = __nv_bfloat16;

// ================================ the BF16 GEMM ================================
//
// C[n, m] = sum_k W[m, k] x[n, k];  W [M, K] row-major BF16, x [N, K] row-major BF16, C [N, M] row-major
// BF16.  A tile is 64 W-rows x NT columns; 4 warps, warp w the 16 rows 16w.., every warp all NT columns
// (NT/8 mmas of m16n8k16 per 16-wide k-step).  The K runs through three smem stages (cp.async).
// The smem tiles are 16-byte-swizzled per row (unit 8) so the ldmatrix rows stay bank-conflict-free.

constexpr int kGemmRows = 64;

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
    asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1])
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(p))));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm("cp.async.cg.shared.global [%0], [%1], 16;\n"
        ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm("cp.async.commit_group;\n" ::: "memory"); }
template<int N> __device__ __forceinline__ void cp_wait() { asm("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// stage one 16-wide k-slice: the A tile (64 x 16) and the B tile (NT x 16), swizzled
template<int NT, int NTHREADS>
__device__ __forceinline__ void gemm_stage(const bf16* wa, const bf16* xa, bf16* ws, bf16* xs, int n0, int N,
                                           int K) {
    constexpr int kUnits = 8;   // the 16-wide row is two 16 B units; the swizzle permutes the two
    for (int item = threadIdx.x; item < kGemmRows * (16 / kUnits); item += NTHREADS) {
        const int r = item / (16 / kUnits), c = (item % (16 / kUnits)) * kUnits;
        cp16(ws + r * 16 + ((c / kUnits ^ (r & 1)) * kUnits), wa + static_cast<int64_t>(r) * K + c);
    }
    for (int item = threadIdx.x; item < NT * (16 / kUnits); item += NTHREADS) {
        const int r = item / (16 / kUnits), c = (item % (16 / kUnits)) * kUnits;
        const int n = n0 + r;
        // xa already carries the n0*K base; the row offset is r*K (clamp to r=0 past the last row)
        cp16(xs + r * 16 + ((c / kUnits ^ (r & 1)) * kUnits), xa + static_cast<int64_t>((n < N ? r : 0)) * K + c);
    }
}

// the 16x16 A / 16x8 B fragments of k-slice kt from stage st (the NInfer partial_kernel's mapping)
__device__ __forceinline__ void gemm_frags(const bf16* ws, const bf16* xs, int warp, int lane, int m,
                                           uint32_t (&a)[4], uint32_t (&b)[2]) {
    const int ar = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int ak = (lane >> 4) * 8;
    ldmatrix_x4(a, &ws[ar * 16 + (((ak >> 3) ^ (ar & 1)) * 8)]);
    const int br = m * 8 + (lane & 7);
    const int bk = ((lane >> 3) & 1) * 8;
    ldmatrix_x2(b, &xs[br * 16 + (((bk >> 3) ^ (br & 1)) * 8)]);
}

template<int NT>
__global__ void __launch_bounds__(128) gemm_bf16_kernel(const bf16* __restrict__ w, const bf16* __restrict__ x,
                                                        bf16* __restrict__ c, int M, int K, int N) {
    constexpr int kStages = 3;
    __shared__ __align__(16) bf16 ws[kStages][kGemmRows * 16];
    __shared__ __align__(16) bf16 xs[kStages][NT * 16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int m0 = static_cast<int>(blockIdx.x) * kGemmRows, n0 = static_cast<int>(blockIdx.y) * NT;
    const int nk = K / 16;
    auto stage = [&](int st, int kt) {
        gemm_stage<NT, 128>(w + static_cast<int64_t>(m0) * K + kt * 16,
                            x + static_cast<int64_t>(n0) * K + kt * 16, ws[st], xs[st], n0, N, K);
    };
    stage(0, 0);
    cp_commit();
    stage(1, 1);
    cp_commit();
    stage(2, min(2, nk - 1));
    cp_commit();
    float acc[NT / 8][4] = {};
#pragma unroll 1
    for (int kt = 0; kt < nk; ++kt) {
        const int st = kt % kStages;
        if (kt + kStages < nk)
            cp_wait<kStages - 1>();
        else
            cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int m = 0; m < NT / 8; ++m) {
            uint32_t a[4], b[2];
            gemm_frags(ws[st], xs[st], warp, lane, m, a, b);
            mma_bf16(acc[m], a[0], a[1], a[2], a[3], b[0], b[1]);
        }
        __syncthreads();
        if (kt + kStages < nk) stage(st, kt + kStages);
        cp_commit();
    }
    const int row = m0 + warp * 16 + (lane >> 2);
#pragma unroll
    for (int m = 0; m < NT / 8; ++m) {
        const int n = n0 + m * 8 + 2 * (lane & 3);
        if (row < M) {
            if (n < N) c[static_cast<int64_t>(n) * M + row] = __float2bfloat16_rn(acc[m][0]);
            if (n + 1 < N) c[static_cast<int64_t>(n + 1) * M + row] = __float2bfloat16_rn(acc[m][1]);
        }
        if (row + 8 < M) {
            if (n < N) c[static_cast<int64_t>(n) * M + row + 8] = __float2bfloat16_rn(acc[m][2]);
            if (n + 1 < N) c[static_cast<int64_t>(n + 1) * M + row + 8] = __float2bfloat16_rn(acc[m][3]);
        }
    }
}

/// C [N, M] = W [M, K] @ x [N, K], N <= 16 (the block's columns).
void gemm_bf16(const bf16* w, const bf16* x, bf16* c, int M, int K, int N, cudaStream_t s) {
    const dim3 grid((M + kGemmRows - 1) / kGemmRows, (N + 15) / 16);
    gemm_bf16_kernel<16><<<grid, 128, 0, s>>>(w, x, c, M, K, N);
}

/// The context-append GEMM: N up to 4096 columns; 8 warps (warp w the 16 rows 16 (w/4).., the 16 columns
/// 16 (w%4)..).
template<int NT>
__global__ void __launch_bounds__(256) gemm_bf16_wide_kernel(const bf16* __restrict__ w, const bf16* __restrict__ x,
                                                             bf16* __restrict__ c, int M, int K, int N) {
    constexpr int kStages = 3;
    __shared__ __align__(16) bf16 ws[kStages][kGemmRows * 16];
    __shared__ __align__(16) bf16 xs[kStages][NT * 16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int m0 = static_cast<int>(blockIdx.x) * kGemmRows, n0 = static_cast<int>(blockIdx.y) * NT;
    const int nk = K / 16;
    auto stage = [&](int st, int kt) {
        gemm_stage<NT, 256>(w + static_cast<int64_t>(m0) * K + kt * 16,
                            x + static_cast<int64_t>(n0) * K + kt * 16, ws[st], xs[st], n0, N, K);
    };
    stage(0, 0);
    cp_commit();
    stage(1, 1);
    cp_commit();
    stage(2, min(2, nk - 1));
    cp_commit();
    const int rowgrp = warp >> 1, colgrp = warp & 1;   // 4 row-groups x 2 col-groups = 8 warps (64x64)
    float acc[4][4] = {};
#pragma unroll 1
    for (int kt = 0; kt < nk; ++kt) {
        const int st = kt % kStages;
        if (kt + kStages < nk)
            cp_wait<kStages - 1>();
        else
            cp_wait<0>();
        __syncthreads();
        const int ar = rowgrp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int ak = (lane >> 4) * 8;
        uint32_t a[4];
        ldmatrix_x4(a, &ws[st][ar * 16 + (((ak >> 3) ^ (ar & 1)) * 8)]);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            uint32_t b[2];
            const int br = colgrp * 32 + m * 8 + (lane & 7);
            const int bk = ((lane >> 3) & 1) * 8;
            ldmatrix_x2(b, &xs[st][br * 16 + (((bk >> 3) ^ (br & 1)) * 8)]);
            mma_bf16(acc[m], a[0], a[1], a[2], a[3], b[0], b[1]);
        }
        __syncthreads();
        if (kt + kStages < nk) stage(st, kt + kStages);
        cp_commit();
    }
    const int row = m0 + rowgrp * 16 + (lane >> 2);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int n = n0 + colgrp * 32 + m * 8 + 2 * (lane & 3);
        if (row < M && n < N) {
            c[static_cast<int64_t>(n) * M + row] = __float2bfloat16_rn(acc[m][0]);
            c[static_cast<int64_t>(n + 1) * M + row] = __float2bfloat16_rn(acc[m][1]);
        }
        if (row + 8 < M && n < N) {
            c[static_cast<int64_t>(n) * M + row + 8] = __float2bfloat16_rn(acc[m][2]);
            c[static_cast<int64_t>(n + 1) * M + row + 8] = __float2bfloat16_rn(acc[m][3]);
        }
    }
}

/// C [N, M] = W [M, K] @ x [N, K], N <= 4096 (the context append).
void gemm_bf16_wide(const bf16* w, const bf16* x, bf16* c, int M, int K, int N, cudaStream_t s) {
    const dim3 grid((M + kGemmRows - 1) / kGemmRows, (N + 63) / 64);
    gemm_bf16_wide_kernel<64><<<grid, 256, 0, s>>>(w, x, c, M, K, N);
}

// ================================ the Q8_0 head GEMM ================================
//
// The target's output head: W [K, M] GGUF Q8_0 - M rows of K/32 blocks {half d; int8[32]} (34 B), the rows
// contiguous (K/32 * 34 B each).  C[n, m] = sum_k W[k, m] x[n, k].  A 64-row tile streams the K in 128-wide
// chunks: each chunk's 64 rows dequant to BF16 in smem (8.7 KB, 256 blocks over 128 threads), then the same
// mma path as the BF16 GEMM.  The qs words are 2-byte aligned at best (34-B blocks), so they load as 8-byte
// halves.

struct q8_block {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(q8_block) == 34);
// C [N, M] = head [M, K] Q8_0 @ x [N, K].  N is tiny (the block's K, 7), M is huge (the vocab, 248k),
// K = the hidden width (2560).  One block per weight row m: dequantize the Q8_0 row to f32 smem ONCE,
// then compute all N dot products (the row's 2720 B are read once per m, not N times).  The old
// one-thread-per-output kernel read the row N times (4.7 GB of traffic) at 32 threads/block and took
// ~72 ms; this is ~1 ms.
template<int NOUT, int NTHREADS, int KMAX>
__global__ void __launch_bounds__(NTHREADS) gemm_q8_kernel(const uint8_t* __restrict__ w,
                                                           const bf16* __restrict__ x,
                                                           bf16* __restrict__ c, int M, int K) {
    constexpr int NWARP = NTHREADS / 32;
    const int m = static_cast<int>(blockIdx.x);
    const int kRowB = K / 32 * 34;
    const uint8_t* wrow = w + static_cast<int64_t>(m) * kRowB;
    __shared__ float sw[KMAX];
    __shared__ float red[NWARP][NOUT];
    for (int i = threadIdx.x; i < K; i += NTHREADS) {
        const int blk = i >> 5, q = i & 31;
        const uint8_t* src = wrow + blk * 34;
        const uint16_t sb = *reinterpret_cast<const uint16_t*>(src);
        const float d = __half2float(*reinterpret_cast<const __half*>(&sb));
        sw[i] = d * static_cast<float>(reinterpret_cast<const int8_t*>(src + 2)[q]);
    }
    __syncthreads();
    float acc[NOUT] = {};
    for (int i = threadIdx.x; i < K; i += NTHREADS) {
        const float wv = sw[i];
#pragma unroll
        for (int n = 0; n < NOUT; ++n)
            acc[n] = fmaf(wv, __bfloat162float(x[(size_t) n * K + i]), acc[n]);
    }
#pragma unroll
    for (int n = 0; n < NOUT; ++n) {
        float v = acc[n];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5][n] = v;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
#pragma unroll
        for (int n = 0; n < NOUT; ++n) {
            float v = (threadIdx.x < NWARP) ? red[threadIdx.x][n] : 0.0f;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (threadIdx.x == 0) c[(size_t) n * M + m] = __float2bfloat16_rn(v);
        }
    }
}

/// C [N, M] = head [M, K] Q8_0 @ x [N, K], N <= 16.
void gemm_q8(const uint8_t* w, const bf16* x, bf16* c, int M, int K, int N, cudaStream_t s) {
    const dim3 grid(M);
    gemm_q8_kernel<7, 256, 2560><<<grid, 256, 0, s>>>(w, x, c, M, K);
}

// ================================ the elementwise ops ================================

/// rmsnorm: x [N, H] BF16 -> y [N, H] BF16, y = x * rsqrt(mean(x^2) + eps) * w (w [H] F32).
__global__ void __launch_bounds__(256) rmsnorm_kernel(const bf16* __restrict__ x, const float* __restrict__ w,
                                                      bf16* __restrict__ y, int H, float eps) {
    const int n = static_cast<int>(blockIdx.x);
    const bf16* in = x + static_cast<int64_t>(n) * H;
    bf16* out = y + static_cast<int64_t>(n) * H;
    __shared__ float part[8];
    float s = 0.0f;
    for (int h = threadIdx.x; h < H; h += 256) {
        const float v = __bfloat162float(in[h]);
        s += v * v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) < 8) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 8) {
        float t = part[threadIdx.x];
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) t += __shfl_xor_sync(0xffu, t, o);
        if (threadIdx.x == 0) part[0] = rsqrtf(t * (1.0f / H) + eps);
    }
    __syncthreads();
    const float inv = part[0];
    for (int h = threadIdx.x; h < H; h += 256)
        out[h] = __float2bfloat16_rn(__bfloat162float(in[h]) * inv * w[h]);
}

void rmsnorm(const bf16* x, const float* w, bf16* y, int N, int H, float eps, cudaStream_t s) {
    rmsnorm_kernel<<<N, 256, 0, s>>>(x, w, y, H, eps);
}

/// The per-head (128-dim) rmsnorm + RoPE over the q and k heads, one rounding (the NInfer
/// rmsnorm_rope contract): each head is normalized with its 128-dim BF16 weight, the pair (j, j + 64)
/// rotates by pos * theta^(-2j/128), j < 64.  q [N, NHq*128], k [N, NHk*128] BF16 (in place),
/// positions [N] I32 (device), the weights [128] BF16.
__global__ void __launch_bounds__(64) rmsnorm_rope_kernel(bf16* __restrict__ q, bf16* __restrict__ k,
                                                           const int* __restrict__ pos,
                                                           const float* __restrict__ qw,
                                                           const float* __restrict__ kw, int NHq, float theta,
                                                           float eps) {
    const int n = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const bool isq = head < NHq;
    const int local = isq ? head : head - NHq;   // the head's index within its own tensor
    const float* w = isq ? qw : kw;   // the per-head norm weights are F32
    bf16* v = (isq ? q : k) + (static_cast<int64_t>(n) * (isq ? NHq : (int)(gridDim.y - NHq)) + local) * 128;
    const int d = threadIdx.x;   // 0..63: the rope pair (d, d+64)
    const float x0 = __bfloat162float(v[d]);
    const float x1 = __bfloat162float(v[d + 64]);
    float ss = x0 * x0 + x1 * x1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    __shared__ float part[2], tot;
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) tot = rsqrtf((part[0] + part[1]) * (1.0f / 128.0f) + eps);
    __syncthreads();
    const float inv = tot;
    const float p = static_cast<float>(pos[n]);
    const float ang = p * powf(theta, -2.0f * static_cast<float>(d) / 128.0f);
    const float c = cosf(ang), sn = sinf(ang);
    const float y0 = __bfloat162float(__float2bfloat16_rn(x0 * inv * w[d]));
    const float y1 = __bfloat162float(__float2bfloat16_rn(x1 * inv * w[d + 64]));
    v[d]     = __float2bfloat16_rn(y0 * c - y1 * sn);
    v[d + 64] = __float2bfloat16_rn(y0 * sn + y1 * c);
}

void rmsnorm_rope(bf16* q, bf16* k, const int* pos, int N, int NHq, int NHk, const float* qw,
                         const float* kw, float theta, float eps, cudaStream_t s) {
    const dim3 grid(N, NHq + NHk);
    rmsnorm_rope_kernel<<<grid, 64, 0, s>>>(q, k, pos, qw, kw, NHq, theta, eps);
}


/// The input side of the dynamic conv: prepared [N, H] = (base[0,0] + d[n,g]) * x[n,h] + I(n) (base[0,1] +
/// d[n,G+g]) * x[n-1,h]; and finish_delta [N, 2, G] = d[n, 2G+g], d[n, 3G+g] (the output side's deltas).
/// x [N, H] BF16 (the normalized residual), d [N, 4G] BF16 (the projection's rows), base [2, 2, H] BF16
/// (side, tap, channel).  G = H/16.
__global__ void __launch_bounds__(256) conv_in_kernel(const bf16* __restrict__ x, const bf16* __restrict__ d,
                                                      const bf16* __restrict__ base, bf16* __restrict__ prepared,
                                                      bf16* __restrict__ fd, int H, int G) {
    const int n = static_cast<int>(blockIdx.x);
    const bf16* xn = x + static_cast<int64_t>(n) * H;
    const bf16* xp = n > 0 ? xn - H : nullptr;
    const bf16* dn = d + static_cast<int64_t>(n) * 4 * G;
    bf16* out = prepared + static_cast<int64_t>(n) * H;
    bf16* fdo = fd + static_cast<int64_t>(n) * 2 * G;
    for (int i = threadIdx.x; i < 2 * G; i += 256)
        fdo[i] = dn[2 * G + i];
    for (int h = threadIdx.x; h < H; h += 256) {
        const int g = h >> 4;
        float v = __bfloat162float(__float2bfloat16_rn(
            (__bfloat162float(base[h]) + __bfloat162float(dn[g])) * __bfloat162float(xn[h])));
        if (n > 0)
            v += __bfloat162float(__float2bfloat16_rn(
                (__bfloat162float(base[H + h]) + __bfloat162float(dn[G + g])) * __bfloat162float(xp[h])));
        out[h] = __float2bfloat16_rn(v);
    }
}

void conv_in(const bf16* x, const bf16* d, const bf16* base, bf16* prepared, bf16* fd, int N, int H, int G,
                    cudaStream_t s) {
    conv_in_kernel<<<N, 256, 0, s>>>(x, d, base, prepared, fd, H, G);
}

/// The output side: residual [N, H] += (base[1,0] + fd[n,0,g]) * z[n,h] + I(n) (base[1,1] + fd[n,1,g]) * z[n-1,h].
__global__ void __launch_bounds__(256) conv_out_kernel(bf16* __restrict__ residual, const bf16* __restrict__ z,
                                                       const bf16* __restrict__ base, const bf16* __restrict__ fd,
                                                       int H, int G) {
    const int n = static_cast<int>(blockIdx.x);
    bf16* r = residual + static_cast<int64_t>(n) * H;
    const bf16* zn = z + static_cast<int64_t>(n) * H;
    const bf16* zp = n > 0 ? zn - H : nullptr;
    const bf16* fdo = fd + static_cast<int64_t>(n) * 2 * G;
    for (int h = threadIdx.x; h < H; h += 256) {
        const int g = h >> 4;
        // the NInfer add: base + delta is rounded to bf16 before the fma
        const float d0 = __bfloat162float(
            __float2bfloat16_rn(__bfloat162float(base[2 * H + h]) + __bfloat162float(fdo[g])));
        float v = d0 * __bfloat162float(zn[h]);
        if (n > 0) {
            const float d1 = __bfloat162float(
                __float2bfloat16_rn(__bfloat162float(base[3 * H + h]) + __bfloat162float(fdo[G + g])));
            v += d1 * __bfloat162float(zp[h]);   // the reference rounds the product, not an fma
        }
        r[h] = __float2bfloat16_rn(__bfloat162float(r[h]) + v);
    }
}

void conv_out(bf16* residual, const bf16* z, const bf16* base, const bf16* fd, int N, int H, int G,
                     cudaStream_t s) {
    conv_out_kernel<<<N, 256, 0, s>>>(residual, z, base, fd, H, G);
}

/// SwiGLU: out [N, I] = silu(gate) * up (all [N, I] BF16).
__global__ void __launch_bounds__(256) swiglu_kernel(const bf16* __restrict__ gate, const bf16* __restrict__ up,
                                                     bf16* __restrict__ out, int I) {
    const int n = static_cast<int>(blockIdx.x);
    const bf16* g = gate + static_cast<int64_t>(n) * I;
    const bf16* u = up + static_cast<int64_t>(n) * I;
    bf16* o = out + static_cast<int64_t>(n) * I;
    for (int i = threadIdx.x; i < I; i += 256) {
        const float gv = __bfloat162float(g[i]);
        o[i] = __float2bfloat16_rn(gv / (1.0f + __expf(-gv)) * __bfloat162float(u[i]));
    }
}

void swiglu(const bf16* gate, const bf16* up, bf16* out, int N, int I, cudaStream_t s) {
    swiglu_kernel<<<N, 256, 0, s>>>(gate, up, out, I);
}

/// The pack tail: hidden [K, H] = rmsnorm(residual[:, 1:1+K], w) (the anchor column dropped).
__global__ void __launch_bounds__(256) pack_tail_kernel(const bf16* __restrict__ residual, const float* __restrict__ w,
                                                        bf16* __restrict__ hidden, int W, int H, float eps) {
    const int k = static_cast<int>(blockIdx.x);
    const bf16* in = residual + static_cast<int64_t>(k + 1) * H;
    bf16* out = hidden + static_cast<int64_t>(k) * H;
    __shared__ float part[8];
    float s = 0.0f;
    for (int h = threadIdx.x; h < H; h += 256) {
        const float v = __bfloat162float(in[h]);
        s += v * v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) < 8) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 8) {
        float t = part[threadIdx.x];
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) t += __shfl_xor_sync(0xffu, t, o);
        if (threadIdx.x == 0) part[0] = rsqrtf(t * (1.0f / H) + eps);
    }
    __syncthreads();
    const float inv = part[0];
    for (int h = threadIdx.x; h < H; h += 256)
        out[h] = __float2bfloat16_rn(__bfloat162float(in[h]) * inv * w[h]);
}

void pack_tail(const bf16* residual, const float* w, bf16* hidden, int K, int W, int H, float eps,
                      cudaStream_t s) {
    pack_tail_kernel<<<K, 256, 0, s>>>(residual, w, hidden, W, H, eps);
}

/// The embedding: out [W, H] = the Q8_0 rows of ids [W] (device), emb [K, M] Q8_0 (the rows contiguous).
__global__ void __launch_bounds__(256) embed_kernel(const uint8_t* __restrict__ emb, const int* __restrict__ ids,
                                                    bf16* __restrict__ out, int H) {
    const int n = static_cast<int>(blockIdx.x);
    const int id = ids[n];
    const uint8_t* row = emb + static_cast<int64_t>(id) * (H / 32 * 34);
    bf16* o = out + static_cast<int64_t>(n) * H;
    // 256 threads x 10 values: thread t takes values 10t..10t+9 (at most two 32-value blocks)
    const int v0 = threadIdx.x * 10;
    for (int q = 0; q < 10; ++q) {
        const int v = v0 + q;
        const q8_block* b = reinterpret_cast<const q8_block*>(row + (v / 32) * 34);
        const float d = __half2float(*reinterpret_cast<const __half*>(&b->d));
        o[v] = __float2bfloat16_rn(d * static_cast<float>(b->qs[v % 32]));
    }
}

inline void embed(const uint8_t* emb, const int* ids, bf16* out, int W, int H, cudaStream_t s) {
    embed_kernel<<<W, 256, 0, s>>>(emb, ids, out, H);
}

/// f32 -> bf16, elementwise (the embedding gather scratch to the residual form).
__global__ void __launch_bounds__(256) f32_to_bf16_kernel(const float* __restrict__ in, bf16* __restrict__ out,
                                                          int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16_rn(in[i]);
}

void f32_to_bf16(const float* in, bf16* out, int64_t n, cudaStream_t s) {
    f32_to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(in, out, n);
}

// ================================ the sliding-window attention ================================
//
// Symmetric (non-causal) GQA over the local cyclic K/V + the block's own K/V: a key at position p_j is
// visible to the query at p_i exactly when abs(p_j - p_i) < window (the NInfer contract).  One CTA per block
// column: 128 threads, warp w the q-heads 4w..4w+3 (lane l the 4 dims 4l..4l+3 of each head), online softmax
// in FP32, the block's K/V last (every block key is within the window of every block query).

// The pointers cross in device slots, not the kernel params: the driver (615.71.09) corrupts the
// constant bank of any kernel whose parameter buffer exceeds 64 B (2026-11-01, the 88 B args below read a
// stale driver pointer in place of the real one).  One 64 B slot per swa call + one for the walk, written
// before the graph capture (the pointers are fixed for the drafter's lifetime, so the graph needs no
// staging nodes); the kernels' own params stay 32 B / 16 B.
struct SwaPtrs {
    const bf16* q;
    const bf16* kb;
    const bf16* vb;
    const bf16* kc;
    const bf16* vc;
    const int* pos;
    const int* frontier_dev;   // the visible prefix (device: the graph bakes the pointers, not the values)
    bf16* out;
};

__global__ void __launch_bounds__(256) swa_body(const SwaPtrs* __restrict__ ptrs, int W, int NHq, int NHk,
                                                int D, int window, float scale) {
    const bf16* __restrict__ q = ptrs->q;
    const bf16* __restrict__ kb = ptrs->kb;
    const bf16* __restrict__ vb = ptrs->vb;
    const bf16* __restrict__ kc = ptrs->kc;
    const bf16* __restrict__ vc = ptrs->vc;
    const int* __restrict__ pos = ptrs->pos;
    const int* __restrict__ frontier_dev = ptrs->frontier_dev;
    bf16* __restrict__ out = ptrs->out;
    const int i = static_cast<int>(blockIdx.x);
    const int pi = pos[i];
    const int frontier = *frontier_dev;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hq0 = warp * 4;
    const int hk = hq0 / (NHq / NHk);
    const int d0 = lane * 4;
    __shared__ float score[8][4];
    float qv[4][4];
    const bf16* qr = q + static_cast<int64_t>(i) * NHq * D + hq0 * D + d0;
#pragma unroll
    for (int h = 0; h < 4; ++h)
#pragma unroll
        for (int d = 0; d < 4; ++d) qv[h][d] = __bfloat162float(qr[h * D + d]);
    float m[4] = {}, l[4] = {}, acc[4][4] = {};
    auto key = [&](int p, bool block) {
        const int slot = block ? (p - pos[0]) : (p & (window - 1));
        const bf16* k = (block ? kb : kc) + (static_cast<int64_t>(slot) * NHk + hk) * D + d0;
        float s[4] = {};
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < 4; ++d) dot += qv[h][d] * __bfloat162float(k[d]);   // the 4 q-heads share one k-head (GQA)
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            s[h] = dot;
        }
        if (lane == 0)
#pragma unroll
            for (int h = 0; h < 4; ++h) score[warp][h] = s[h] * scale;
        __syncthreads();
        const float s0 = score[warp][0], s1 = score[warp][1], s2 = score[warp][2], s3 = score[warp][3];
        const float mn = fmaxf(fmaxf(fmaxf(fmaxf(m[0], s0), fmaxf(m[1], s1)), fmaxf(m[2], s2)), s3);
        const float r0 = __expf(m[0] - mn), r1 = __expf(m[1] - mn), r2 = __expf(m[2] - mn), r3 = __expf(m[3] - mn);
        const float e0 = __expf(s0 - mn), e1 = __expf(s1 - mn), e2 = __expf(s2 - mn), e3 = __expf(s3 - mn);
        const bf16* v = (block ? vb : vc) + (static_cast<int64_t>(slot) * NHk + hk) * D + d0;
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const float r = h == 0 ? r0 : h == 1 ? r1 : h == 2 ? r2 : r3;
            const float e = h == 0 ? e0 : h == 1 ? e1 : h == 2 ? e2 : e3;
            l[h] = l[h] * r + e;
#pragma unroll
            for (int d = 0; d < 4; ++d) acc[h][d] = acc[h][d] * r + e * __bfloat162float(v[d]);   // the 4 q-heads share one v-head (GQA)
        }
        m[0] = mn;
        m[1] = mn;
        m[2] = mn;
        m[3] = mn;
        __syncthreads();
    };
    const int lo = max(0, pi - window + 1);
    for (int p = lo; p < frontier; ++p) key(p, false);
    for (int j = 0; j < W; ++j) key(pos[j], true);
    bf16* o = out + static_cast<int64_t>(i) * NHq * D + hq0 * D + d0;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float inv = l[h] > 0.0f ? 1.0f / l[h] : 0.0f;
#pragma unroll
        for (int d = 0; d < 4; ++d) o[h * D + d] = __float2bfloat16_rn(acc[h][d] * inv);
    }
}

// ptrs_dev: this call's 64 B arena slot (written before the capture; the kernel reads it at every replay).
void swa(const bf16* q, const bf16* kb, const bf16* vb, const int* pos, const bf16* kc, const bf16* vc,
                int W, int NHq, int NHk, int D, int window, const int* frontier_dev, float scale, bf16* out,
                cudaStream_t s, void* ptrs_dev) {
    swa_body<<<W, 256, 0, s>>>((const SwaPtrs*) ptrs_dev, W, NHq, NHk, D, window, scale);
}

/// The synchronous slot write (the default stream; called once before the graph capture - the pointers are
/// fixed for the drafter's lifetime).
void swa_write_slot(void* slot, const bf16* q, const bf16* kb, const bf16* vb, const bf16* kc, const bf16* vc,
                    const int* pos, const int* frontier_dev, bf16* out) {
    SwaPtrs p{q, kb, vb, kc, vc, pos, frontier_dev, out};
    if (cudaMemcpy(slot, &p, sizeof p, cudaMemcpyHostToDevice) != cudaSuccess)
        std::abort();
}

// ================================ the top-k ================================
//
// Per column n of logits [N, V] BF16: the top-KT (id, score).  Pass 1: one CTA per (V/1024 chunk, column)
// keeps its chunk's local top-KT (a shared sorted list, thread 0 inserts over the chunk's 1024 values,
// staged in smem by the 256 threads); pass 2: the tree merge of the chunk lists (two sorted 16-lists ->
// one, per (pair, column)).

constexpr int kTopK = 16;
constexpr float kNegInf = -3.4e38f;

__global__ void __launch_bounds__(256) topk_local_kernel(const bf16* __restrict__ logits, int V, int N,
                                                         int* __restrict__ part_ids, float* __restrict__ part_scores) {
    const int n = static_cast<int>(blockIdx.y);
    const int chunk = static_cast<int>(blockIdx.x);
    const bf16* row = logits + static_cast<int64_t>(n) * V + static_cast<int64_t>(chunk) * 1024;
    const int valid = min(1024, V - chunk * 1024);   // the last chunk is partial (V % 1024)
    __shared__ float sv[kTopK];
    __shared__ int si[kTopK];
    __shared__ float svals[1024];
    if (threadIdx.x < kTopK) {
        sv[threadIdx.x] = kNegInf;   // the empty slots sort last (the insertion is `s > sv[j]`)
        si[threadIdx.x] = -1;
    }
    // every value of the chunk must cross the top-k: the 256 threads stage the 1024 values (4 each;
    // the partial tail of the last chunk is padded with -inf, not the next row's bytes), then thread 0
    // walks the staged list into the sorted top-KT
    const int v0 = 4 * threadIdx.x;
    svals[v0 + 0] = (v0 + 0 < valid) ? __bfloat162float(row[v0 + 0]) : kNegInf;
    svals[v0 + 1] = (v0 + 1 < valid) ? __bfloat162float(row[v0 + 1]) : kNegInf;
    svals[v0 + 2] = (v0 + 2 < valid) ? __bfloat162float(row[v0 + 2]) : kNegInf;
    svals[v0 + 3] = (v0 + 3 < valid) ? __bfloat162float(row[v0 + 3]) : kNegInf;
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll 4
        for (int v = 0; v < valid; ++v) {
            const float s = svals[v];
            const int id = chunk * 1024 + v;
            int j = kTopK - 1;
            if (s > sv[j]) {
                while (j > 0 && sv[j - 1] < s) {
                    sv[j] = sv[j - 1];
                    si[j] = si[j - 1];
                    --j;
                }
                sv[j] = s;
                si[j] = id;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < kTopK) {
        part_ids[(static_cast<int64_t>(chunk) * N + n) * kTopK + threadIdx.x] = si[threadIdx.x];
        part_scores[(static_cast<int64_t>(chunk) * N + n) * kTopK + threadIdx.x] = sv[threadIdx.x];
    }
}

/// One merge level: in [C, N, KT] (sorted descending) -> out [ceil(C/2), N, KT]; pairs (2c, 2c+1) merge (one
/// thread per output slot runs the two-pointer merge to its slot), an odd last chunk copies through.
__global__ void __launch_bounds__(256) topk_merge_kernel(const int* __restrict__ in_i, const float* __restrict__ in_s,
                                                         int C, int N, int* __restrict__ out_i,
                                                         float* __restrict__ out_s) {
    const int c2 = static_cast<int>(blockIdx.x);
    const int n = static_cast<int>(blockIdx.y);
    int* o_i = out_i + (static_cast<int64_t>(c2) * N + n) * kTopK;
    float* o_s = out_s + (static_cast<int64_t>(c2) * N + n) * kTopK;
    const int slot = threadIdx.x;
    if (slot >= kTopK) return;
    if (2 * c2 + 1 >= C) {   // the odd last chunk: copy through
        o_i[slot] = in_i[(static_cast<int64_t>(2 * c2) * N + n) * kTopK + slot];
        o_s[slot] = in_s[(static_cast<int64_t>(2 * c2) * N + n) * kTopK + slot];
        return;
    }
    const int* a_i = in_i + (static_cast<int64_t>(2 * c2) * N + n) * kTopK;
    const float* a_s = in_s + (static_cast<int64_t>(2 * c2) * N + n) * kTopK;
    const int* b_i = in_i + (static_cast<int64_t>(2 * c2 + 1) * N + n) * kTopK;
    const float* b_s = in_s + (static_cast<int64_t>(2 * c2 + 1) * N + n) * kTopK;
    int ia = 0, ib = 0;
    for (int step = 0; step <= slot; ++step) {
        // the exhausted side shows -inf (it can never win the merge); at most one side is exhausted at
        // step <= 15 (the two lists hold 32 entries together)
        const float sa = ia < kTopK ? a_s[ia] : kNegInf;
        const float sb = ib < kTopK ? b_s[ib] : kNegInf;
        if (sa >= sb && ia < kTopK) {
            o_i[slot] = a_i[ia];
            o_s[slot] = sa;
            ++ia;
        } else if (ib < kTopK) {
            o_i[slot] = b_i[ib];
            o_s[slot] = sb;
            ++ib;
        }
    }
}

void topk_local(const bf16* logits, int V, int N, int* part_ids, float* part_scores, cudaStream_t s) {
    const dim3 grid((V + 1023) / 1024, N);
    topk_local_kernel<<<grid, 256, 0, s>>>(logits, V, N, part_ids, part_scores);
}

void topk_merge(const int* in_i, const float* in_s, int C, int N, int* out_i, float* out_s, cudaStream_t s) {
    const dim3 grid((C + 1) / 2, N);
    topk_merge_kernel<<<grid, 256, 0, s>>>(in_i, in_s, C, N, out_i, out_s);
}

// ================================ the selector walk ================================
//
// The greedy candidate-path walk (the NInfer selector_walk_kernel, the sampling path dropped): one CTA,
// 512 threads (16 warps = the 16 candidates), K steps.  edge[c] = unary[c, i] + sum_r pred_cb[r, pred] *
// hidden[r, i] * succ_cb[r, cand[c, i]]; the draft is the lowest candidate rank attaining max(edge).
// R = the selector rank (256).

// The 72 B SelArgs crossed the driver's 64 B parameter limit (the same constant-bank bug as swa's);
// the pointers ride a 64 B arena slot, the kernel's own params are 16 B.
struct SelPtrs {
    const int* ids;        // [K, 16] the candidates (rank fastest)
    const float* unary;    // [K, 16]
    const bf16* hidden;    // [K, R]
    const int* anchor_dev; // the anchor token (device: the graph bakes the pointers, not the values)
    const bf16* pred;      // [V, R]
    const bf16* succ;      // [V, R]
    int* drafts;           // [K]
    int* n_drafts;
};

struct alignas(16) SelShared {
    bf16 succ[kTopK][256];
    float product[256];
    float edge[kTopK];
    int predecessor;
};

__global__ void __launch_bounds__(512, 1) selector_walk_kernel(const SelPtrs* __restrict__ ptrs, int steps, int R) {
    const int* __restrict__ ids = ptrs->ids;
    const float* __restrict__ unary = ptrs->unary;
    const bf16* __restrict__ hidden = ptrs->hidden;
    const int* __restrict__ anchor_dev = ptrs->anchor_dev;
    const bf16* __restrict__ pred = ptrs->pred;
    const bf16* __restrict__ succ = ptrs->succ;
    int* __restrict__ drafts = ptrs->drafts;
    int* __restrict__ n_drafts = ptrs->n_drafts;
    __shared__ SelShared sh;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (tid == 0) sh.predecessor = *anchor_dev;
#pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        // prefetch the candidate's successor row (256 bf16 = 512 B, 32 x 16 B per warp)
        {
            int token = lane == 0 ? ids[step * kTopK + warp] : 0;
            token = __shfl_sync(0xffffffffu, token, 0);
            const bf16* src = succ + static_cast<int64_t>(token) * R + lane * 8;
            // one 16 B chunk per lane: 32 lanes x 8 elems = R (256). A loop over c8 with the
            // index lane*8 + c8*8 would write 32 x 256 elems per row (OOB into the next row,
            // 32-way race) - the row is exactly one chunk per lane.
            *reinterpret_cast<uint4*>(&sh.succ[warp][lane * 8]) = *reinterpret_cast<const uint4*>(src);
        }
        __syncthreads();
        if (tid < R)
            sh.product[tid] =
                __bfloat162float(pred[static_cast<int64_t>(sh.predecessor) * R + tid]) *
                __bfloat162float(hidden[step * R + tid]);
        __syncthreads();
        {
            float sum = 0.0f;
#pragma unroll 4
            for (int r = lane; r < R; r += 32) sum = fmaf(sh.product[r], __bfloat162float(sh.succ[warp][r]), sum);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if (lane == 0) sh.edge[warp] = unary[step * kTopK + warp] + sum;
        }
        __syncthreads();
        if (warp == 0) {
            const float e = lane < kTopK ? sh.edge[lane] : kNegInf;   // the dead lanes sort last
            float m = e;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            m = __shfl_sync(0xffffffffu, m, 0);
            const unsigned winners = __ballot_sync(0xffffffffu, lane < kTopK && e == m);
            const int sel = winners == 0 ? 0 : __ffs(winners) - 1;
            if (lane == 0) {
                sh.predecessor = ids[step * kTopK + sel];
                drafts[step] = sh.predecessor;
            }
        }
        __syncthreads();
    }
    if (tid == 0) *n_drafts = steps;
}

void selector_walk(const int* ids, const float* unary, const bf16* hidden, const int* anchor_dev,
                          const bf16* pred, const bf16* succ, int* drafts, int* n_drafts, int steps, int R,
                          cudaStream_t s, void* ptrs_dev) {
    selector_walk_kernel<<<1, 512, 0, s>>>((const SelPtrs*) ptrs_dev, steps, R);
}

/// The synchronous slot write (the default stream; called once before the graph capture).
void sel_write_slot(void* slot, const int* ids, const float* unary, const bf16* hidden, const int* anchor_dev,
                    const bf16* pred, const bf16* succ, int* drafts, int* n_drafts) {
    SelPtrs p{ids, unary, hidden, anchor_dev, pred, succ, drafts, n_drafts};
    if (cudaMemcpy(slot, &p, sizeof p, cudaMemcpyHostToDevice) != cudaSuccess)
        std::abort();
}

// ================================ the context store ================================
//
// The appended K/V into the cyclic slots: for i in [0, n): slot (pos0 + i) % window; K_l[slot, :],
// V_l[slot, :] (KV = the kv width, 512).

__global__ void __launch_bounds__(128) ctx_store_kernel(const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                        int64_t pos0, int window, int KV, bf16* __restrict__ kc,
                                                        bf16* __restrict__ vc) {
    const int i = static_cast<int>(blockIdx.x);
    const int slot = static_cast<int>((pos0 + i) % window);
    const bf16* ks = k + static_cast<int64_t>(i) * KV;
    const bf16* vs = v + static_cast<int64_t>(i) * KV;
    bf16* kd = kc + static_cast<int64_t>(slot) * KV;
    bf16* vd = vc + static_cast<int64_t>(slot) * KV;
    for (int t = threadIdx.x; t < KV / 8; ++t) {
        const int o = t * 8;
        *reinterpret_cast<uint4*>(kd + o) = *reinterpret_cast<const uint4*>(ks + o);
        *reinterpret_cast<uint4*>(vd + o) = *reinterpret_cast<const uint4*>(vs + o);
    }
}

void ctx_store(const bf16* k, const bf16* v, int64_t pos0, int n, int window, int KV, bf16* kc, bf16* vc,
                      cudaStream_t s) {
    ctx_store_kernel<<<n, 128, 0, s>>>(k, v, pos0, window, KV, kc, vc);
}

/// The frontier scalar: *frontier = value.
__global__ void frontier_kernel(int* frontier, int value) { *frontier = value; }

}  // namespace dflash2_detail
}  // namespace basalt::core
