// Standalone test: gemm_bf16 and gemm_bf16_wide against a CPU bf16 reference.
#include "src/core/dflash2/forward_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace basalt::core::dflash2_detail;

static float bf2f(uint16_t u) { return (float) ((uint32_t) u << 16); }
static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

// CPU reference: C [N, M] = W [M, K] @ x [N, K], accumulate in f32, round to bf16
static void ref_gemm(const uint16_t* w, const uint16_t* x, uint16_t* c, int M, int K, int N) {
    for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; ++m) {
            float s = 0;
            for (int k = 0; k < K; ++k) s += bf2f(w[(size_t) m * K + k]) * bf2f(x[(size_t) n * K + k]);
            c[(size_t) n * M + m] = f2bf(s);
        }
}

static int run(const char* name, bool wide, int M, int K, int N) {
    std::vector<uint16_t> w((size_t) M * K), x((size_t) N * K), c((size_t) N * M, 0), ref((size_t) N * M, 0);
    // deterministic pseudo-random in [-1, 1]
    uint32_t seed = 12345 + M * 7 + K * 13 + N;
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (float) ((seed >> 8) / 16777216.0 - 1.0);
    };
    for (auto& v : w) v = f2bf(rnd() * 0.5f);
    for (auto& v : x) v = f2bf(rnd() * 0.5f);
    ref_gemm(w.data(), x.data(), ref.data(), M, K, N);

    uint16_t *dw, *dx, *dc;
    cudaMalloc(&dw, w.size() * 2);
    cudaMalloc(&dx, x.size() * 2);
    cudaMalloc(&dc, c.size() * 2);
    cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(dc, 0, c.size() * 2);
    if (wide)
        gemm_bf16_wide((bf16*) dw, (bf16*) dx, (bf16*) dc, M, K, N, 0);
    else
        gemm_bf16((bf16*) dw, (bf16*) dx, (bf16*) dc, M, K, N, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), dc, c.size() * 2, cudaMemcpyDeviceToHost);

    int bad = 0;
    double maxrel = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        const float a = bf2f(c[i]), b = bf2f(ref[i]);
        const float d = a > b ? a - b : b - a;
        const float m = a > b ? a : b;
        const float rel = m > 0 ? d / m : (d == 0 ? 0 : 1);
        if (rel > maxrel) maxrel = rel;
        if (m > 0 && d / m > 8.0e-3f) ++bad;
    }
    std::printf("%s M=%d K=%d N=%d: %d/%zu bad, maxrel=%.5f\n", name, M, K, N, bad, c.size(), maxrel);
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dc);
    return bad;
}

int main() {
    int bad = 0;
    // the block GEMM shapes (N=8)
    bad += run("gemm_bf16", false, 640, 2560, 8);    // attn_conv_kernel: M=4G=640
    bad += run("gemm_bf16", false, 1024, 2560, 8);   // q: M=Q=1024
    bad += run("gemm_bf16", false, 512, 2560, 8);    // k/v: M=KV=512
    bad += run("gemm_bf16", false, 2560, 1024, 8);   // o_proj: M=H=2560, K=Q
    bad += run("gemm_bf16", false, 7680, 2560, 8);   // gate/up: M=I=7680
    bad += run("gemm_bf16", false, 2560, 7680, 8);   // down: M=H, K=I
    bad += run("gemm_bf16", false, 2560, 2560, 8);   // (square)
    bad += run("gemm_bf16", false, 256, 2560, 7);    // sel_hidden: M=R=256, N=K=7
    bad += run("gemm_bf16", false, 248320, 256, 7);  // head: M=V, K=R (big M)
    // the wide GEMM shapes (context append)
    bad += run("gemm_bf16_wide", true, 2560, 12800, 64);   // fc: M=H, K=ntap*h
    bad += run("gemm_bf16_wide", true, 512, 2560, 64);     // k/v ctx
    bad += run("gemm_bf16_wide", true, 2560, 2560, 1);     // N=1
    bad += run("gemm_bf16_wide", true, 2560, 2560, 512);   // N=512
    bad += run("gemm_bf16_wide", true, 7680, 2560, 333);   // odd N
    std::printf("gemm test: %s\n", bad == 0 ? "PASS" : "FAIL");
    return bad ? 1 : 0;
}
