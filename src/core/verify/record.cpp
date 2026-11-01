#include "common.hpp"
#include "record.hpp"
#include "expert_plan.hpp"
#include "basalt/core/verify.hpp"
#include "basalt/core/native_head.hpp"
#include "basalt/kernels/elementwise.hpp"
#include "basalt/kernels/cvec.hpp"
#include "basalt/kernels/verify_kernels.hpp"
#include <string>

namespace basalt::core {
using namespace verify_detail;

// The decode window's overlap structure (all always on): in windows of 2+ tokens the shared expert's side kernels run
// at the greatest stream priority, so their blocks are scheduled ahead of the routed experts' queued blocks; in windows
// of 1-2 tokens a GDN layer's z projection is forked at that priority beside the qkv projection; the MoE route is
// split (CUDA0's plan and q8_1 rows on the main stream, the ring for the peer card and the CPU pool on a stream of its
// own) with the hc read's router items writing the FFN input's q8_1 rows and the gate/up launch building CUDA0's plan
// itself; the combine runs as the wait kernel's programmatic dependent.  A QSA layer's K, V and the indexer key/query
// run on branches of their own beside the query projection; its block scores skip a window whose selections are all
// the identity; the head's hyper-connection read does every column in one launch; layer 1's PLE takes two launches;
// the attention's merge also writes the output projection's q8_1 rows.  The attention's hc read writes the
// projections' q8_1 rows, and a GDN layer's qkv and z projections are one dense_tc_pair launch.

// ================================ THE WINDOW, AS CAPTURED ================================
//
// The stream is ordered pre(0) | post(0) pre(1) | post(1) pre(2) | ...: each layer's mixer and router, then its
// experts and combine, with the next layer's mixer launched behind them.
bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {
    WindowRecorder recorder(*this, T, cs, err);
    return recorder.run();
}

bool Verifier::WindowRecorder::run() {
    using namespace basalt::kernels;
    if (!inputs()) return false;
    fuse_head_gr = (v_.le_ == g.n_layers) && (v_.head_ != nullptr && v_.head_->loaded()) &&
                              !cvec().covers(g.n_layers - 1);
    if (!pre(v_.lb_)) return false;
    for (int64_t l = v_.lb_; l < v_.le_; ++l) {
        if (!post(l)) return false;
        // the DFlash2 drafter's hidden-state tap: after the layer's FFN write, the residual is the layer's output;
        // record its hyper-connection mix (the target's hidden state) for the window's rows, when this is a tapped
        // layer.  A pure read; the window's arithmetic is untouched.
        if (v_.tap_layers_ != nullptr) {
            for (int i = 0; i < v_.tap_n_taps_; ++i)
                if (v_.tap_layers_[(size_t) i] == l) { tap(l, i); break; }
        }
        if (l + 1 < v_.le_ && !pre(l + 1)) return false;
    }
    if (lz_open) cudaStreamWaitEvent(cs, v_.lz_join_, 0);
    if (v_.le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
        float* hout = v_.hand_out_ + (size_t) hrow0 * HB;
        copy_from_mapped(hout, v_.R_, (int64_t) T * HC * N, cs);
        copy_from_mapped(hout + (size_t) T * HC * N, v_.bo_, (int64_t) T * N, cs);
        copy_from_mapped(hout + (size_t) T * (HC + 1) * N, v_.inj2_, (int64_t) T * HC, cs);
        if (!v_.batch_rec_ && v_.next_ != nullptr && v_.next_->m_flag_ != nullptr) signal_flag(v_.next_->m_flag_ + 12, 1, cs);
        return true;
    }
    return head();
}

// The DFlash2 drafter's hidden-state tap: after layer `l`'s FFN write, the residual is the layer's output.  Record its
// hyper-connection mix (n_embd wide, the target's hidden state) for the window's T rows into the tap's features buffer
// at the tapped-layer slot `i`.  A pure read of the residual (gr_read over the output_hc_* weights); the window's
// arithmetic is untouched.  The mix is written straight into the strided features buffer ([T, n_taps, n_embd],
// token-major), so no copy is needed.
void Verifier::WindowRecorder::tap(int64_t l, int i) {
    (void) l;
    if (v_.tap_layers_ == nullptr || v_.tap_n_taps_ == 0 || v_.tap_features_ == nullptr) return;
    const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                    *hu = wt.find("output_hc_up.weight");
    if (!hn || !hd || !hu) return;
    BlockBuffers bb = ss.block;
    for (int t = 0; t < T; ++t) {
        bb.R = Rt(t);
        bb.mixed = v_.tap_features_ + (size_t) t * (size_t) v_.tap_n_taps_ * N + (size_t) i * N;
        bb.inject = v_.tap_inj_;
        basalt::kernels::gr_read(bb.R, (const float*) hn->data, (const uint16_t*) hd->data,
                                 (const uint16_t*) hu->data, nullptr, EPS, gs, bb.gr,
                                 bb.mixed, bb.inject, cs);
    }
}

}  // namespace basalt::core
