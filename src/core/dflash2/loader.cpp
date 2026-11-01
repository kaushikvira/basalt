// src/core/dflash2/loader.cpp - the DFlash2 drafter's C++ loader: the `dflash2/dense` pool, the
// `dflash2/dense.index`, and the geometry from `meta["dflash2"]`.  See include/basalt/core/dflash2.hpp.
//
// This is the scaffolding of the port: `load` (host-side: the index, the internal layout, the geometry),
// `plan_bytes` (the VRAM: the dense pool + the local cyclic K/V + the block's working buffers), and `bind`
// (device-side: the pool, the K/V arena, the buffers).  The block-diffusion `draft()` is a STUB (the forward
// + kernels are the next sub-task).
#include "basalt/core/dflash2.hpp"

#include "basalt/core/direct_upload.hpp"
#include "basalt/core/session.hpp"

#include <cstring>
#include <sstream>

namespace basalt::core {

// ================================ the geometry ================================

namespace {
/// A positive-integer field of a JSON object; false with `err` when it is missing or not a number.
bool meta_int(const Json* o, const char* key, int& out, std::string& err) {
    const Json* v = o->get(key);
    if (v == nullptr || !v->is_number()) { err = std::string("dflash2: meta is missing '") + key + "'"; return false; }
    out = (int) v->i64();
    return true;
}
}  // namespace

bool DFlash2Geometry::from_meta(const Json& meta, DFlash2Geometry& out, std::string& err) {
    const Json* d = meta.get("dflash2");
    if (d == nullptr || !d->is_object()) { err = "dflash2: meta has no 'dflash2' object"; return false; }
    const Json* cfg = d->get("dflash_config");
    if (cfg == nullptr || !cfg->is_object()) { err = "dflash2: meta has no 'dflash_config' object"; return false; }
    out = DFlash2Geometry{};
    if (!meta_int(cfg, "block_size", out.block_size, err) ||
        !meta_int(cfg, "conv_group_size", out.conv_group_size, err) ||
        !meta_int(cfg, "conv_kernel_size", out.conv_kernel_size, err) ||
        !meta_int(cfg, "selector_rank", out.selector_rank, err) ||
        !meta_int(cfg, "selector_top_k", out.selector_top_k, err) ||
        !meta_int(cfg, "mask_token_id", out.mask_token_id, err) ||
        !meta_int(d, "n_layers", out.n_layers, err) ||
        !meta_int(d, "hidden_size", out.hidden_size, err) ||
        !meta_int(d, "head_dim", out.head_dim, err) ||
        !meta_int(d, "num_attention_heads", out.num_attention_heads, err) ||
        !meta_int(d, "num_key_value_heads", out.num_key_value_heads, err) ||
        !meta_int(d, "intermediate_size", out.intermediate_size, err) ||
        !meta_int(d, "vocab_size", out.vocab_size, err) ||
        !meta_int(d, "max_position_embeddings", out.max_position_embeddings, err)) {
        return false;
    }
    const Json* taps = cfg->get("target_layer_ids");
    if (taps == nullptr || !taps->is_array() || taps->items.empty() || taps->count != taps->items.size()) {
        err = "dflash2: 'target_layer_ids' is missing or truncated";
        return false;
    }
    for (const Json& x : taps->items) {
        if (!x.is_number()) { err = "dflash2: 'target_layer_ids' has a non-integer"; return false; }
        out.target_layer_ids.push_back((int) x.i64());
    }
    const Json* lt = d->get("layer_types");
    if (lt != nullptr && lt->is_array()) {
        for (const Json& x : lt->items) {
            if (!x.is_string()) { err = "dflash2: 'layer_types' has a non-string"; return false; }
            out.layer_types.push_back(x.s);
        }
    }
    if (const Json* v = d->get("rms_norm_eps"))
        if (v->is_number()) out.rms_norm_eps = (float) v->num();
    if (const Json* v = d->get("rope_theta"))
        if (v->is_number()) out.rope_theta = (float) v->num();
    return out.valid(err);
}

bool DFlash2Geometry::valid(std::string& err) const {
    if (block_size <= 0 || conv_group_size <= 0 || conv_kernel_size <= 0 || selector_rank <= 0 ||
        selector_top_k <= 0 || n_layers <= 0 || hidden_size <= 0 || head_dim <= 0 || num_attention_heads <= 0 ||
        num_key_value_heads <= 0 || intermediate_size <= 0 || vocab_size <= 0 || max_position_embeddings <= 0 ||
        sliding_window <= 0) {
        err = "dflash2: a geometry field is not positive";
        return false;
    }
    if (num_attention_heads % num_key_value_heads) { err = "dflash2: the heads are not a multiple of the kv heads"; return false; }
    if (hidden_size % conv_group_size) { err = "dflash2: the hidden width is not a multiple of the conv group"; return false; }
    if (target_layer_ids.empty()) { err = "dflash2: no target_layer_ids"; return false; }
    for (int t : target_layer_ids)
        if (t < 0) { err = "dflash2: a negative target_layer_id"; return false; }
    if (mask_token_id < 0 || mask_token_id >= vocab_size) { err = "dflash2: mask_token_id is outside the vocabulary"; return false; }
    if (!layer_types.empty() && (int) layer_types.size() != n_layers) { err = "dflash2: layer_types does not span the layers"; return false; }
    return true;
}

// ================================ the loader ================================

bool DFlash2Drafter::load(const DFlash2Files& files, const Json& meta, std::string& err) {
    if (!files.present()) { err = "dflash2: the dflash2/dense (or dflash2/dense.index) section is missing"; return false; }
    DFlash2Geometry g;
    if (!DFlash2Geometry::from_meta(meta, g, err)) return false;
    g_ = g;
    files_ = files;
    tensors_.clear();
    std::vector<uint8_t> idx_text;
    if (!read_range(files_.index, idx_text, err)) { err = "dflash2: cannot read " + files_.index.path; return false; }
    std::istringstream idx(std::string(idx_text.begin(), idx_text.end()));
    std::string line;
    while (std::getline(idx, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        DFlash2Tensor t;
        is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
        if (!is) { err = "dflash2: malformed dense index line: " + line; return false; }
        tensors_.push_back(std::move(t));
    }
    if (tensors_.empty()) { err = "dflash2: the dense index is empty"; return false; }
    // the internal layout, resolved by role
    w_ = DFlash2Weights{};
    w_.layers.resize((size_t) g_.n_layers);
    for (const DFlash2Tensor& t : tensors_) resolve(t);
    // the shapes the kernels depend on, checked against the geometry (a mismatch is a load-time stop)
    const int64_t h = g_.hidden_size, d = g_.head_dim, q = g_.q_width(), kv = g_.kv_width();
    const int64_t inter = g_.intermediate_size, V = g_.vocab_size, rank = g_.selector_rank;
    const int64_t taps = (int64_t) g_.target_layer_ids.size();
    const int64_t conv_rows = 2 * (int64_t) g_.conv_kernel_size * (h / g_.conv_group_size);
    const int64_t conv_cols = (int64_t) g_.conv_kernel_size * h;
    auto shape = [&](const DFlash2Tensor& t, const char* role, int64_t rows, int64_t cols) -> bool {
        if (t.name.empty()) { err = std::string("dflash2: ") + role + " is missing from the index"; return false; }
        if (t.rows != rows || t.cols != cols) {
            err = std::string("dflash2: ") + role + " is " + std::to_string(t.rows) + "x" + std::to_string(t.cols) +
                  ", the geometry wants " + std::to_string(rows) + "x" + std::to_string(cols);
            return false;
        }
        return true;
    };
    if (!shape(w_.feature_projection, "feature_projection", h, taps * h) ||
        !shape(w_.context_norm, "context_norm", 1, h) ||
        !shape(w_.final_norm, "final_norm", 1, h) ||
        !shape(w_.sel_hidden, "candidate_selector.hidden_projection", rank, h) ||
        !shape(w_.sel_pred, "candidate_selector.predecessor_codebook", V, rank) ||
        !shape(w_.sel_succ, "candidate_selector.successor_codebook", V, rank)) {
        return false;
    }
    for (int i = 0; i < g_.n_layers; ++i) {
        const DFlash2LayerW& L = w_.layers[(size_t) i];
        const char* pf = "layer ";
        (void) pf;
        if (!shape(L.q, "q_proj", q, h) || !shape(L.k, "k_proj", kv, h) || !shape(L.v, "v_proj", kv, h) ||
            !shape(L.o_proj, "o_proj", h, q) || !shape(L.q_norm, "q_norm", 1, d) || !shape(L.k_norm, "k_norm", 1, d) ||
            !shape(L.input_norm, "input_norm", 1, h) || !shape(L.post_attention_norm, "post_attention_norm", 1, h) ||
            !shape(L.gate, "gate_proj", inter, h) || !shape(L.up, "up_proj", inter, h) ||
            !shape(L.down, "down_proj", h, inter) ||
            !shape(L.attn_conv_base, "attention_conv.base_kernel", 2, conv_cols) ||
            !shape(L.attn_conv_kernel, "attention_conv.kernel_projection", conv_rows, h) ||
            !shape(L.mlp_conv_base, "mlp_conv.base_kernel", 2, conv_cols) ||
            !shape(L.mlp_conv_kernel, "mlp_conv.kernel_projection", conv_rows, h)) {
            return false;
        }
    }
    return true;
}

void DFlash2Drafter::resolve(const DFlash2Tensor& t) {
    const std::string& n = t.name;
    auto set = [&](DFlash2Tensor& slot) { slot = t; };
    if (n == "fc.weight") { set(w_.feature_projection); return; }
    if (n == "hidden_norm.weight") { set(w_.context_norm); return; }
    if (n == "norm.weight") { set(w_.final_norm); return; }
    if (n == "candidate_selector.hidden_projection.weight") { set(w_.sel_hidden); return; }
    if (n == "candidate_selector.predecessor_codebook") { set(w_.sel_pred); return; }
    if (n == "candidate_selector.successor_codebook") { set(w_.sel_succ); return; }
    // per-layer: layers.{i}.{...}
    static const std::string pref = "layers.";
    if (n.rfind(pref, 0) != 0) return;
    const size_t dot = n.find('.', pref.size());
    if (dot == std::string::npos) return;
    int i = 0;
    for (size_t j = pref.size(); j < dot; ++j) {
        const char c = n[j];
        if (c < '0' || c > '9') return;
        i = i * 10 + (c - '0');
    }
    if (i < 0 || i >= (int) w_.layers.size()) return;
    DFlash2LayerW& L = w_.layers[(size_t) i];
    const std::string rest = n.substr(dot + 1);
    if (rest == "self_attn.q_proj.weight") set(L.q);
    else if (rest == "self_attn.k_proj.weight") set(L.k);
    else if (rest == "self_attn.v_proj.weight") set(L.v);
    else if (rest == "self_attn.o_proj.weight") set(L.o_proj);
    else if (rest == "self_attn.q_norm.weight") set(L.q_norm);
    else if (rest == "self_attn.k_norm.weight") set(L.k_norm);
    else if (rest == "input_layernorm.weight") set(L.input_norm);
    else if (rest == "post_attention_layernorm.weight") set(L.post_attention_norm);
    else if (rest == "mlp.gate_proj.weight") set(L.gate);
    else if (rest == "mlp.up_proj.weight") set(L.up);
    else if (rest == "mlp.down_proj.weight") set(L.down);
    else if (rest == "attention_conv.base_kernel") set(L.attn_conv_base);
    else if (rest == "attention_conv.kernel_projection.weight") set(L.attn_conv_kernel);
    else if (rest == "mlp_conv.base_kernel") set(L.mlp_conv_base);
    else if (rest == "mlp_conv.kernel_projection.weight") set(L.mlp_conv_kernel);
}

// ================================ the VRAM ================================

uint64_t DFlash2Drafter::activations_bytes(const DFlash2Geometry& g) {
    const uint64_t B = (uint64_t) g.block_size;
    const uint64_t h = (uint64_t) g.hidden_size;
    const uint64_t q = (uint64_t) g.q_width();
    const uint64_t kv = (uint64_t) g.kv_width();
    const uint64_t inter = (uint64_t) g.intermediate_size;
    const uint64_t conv = (uint64_t) (h / (uint64_t) g.conv_group_size) * 2u * (uint64_t) g.conv_kernel_size;
    // per layer (bf16): the input, q, k, v, the attention out, the MLP in / intermediate / out, the two conv outputs
    const uint64_t per_layer = B * (h + q + 2u * kv + h + inter + h + 2u * conv);
    const uint64_t layers = per_layer * (uint64_t) g.n_layers;
    // the selector: the block's projected rows (rank, bf16) + the top-k candidates' scores (fp32)
    const uint64_t selector = B * ((uint64_t) g.selector_rank * 2u + (uint64_t) g.selector_top_k * 4u);
    // fp32 working: the block's normalized rows + the attention scores
    const uint64_t fp32 = B * (h * 4u + q * 4u);
    // a 4 MiB headroom for the forward's scratch (the exact buffers are the next sub-task), 256-aligned
    return (layers + selector + fp32 + (4u << 20) + 255) & ~255ull;
}

bool DFlash2Drafter::plan_bytes(const DFlash2Files& files, const DFlash2Geometry& g, uint64_t& out, std::string& err) {
    if (!g.valid(err)) return false;
    if (files.dense.empty() || !files.dense.exists()) { err = "dflash2: cannot read " + files.dense.path; return false; }
    const uint64_t dense = files.dense.size();
    const uint64_t kv = g.kv_bytes(g.sliding_window);
    const uint64_t act = activations_bytes(g);
    out = dense + kv + act;
    return true;
}

// ================================ the bind ================================

bool DFlash2Drafter::bind(SessionState& ss, std::string& err) {
    (void) ss;   // the forward sub-task backs the K/V from the session's KvArena; the stub owns its own
    if (tensors_.empty()) { err = "dflash2: bind before load"; return false; }
    cudaGetDevice(&device_);
    kv_cells_ = g_.sliding_window;
    // ---- the dense pool
    const uint64_t dense = files_.dense.size();
    if (cudaMalloc((void**) &dense_, dense) != cudaSuccess) {
        err = "dflash2: the dense pool does not fit in VRAM (" + std::to_string(dense >> 20) + " MiB)";
        return false;
    }
    if (!DirectUploader().upload(files_.dense.path, files_.dense.offset, dense, dense_, err)) {
        err = "dflash2: cannot read the dense pool: " + err;
        cudaFree(dense_);
        dense_ = nullptr;
        return false;
    }
    dense_bytes_ = dense;
    // the internal layout's device pointers (the pool's bytes, in place)
    auto ptr = [&](DFlash2Tensor& t) {
        if (t.name.empty()) return;
        if (t.off + t.bytes > dense_bytes_) {
            err = "dflash2: the index names bytes past the dense pool: " + t.name;
        } else {
            t.dev = (const void*) (dense_ + t.off);
        }
    };
    ptr(w_.feature_projection);
    ptr(w_.context_norm);
    ptr(w_.final_norm);
    ptr(w_.sel_hidden);
    ptr(w_.sel_pred);
    ptr(w_.sel_succ);
    for (auto& L : w_.layers) {
        ptr(L.q); ptr(L.k); ptr(L.v); ptr(L.o_proj);
        ptr(L.q_norm); ptr(L.k_norm); ptr(L.input_norm); ptr(L.post_attention_norm);
        ptr(L.gate); ptr(L.up); ptr(L.down);
        ptr(L.attn_conv_base); ptr(L.attn_conv_kernel); ptr(L.mlp_conv_base); ptr(L.mlp_conv_kernel);
    }
    if (!err.empty()) { cudaFree(dense_); dense_ = nullptr; return false; }
    // ---- the local cyclic K/V (one sliding window per layer, K and V)
    const uint64_t kv_bytes = g_.kv_bytes(kv_cells_);
    if (cudaMalloc((void**) &kv_arena_, kv_bytes) != cudaSuccess) { err = "dflash2: the K/V arena does not fit in VRAM"; return false; }
    cudaMemset(kv_arena_, 0, kv_bytes);
    kv_k_.resize((size_t) g_.n_layers);
    kv_v_.resize((size_t) g_.n_layers);
    const uint64_t layer_elems = (uint64_t) kv_cells_ * (uint64_t) g_.num_key_value_heads * (uint64_t) g_.head_dim;   // uint16's per K (or V)
    for (int i = 0; i < g_.n_layers; ++i) {
        kv_k_[(size_t) i] = kv_arena_ + (size_t) i * 2 * layer_elems;
        kv_v_[(size_t) i] = kv_k_[(size_t) i] + (size_t) layer_elems;
    }
    // ---- the block's working buffers
    arena_bytes_ = activations_bytes(g_);
    if (cudaMalloc(&arena_, arena_bytes_) != cudaSuccess) { err = "dflash2: the working buffers do not fit in VRAM"; return false; }
    cudaMemset(arena_, 0, arena_bytes_);
    // ---- the stream and the done event
    if (cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, 0) != cudaSuccess) { err = "dflash2: the stream"; return false; }
    if (cudaEventCreateWithFlags(&ev_done_, cudaEventDisableTiming) != cudaSuccess) { err = "dflash2: the done event"; return false; }
    vram_ = dense_bytes_ + kv_bytes + arena_bytes_;
    return true;
}

// ================================ the stubs ================================

bool DFlash2Drafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                           float* probs, int* n_drafts) {
    (void) T;
    (void) tokens;
    (void) p;
    (void) a;
    (void) drafts;
    (void) probs;
    (void) err;
    if (n_drafts) *n_drafts = 0;
    // STUB: the block-diffusion forward (the 5 layers + the selector) is the next sub-task.
    return true;
}

bool DFlash2Drafter::prewarm(std::string& err) {
    (void) err;
    // STUB: the draft graphs come with the forward (the next sub-task).
    return true;
}

bool DFlash2Drafter::idle(std::string& err) {
    if (cs_ && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "dflash2: its stream failed"; return false; }
    return true;
}

DFlash2Drafter::~DFlash2Drafter() {
    if (cs_) cudaStreamSynchronize(cs_);
    if (ev_done_) cudaEventDestroy(ev_done_);
    if (cs_) cudaStreamDestroy(cs_);
    if (dense_) cudaFree(dense_);
    if (kv_arena_) cudaFree(kv_arena_);
    if (arena_) cudaFree(arena_);
}

}  // namespace basalt::core
