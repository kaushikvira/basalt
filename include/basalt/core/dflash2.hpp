// include/basalt/core/dflash2.hpp - the DFlash2 block-diffusion drafter on the GPU.
//
// A dense 5-layer drafter (Qwen3.8-Flash-Next, 0.6B) ported from NInfer.  It reads the target model's
// hidden taps (`feature_projection` over `target_layer_ids`), runs a block of `block_size` tokens through
// its own 5 layers (sliding-window attention, per-layer dynamic convs, a dense MLP), and a candidate
// selector picks the block's tokens.  It keeps its own local cyclic K/V (one sliding window per layer).
//
// This is the SCAFFOLDING of the port: the interface, the C++ loader (the `dflash2/dense` pool + the
// `dflash2/dense.index` + the geometry from `meta["dflash2"]`), the VRAM planning and the bind.  The
// block-diffusion `draft()` forward is a STUB (it returns an empty block); the forward + kernels are the
// next sub-task.  The structure mirrors `MtpDrafter` (mtp.hpp): the same contract (`load` / `plan_bytes` /
// `bind` / `draft` / `done_event` / `prewarm`), the same lifecycle (host-side `load`, device-side `bind`),
// the same VRAM accounting (`plan_bytes` before `bind`).
#pragma once

#include "basalt/core/model_file.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

struct SessionState;
class WeightTable;
class NativeHead;

/// The DFlash2 drafter's geometry, read from `meta["dflash2"]` (the converter's `provenance()`): the
/// `dflash_config` block (block_size, conv_*, selector_*, target_layer_ids, mask_token_id) plus the head /
/// MLP / vocab fields.  Every number a kernel depends on; a change is a kernel contract, not a knob.
struct DFlash2Geometry {
    // dflash_config
    int block_size = 0;            ///< tokens per diffusion block (8)
    int conv_group_size = 0;       ///< hidden channels per dynamic-conv group (16)
    int conv_kernel_size = 0;      ///< taps of the dynamic conv (2)
    int selector_rank = 0;         ///< the candidate selector's projection rank (256)
    int selector_top_k = 0;        ///< candidates the selector keeps (16)
    std::vector<int> target_layer_ids;   ///< the target layers tapped for the features ([5,15,25,35,45])
    int mask_token_id = 0;         ///< the diffusion mask token (248077)
    // the drafter's own model
    int n_layers = 0;              ///< the drafter's layers (5)
    int hidden_size = 0;           ///< 2560
    int head_dim = 0;              ///< 128
    int num_attention_heads = 0;   ///< 32
    int num_key_value_heads = 0;   ///< 8
    int intermediate_size = 0;     ///< 7680
    int vocab_size = 0;            ///< 248320
    int max_position_embeddings = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1e7f;
    std::vector<std::string> layer_types;   ///< per layer: "sliding_attention" / "full_attention"
    /// The local cyclic K/V cap (the sliding window, 2048).  Not in the converter's provenance; the caller
    /// sets it (the default is the Flash Next checkpoint's window).
    int sliding_window = 2048;

    int q_width() const { return num_attention_heads * head_dim; }
    int kv_width() const { return num_key_value_heads * head_dim; }
    /// Bytes of one layer's cyclic K/V (`cells` cells, K and V, fp16).
    uint64_t kv_layer_bytes(int64_t cells) const {
        return (uint64_t) cells * (uint64_t) num_key_value_heads * (uint64_t) head_dim * 2u * 2u;
    }
    /// Bytes of the whole drafter's cyclic K/V (every layer, `cells` cells each).
    uint64_t kv_bytes(int64_t cells) const { return kv_layer_bytes(cells) * (uint64_t) n_layers; }

    /// Reads the geometry from `meta["dflash2"]` (the converter's `provenance()`); false with `err` when a
    /// field is missing or implausible.
    static bool from_meta(const Json& meta, DFlash2Geometry& out, std::string& err);
    /// A self-consistent geometry (the widths divide, the taps are in range, the window is positive).
    bool valid(std::string& err) const;
};

/// The drafter's files: the `dflash2/...` sections of a .basalt.
struct DFlash2Files {
    FileRange dense, index;   ///< dflash2/dense, dflash2/dense.index
    static DFlash2Files from(const ModelSources& s) {
        DFlash2Files f;
        f.dense = s.dflash2_dense;
        f.index = s.dflash2_dense_index;
        return f;
    }
    bool present() const { return !dense.empty() && !index.empty(); }
};

/// One tensor of the `dflash2/dense` pool, as the index names it (one line: `name kind rows cols offset
/// bytes`).  `kind` is "bf16" (the projections) or "f32" (the norms).  `dev` is set by `bind` (the tensor's
/// bytes in the device pool).
struct DFlash2Tensor {
    std::string name;   ///< the index (safetensors) name
    std::string kind;   ///< "bf16" / "f32"
    int64_t rows = 0, cols = 0;
    uint64_t off = 0, bytes = 0;
    const void* dev = nullptr;
};

/// The drafter's internal layout: the pool's tensors resolved by ROLE (the NInfer mapping).  The device
/// pointers are set by `bind`; a role the index lacks is left null (and `load` reports it).  `context_key` /
/// `context_value` are aliases of `k` / `v` (the same bytes).
struct DFlash2LayerW {
    DFlash2Tensor q, k, v, o_proj;
    DFlash2Tensor q_norm, k_norm;
    DFlash2Tensor input_norm, post_attention_norm;
    DFlash2Tensor gate, up, down;
    DFlash2Tensor attn_conv_base, attn_conv_kernel;   ///< base_kernel (2, taps, h), kernel_projection (rows, h)
    DFlash2Tensor mlp_conv_base, mlp_conv_kernel;
};

struct DFlash2Weights {
    DFlash2Tensor feature_projection;   ///< (h, n_taps * h)
    DFlash2Tensor context_norm;         ///< (h,)
    DFlash2Tensor final_norm;           ///< (h,)
    std::vector<DFlash2LayerW> layers;  ///< n_layers
    DFlash2Tensor sel_hidden;           ///< (rank, h)
    DFlash2Tensor sel_pred, sel_succ;   ///< (vocab, rank) each
};

class DFlash2Drafter {
public:
    DFlash2Drafter() = default;
    ~DFlash2Drafter();
    DFlash2Drafter(const DFlash2Drafter&) = delete;
    DFlash2Drafter& operator=(const DFlash2Drafter&) = delete;

    /// Host-side: read `dflash2/dense.index`, resolve the pool's tensors to the internal layout, and read the
    /// geometry from `meta["dflash2"]`.  Validates the shapes against the geometry.  No device allocation.
    bool load(const DFlash2Files& files, const Json& meta, std::string& err);

    /// The bytes `bind` places on the device: the dense pool (the section size), the local cyclic K/V (one
    /// sliding window per layer) and the block's working buffers, from the section size and the geometry.
    static bool plan_bytes(const DFlash2Files& files, const DFlash2Geometry& g, uint64_t& out, std::string& err);

    /// Device-side: allocate the dense pool and the K/V arena, copy the pool in, zero the K/V, and set the
    /// internal layout's device pointers.  Takes the target's embedding + head (the drafter reuses them, per
    /// NInfer's bind_draft).  After `load`.  (The allocation happens here, not in `load` — a reordered
    /// lifecycle vs MTP.)
    bool bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err);

    /// One draft round: the block-diffusion forward.  STUB - returns an empty block (the forward + kernels are
    /// the next sub-task).  Mirrors MtpDrafter::draft's signature so the decode-loop integration is a drop-in.
    bool draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
               float* probs = nullptr, int* n_drafts = nullptr);

    /// Captures every draft graph now rather than at first use.  STUB - a no-op (the graphs come with the
    /// forward).  After `bind`.
    bool prewarm(std::string& err);

    /// An event behind everything queued on the drafter's stream so far (a verify window on this card waits for
    /// it).  Null before `bind`.
    cudaEvent_t done_event() {
        if (cs_ && ev_done_) cudaEventRecord(ev_done_, cs_);
        return ev_done_;
    }

    uint64_t vram_bytes() const { return vram_; }
    int device() const { return device_; }
    const DFlash2Geometry& geometry() const { return g_; }
    const DFlash2Weights& weights() const { return w_; }
    /// The drafter's stream is idle (a window may use the card).  True before `bind`.
    bool idle(std::string& err);

private:
    /// The index line's tensor, copied into the role of the internal layout it names (an unrecognized name is
    /// ignored; `load` then reports any role the index left empty).
    void resolve(const DFlash2Tensor& t);
    /// The working-buffer bytes for one block of `g.block_size` tokens through the layers + selector.
    static uint64_t activations_bytes(const DFlash2Geometry& g);

    DFlash2Geometry g_;
    DFlash2Files files_;
    DFlash2Weights w_;
    std::vector<DFlash2Tensor> tensors_;   ///< the index, in pool order
    int resolved_count_ = 0;               ///< S3: the tensors resolve() claimed (== tensors_.size() after load)
    const WeightTable* wt_ = nullptr;      ///< S2: the target's weight table (embedding + head)
    const NativeHead* head_ = nullptr;     ///< S2: the target's output head
    uint64_t vram_ = 0;
    int device_ = -1;
    cudaStream_t cs_ = nullptr;
    cudaEvent_t ev_done_ = nullptr;
    // device
    uint8_t* dense_ = nullptr;             ///< the dflash2/dense pool
    uint64_t dense_bytes_ = 0;
    uint16_t* kv_arena_ = nullptr;         ///< the n_layers cyclic K/V (K and V per layer)
    int64_t kv_cells_ = 0;                 ///< the window cap (g.sliding_window)
    std::vector<uint16_t*> kv_k_;          ///< [layer] K pool
    std::vector<uint16_t*> kv_v_;          ///< [layer] V pool
    void* arena_ = nullptr;                ///< the block's working buffers
    uint64_t arena_bytes_ = 0;
};

}  // namespace basalt::core
