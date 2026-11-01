# DFlash2 Retrain — Log

## 2026-11-01: Batched training loop + 1-epoch train + go/no-go (LOSE)

**Commit:** (this entry)

### Optimization: batched the forward (option A)
The reference model's `DFlash2DraftModel.forward()` is already batch-compatible —
fc+norm, the 5 decoder layers (SDPA with a `[1,1,q,k]` causal mask, dynamic conv,
MLP) and RoPE all broadcast over the batch dim. So `DFlash2Trainer.forward()` now
takes `[B, W, 5, 2560]` context + `[B]` anchors and runs one forward per batch;
the per-sample Python loop is gone. Also: `--max-steps` for quick tests,
logging every 5 steps (loss, ms/step, ETA, GPU GB), no per-step `.item()` syncs.

### Measured speedup (B=32, W=2048, GPU 1)
- Per-sample loop (baseline): **~540–560 ms/step** steady state
- Batched: **~340–370 ms/step** steady state → **1.6×**, not the expected 3–5×.
  The GPU was already busy per sample (each sample's forward is ~10–15 ms of
  kernel time), so the launch-overhead premise overestimated the win.
- Profiler: one step is 331 ms of pure CUDA kernel time (fc GEMM dominant), so
  the loop is now at the GPU-bound floor. Further speedup = less compute
  (W=1024 → ~2×), not better scheduling.

### 1-epoch train (real taps, 99,488 tokens, 3,045 steps)
- 18.0 min, avg loss 14.66 → 11.25 (final steps ~8.5–12, still falling)
- GPU memory 11.0 GB (taps 2.55 GB + model 1.2 GB + head 2.4 GB + states/acts)
- Checkpoint: `~/models/basalt-swift/dflash2-fn-finetuned/` — 81 tensors, all
  BF16, same shapes/keys as the original, 1.2 GB. fc.weight mean |Δ| = 2.2e-5.

### Pack + acceptance (go/no-go)
- Pack: `replace_dflash2.py` (the base already carries a dflash2 section, so
  `add_dflash2.py` refuses) → `Swift-1.5-FN-NVFP4-dflash2-ft.basalt` (123.1 GiB).
  Written to `~/work/models/basalt-swift/` — `/home/kv/models` had only 52 GB free.
- Bench (GPU 1, spec 8, greedy, seed 42, conv/struct/prose):

| drafter | conv t/s (acc %) | struct t/s (acc %) | prose t/s (acc %) |
|---|---|---|---|
| **MTP (baseline)** | 229 (69.6) | 367 (97.4) | 208 (66.1) |
| DFlash2 original | 43 (0.7) | 35 (1.1) | 41 (0.3) |
| DFlash2 fine-tuned | 35 (0.4) | 34 (0.6) | 42 (0.5) |

**Gate: acceptance ≥ 70% = WIN. Result: LOSE — MTP stays.**

### Key finding: the DFlash2 *port* is broken, not the fine-tune
The **original, unmodified** z-lab drafter also scores 0.3–1.1% acceptance while
MTP scores 66–97% on the same fixtures. The fine-tuned drafter (0.4–0.6%) is
statistically indistinguishable from the original — training worked (loss
14.7 → 11.2 on 99.5K tokens of real taps), the pack is structurally correct
(same sections, updated sha, 81/81 BF16 tensors), so the ~0.5% acceptance is a
bug in the engine's DFlash2 path (tap extraction / fc+norm / RoPE positions /
anchor handling — one of these), amplified by the BF16-train vs NVFP4-serve gap.
Fixing the port is the prerequisite for any DFlash2 win; more training data on
top of a broken port is wasted compute.

### Next steps
1. Debug the engine's DFlash2 decode path against the PyTorch reference (the
   `/tmp/dflash2-fixture/` ref tensors from the earlier session exist for this).
2. Only after the port accepts ≥ ~50% with the *original* drafter, re-run the
   fine-tune (more data/epochs are then worth it — loss was still falling).

## 2026-11-01: Phase 4 steps 4-5 (PyTorch forward + training loop)

**Commit:** (this entry)

### What was done
- Adapted the open-source DFlash2 PyTorch model (`z-lab/dflash`) for the Flash Next checkpoint.
- Created `dflash2_fn_config.py` — loads the config from `config.json`, asserts invariants.
- Created `retrain_dflash2.py` — the full training script:
  - Loads the 81-tensor DFlash2 drafter (603M params, 1.2 GB BF16) from safetensors.
  - Loads the frozen target embed+head (2.4 GB BF16) from the basalt-swift sidecar.
  - Implements the training forward: context taps → fc → 5 layers (sliding-window attn + dynamic conv) → final_norm → target head → 7-position logits.
  - AdamW optimizer with param groups (codebooks at 0.1× LR).
  - Sliding-window dataset over the taps file.
  - Cross-entropy loss over the 7 draft positions.
  - Save in the same safetensors format as the checkpoint.

### Dry-run result (CPU, synthetic data, window=64)
```
step 1/2 | loss=15.5797 | lr=1.00e-05 | 1.6s
step 2/2 | loss=13.0179 | lr=1.00e-05 | 2.4s
epoch 1/1 | avg_loss=14.2988
```
Expected: random CE over 248K vocab ≈ ln(248320) ≈ 12.4. Starting at ~15.6 and dropping
is correct for 2 steps on random data. The loop works end-to-end.

### Adaptations from the reference
1. **RoPE batch dim**: transformers 5.x `Qwen3RotaryEmbedding` returns `[seq, dim]`
   instead of `[batch, seq, dim]`. Wrapped with `_RoPEBatchWrapper`.
2. **Target head as raw weight**: the embed+head file stores the weight matrix directly
   (not an `nn.Linear` module). Used `F.linear` for the matmul.
3. **Safetensors name mapping**: `candidate_selector.{predecessor,successor}_codebook`
   in the file → `.weight` suffix in PyTorch (nn.Embedding convention).
4. **Trailing null bytes in header**: the target-embed-head safetensors file has a
   non-standard header (padded to 8-byte alignment). Handled with manual JSON parse.

### Tensor mapping (81 tensors, all verified)
- `fc.weight` [2560, 12800] — feature projection (5×2560 → 2560)
- `hidden_norm.weight` [2560] — context norm
- `norm.weight` [2560] — final norm
- `layers.{0..4}.self_attn.{q,k,v,o}_proj.weight` — attention projections
- `layers.{0..4}.self_attn.{q,k}_norm.weight` [128] — per-head RMSNorm
- `layers.{0..4}.{input,post_attention}_layernorm.weight` [2560]
- `layers.{0..4}.mlp.{gate,up,down}_proj.weight` — SwiGLU
- `layers.{0..4}.{attention,mlp}_conv.base_kernel` [2, 2, 2560] — dynamic conv base
- `layers.{0..4}.{attention,mlp}_conv.kernel_projection.weight` [640, 2560]
- `candidate_selector.hidden_projection.weight` [256, 2560]
- `candidate_selector.{predecessor,successor}_codebook` [248320, 256]

### Next steps
1. Validate the PyTorch forward against the CUDA reference (GPU 1, fixed input → 7 logits).
2. Run the 1-epoch train on GPU 1 with the real taps file.
3. Evaluate acceptance on the bench fixtures (go/no-go: ≥70%).
