# DFlash2 Retrain — Log

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
