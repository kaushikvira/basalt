# DFlash2 — Strata / z-lab / basalt comparison

Read-only research to help debug the acceptance regression (12–14% → 0.3–1.1% on the
NVFP4 pack). Compares the three implementations stage by stage. No code changes, no GPU.

Date: 2026-11-02. Worktree `~/work/basalt-dflash2`, branch `feat/dflash2`.

## 0. TL;DR

- **The CUDA forward matches the references at every stage** (fc+norm, dynamic conv,
  symmetric sliding-window attention, RoPE θ=10000, SwiGLU MLP, final norm, Q8_0 head,
  greedy selector walk). The Phase-1 gate (`tests/dflash2/gate.cpp` vs
  `make_fixture.py`'s NumPy reference) passes with fixed inputs: `drafts 7/7 = ref,
  deterministic`.
- **The decode loop matches Strata's methodology** (anchor + mask block, absolute
  positions, leading-match verify against the engine's own greedy tokens).
- **The tap is the hyper-connection mix** (mean of the 4 gated streams) read after the
  tapped layer's FFN write — the quantity Strata says the drafter was trained on.
- **The pack weights are exact**: both `Swift-1.5-FN-NVFP4-dflash2.basalt` (original
  drafter) and `…-ft.basalt` (fine-tuned) carry byte-identical pool tensors to their
  source safetensors (norms upcast BF16→F32 as designed). Pack meta is correct
  (`rope_theta 10000`, `is_causal false`, `sliding_window 2048`, taps `[5,15,25,35,45]`).
- **No code diff** in the dflash2 path between the binary that measured 12–14%
  (~19:30, `6db0b2a8`) and the go/no-go binary (`c6a5bf97`): the intervening commits
  (`819e9487`, `f4f4e693`, `9bb0ae3f`) are neutral on the default, non-dump path.
- **Two latent prefill-seed bugs** (long prompts only — NOT the bench fixtures, which
  are 17–80 tokens): the generate path appends stale rolling-buffer rows into the wrong
  cyclic slots; the serve path reads the TAIL of an n_ids-row buffer that the tap only
  ever writes at the HEAD (uninitialized memory).
- **Bottom line:** the comparison does **not** surface a forward/decode discrepancy that
  explains a 10–40× drop on the short-prompt fixtures. The leading explanations are
  (a) the 12–14% baseline was on IQ3_S-RCO (or a flawed NVFP4 number) and the drafter is
  simply out-of-distribution on basalt's NVFP4 target, and (b) a subtle tap/feature bug
  the fixed-input gate does not cover. See §4–§5.

## 1. Architecture comparison

The drafter: 5-layer dense Qwen3-geometry block (H=2560, 32q/8kv heads, head_dim 128,
intermediate 7680, vocab 248320, block 8, conv group 16 / kernel 2, selector rank 256 /
top-k 16, taps [5,15,25,35,45], mask 248077, θ=10000, window 2048, `is_causal: false`).

| Stage | z-lab (`dflash_model_reference.py`) | Strata (`dflash2_ref.py`) | basalt (`forward.cu/.cpp`) | Match |
|---|---|---|---|---|
| Context features | `concat(hidden_states[lid+1])` for lid in taps (layer output, 0-indexed) | tap file: "mean of the 4 hyper-connection streams" per tapped layer | `gr_read` HC mix after tapped layer's FFN write (`record.cpp::tap`) | **Yes** (same layer outputs; HC mix is the Flash-Next "hidden state") |
| fc + norm | `hidden_norm(fc(target_hidden))` | `rms(feats@fc.T, hidden_norm)` | `gemm(fc,feats)` → `rmsnorm(context_norm)` | **Yes** |
| Context K/V | `k_norm(k_proj(ctx))` + RoPE(abs pos); `v_proj(ctx)` | same | `gemm(k_proj,ctx)` → `rmsnorm_rope(k_norm,pos)`; `gemm(v_proj,ctx)`; cyclic slot `pos%win` | **Yes** |
| Block input | `embedding([anchor]+masks) * scale(1.0)` | `embed[ids]` | Q8_0 target embedding gather → bf16 | **Yes** (Q8_0 is the pack's embedding) |
| Per-layer: norm→conv→attn | `input_layernorm` → `attention_conv.prepare` → q/k/v (+`q_norm`/`k_norm`+RoPE) → attn → `o_proj` → `attention_conv.finish` → +res | same | `rmsnorm` → `gemm(kernel_proj)` → `conv_in` → q/k/v `gemm` → `rmsnorm_rope` → `swa` → `o_proj gemm` → `conv_out` → +res | **Yes** |
| Dynamic conv | `(base[o]+dyn[n,o,g])·x[n-o]`, o∈{0,1}, group 16; `prepare`=side0, `finish`=side1 | same | `conv_in`/`conv_out` kernels, same layout (`d[n,0:G]`,`[G:2G]`,`[2G:3G]`,`[3G:4G]`) | **Yes** |
| Attention mask | `is_causal=false` ⇒ `|qpos−kpos| < 2048` over ctx+block (symmetric) | `abs(pos−kpos) < win` over `[lo,s)` ctx + block | `swa`: ctx `[max(0,pi−win+1), frontier)` + **all** W block keys (symmetric) | **Yes** (the `is_causal:false` in the config makes z-lab symmetric, not causal) |
| Per-head RMSNorm | `q_norm`/`k_norm` over head_dim (128) | same | `rmsnorm_rope` per 128-dim head | **Yes** |
| RoPE | θ from config (10000), rotate_half, abs positions | `inv_freq=θ^(−2j/128)`, pair (j,j+64) | `ang=p·θ^(−2d/128)`, pair (d,d+64) | **Yes** (θ=10000 in pack meta) |
| MLP | SwiGLU `silu(gate)·up → down` | same | `gemm gate/up` → `swiglu` → `gemm down` | **Yes** |
| Final norm + head | `norm(hidden)`, rows 1..b−1 → target head → logits | `rms(h,norm)`, rows 1.. | `pack_tail(residual,final_norm)` → K rows → Q8_0 head `gemm` | **Yes** (Q8_0 is the pack's head) |
| Selector | `unary=topk(logits,16)`; `edge=unary+Σ_r pred_cb[r,pred]·hid[r]·succ_cb[r,cand]`; greedy argmax; lowest rank on ties | same | `topk(16)` → `hidden_r=sel_hidden@hidden_full` → `selector_walk` (same edge, lowest rank on ties) | **Yes** |

Notes:
- The `is_causal: false` field is the key that makes z-lab's attention **symmetric**
  (bidirectional over the block). A causal reading of the reference would be a mistake;
  basalt's `swa` and Strata's `draft()` both use the symmetric form, so all three agree.
- The gate fixture (`make_fixture.py::reference`) implements the same symmetric SWA and
  is what `gate.cpp` verifies the CUDA against — so the forward is validated against a
  reference that matches z-lab/Strata.

## 2. Decode-loop comparison (Strata `run_eval.py` vs basalt `decode.cpp`)

| Aspect | Strata (offline) | basalt (live) | Match |
|---|---|---|---|
| Context source | full-sequence tap file; `set_context` builds K/V for all positions | local cyclic K/V seeded by prefill (prompt tail) + per-round `append_context` (committed tokens) | equivalent (both = past taps) |
| Anchor | `seq[s]` at position `s` | `outv[a]` (last accepted) at position `p+a`; block at `[p+a+1, …)` | **Yes** (anchor is the next window's first token) |
| Block | `[anchor] + [mask]·(b−1)`, positions `[s, s+b)` | `[anchor] + [mask]·K`, positions `[pa, pa+W)`, `pa=p+a+1` | **Yes** |
| Draft | selector walk → `path` (b−1 tokens) | `selector_walk` → `drafts` (K tokens) | **Yes** |
| Verify | `a = leading matches of path vs seq[s+1:s+b]` (engine's greedy) | `a = leading matches of window[1..] vs outv[0..]` (window's greedy) | **Yes** |
| Advance | `s += a+1` | `p += a+1; x = outv[a]` | **Yes** |
| Tap → K/V | n/a (precomputed) | readback rows `0..a` → `append_context(a+1, p)` (committed tokens at `[p, p+a+1)`) | **Yes** (matches NInfer "rows 0..a") |

The off-by-ones that were fixed in Phase 3 (`6db0b2a8`) — the decode-append row shift
and the draft-anchor position — are correct in the current code.

## 3. Differences found

1. **Prefill seed — generate path (long prompts only).** `prefill.cpp` sets
   `dflash2_tap_n = min(sliding_window, n_batched)` but the rolling tap buffer only
   holds the **last chunk's** rows (`run.cpp::tap` writes `min(T_chunk,2048)` rows at the
   buffer HEAD, overwriting per chunk). When `last_chunk < 2048` (prompt length not a
   multiple of the chunk size), the readback pulls stale rows and `append_context` stores
   every feature in a cyclic slot shifted by `(2048−last_chunk) mod 2048` (plus the
   prior chunk's data). Corrupts the drafter's context K/V for the first ~2048 generated
   tokens. **Not applicable to the bench fixtures** (17–80 tokens ⇒ single chunk, all
   rows valid). The `last_chunk` clamp is only applied in `--dump-taps` mode.

2. **Prefill seed — serve path (long prompts only).** `request_prefill.cpp` allocates an
   `n_ids`-row tap buffer and the readback copies the **TAIL**
   (`dflash2_pf_tap_ + (n_ids−tap_n)`), but `run.cpp::tap` writes only the **HEAD**
   (rows `[0, last_chunk)`). For `n_ids > 2048` the tail is never written ⇒ the seed is
   uninitialized device memory. **Not applicable to the bench fixtures.**

3. **hc8 8-bit tap path.** `hc8_enabled()` is `true`; the tap's `gr_read` takes the
   int8+fp16-scale copies of `output_hc_down/up` when present. The per-element int8
   error (~0.8%) averages out over the 10240-wide dot product to ~0.01% relative on the
   mix — **negligible**, not a regression driver. (The `fused_gr_check` bit-for-bit
   guarantee applies to the fused variants, not to this direct `hc8_gr_read_multi` call,
   but the magnitude is immaterial.)

4. **BF16 feature rounding.** `append_context` rounds the f32 taps to bf16 on the host
   before the fc GEMM. Matches z-lab's bf16 on-file taps; Strata's offline port uses
   fp32. Minor, consistent with the training dtype.

5. **Q8_0 head + embedding.** The NVFP4 pack's head and embedding are Q8_0; the
   references dequantize to full precision. Inherent to the pack (present in both the
   12–14% and the 0.3–1.1% measurements), so it cannot explain the delta — but it does
   cap the absolute acceptance.

## 4. Hypotheses (ranked)

1. **The 12–14% baseline was on IQ3_S-RCO, not NVFP4 (or the NVFP4 number was
   misattributed/flawed).** The Phase-3 determination measured 14.9/26.0/10.3% on
   IQ3_S-RCO and explicitly found the drafter "out-of-distribution" there. The ~19:30
   entry ("DFlash2 12–14% vs MTP 65–91% on both IQ3_S and NVFP4") is ambiguous and may
   be restating the IQ3_S numbers. If NVFP4 was always ~0.3–1.1% in basalt, there is no
   code regression — the drafter (trained against the NInfer/FP8 target) is simply a poor
   fit for basalt's NVFP4 4-bit quantization. The fine-tuned drafter (trained on 99.5K
   real basalt-NVFP4 taps, loss 14.7→11.2, still falling) scoring 0.3–0.6% is consistent
   with "1 epoch was not enough," not with a forward bug.

2. **A subtle tap/feature bug the fixed-input gate does not cover.** The gate validates
   the forward on a synthetic fixture (random tokens/features, n_ctx=64); it does not
   validate the live tap extraction, the prefill seed, or the decode loop on real
   sequences. A bug in the HC-mix read, the tap position pairing, or the seed would show
   as ~0.5% acceptance while the forward gate stays green. This is the retrain LOG's
   hypothesis ("tap extraction / fc+norm / RoPE positions / anchor handling").

3. **The 19:30 NVFP4 measurement was flawed** (different binary build, run shape, or
   GPU state). The binary at 19:30 may not have been the `6db0b2a8` build (the last
   recorded rebuild was 14:46, pre-`6db0b2a8`), which carried the two off-by-ones — but
   those *lower* acceptance, so a stale binary would read lower, not 12–14%.

## 5. Recommended next steps

1. **Re-establish the current number.** Re-run the harness on
   `Swift-1.5-FN-NVFP4-dflash2.basalt` with the current binary (`c6a5bf97`),
   `--spec 8 --greedy --seed 42 --max-new 512`, conv/struct/prose. Confirm 0.3–1.1%.
   Also re-run on `Swift-1.5-FN-IQ3_S-RCO-dflash2.basalt` to confirm 12–14% still holds
   there. This separates "regression" from "NVFP4 was always low."
2. **Validate the live tap features against the PyTorch reference.** Dump the engine's
   taps (`--dump-taps`) on a fixed prompt; in Python, recompute the HC mix from the same
   residual (or compare the dumped features to a bf16 reference forward's
   `hidden_states[lid+1]`). Any systematic offset/localization bug surfaces here. This is
   the highest-value check and is what the retrain LOG's "Next steps" calls for.
3. **Validate the forward on REAL inputs, not just the fixture.** Feed a real prompt's
   taps + anchor through both the CUDA `draft()` and the PyTorch/NumPy reference; compare
   the 7 drafted tokens. Catches any forward divergence the fixed-input gate misses.
4. **Confirm the prefill seed is correct for the fixtures** (short prompts ⇒ single
   chunk ⇒ valid). Then **fix the two latent long-prompt seed bugs** (generate: clamp
   `dflash2_tap_n` to `last_chunk`; serve: read the buffer HEAD, or size it to the window
   and read the tail) so long-context serving is not silently corrupted.
5. **Only after the port accepts ≥ ~50% with the original drafter**, re-run the fine-tune
   with more epochs/data (loss was still falling at 1 epoch).

## Files referenced
- `retrain/strata-ref/{dflash2_ref.py,run_eval.py,run_all2.py,orch.py,README.md}`
- `retrain/{dflash_model_reference.py,retrain_dflash2.py,dflash2_fn_config.py,LOG.md}`
- `src/core/dflash2/{forward.cu,forward.cpp,loader.cpp}`, `include/basalt/core/dflash2.hpp`
- `src/program/generate/{decode.cpp,prefill.cpp,sessions.cpp,weights.cpp}`
- `src/program/serve/{request_decode.cpp,request_prefill.cpp,verify.cpp}`
- `src/core/verify/{record.cpp,hc8.cpp}`, `src/prefill/run.cpp`, `src/kernels/cuda/{gr.cu,cvec.cu,hc_fused.cu}`
- `tests/dflash2/{make_fixture.py,dbg_ref.py,gate.cpp,harness.py}`
- `specs/dflash2-port/LOG.md` (v-llm-gateway), `retrain/LOG.md`
- Packs: `/home/kv/models/basalt-swift/Swift-1.5-FN-NVFP4{-,-dflash2,-dflash2-ft}.basalt`
