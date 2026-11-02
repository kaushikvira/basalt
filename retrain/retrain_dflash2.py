#!/usr/bin/env python3
"""DFlash2 retrain — fine-tune the 603M drafter on basalt target taps.

Adapted from z-lab/dflash (open-source PyTorch reference). Trains the 5-layer
DFlash2 drafter to predict the next 7 tokens given the target's 5-layer
hidden-state taps + the anchor token. The target's embedding + head are frozen.

v2 recipe (the go/no-go fine-tune):
  - Partial masking: each sample sees k of the 7 real tokens at random
    positions (k ~ Uniform(1..6)); CE loss is computed on the masked
    positions only (the standard block-diffusion training objective — the
    v1 full-mask recipe was not how the drafter was trained).
  - LR schedule: 1e-4 peak, 100-step warmup, cosine decay to 1e-5.
  - Shuffled training order (seeded).

Usage:
    # Dry-run (CPU, small context, 3 steps):
    python retrain_dflash2.py --dry-run

    # Full training (GPU):
    python retrain_dflash2.py --epochs 3 --batch-size 32 --lr 1e-4
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add this directory to path for the reference import
sys.path.insert(0, str(Path(__file__).parent))

from dflash_model_reference import DFlash2DraftModel
from dflash2_fn_config import load_fn_config, assert_config_invariants

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_SAFETENSORS = Path("/home/kv/work/models/dflash2-fn/model.safetensors")
TARGET_EMBED_HEAD = Path("/home/kv/models/basalt-swift/target-embed-head.bf16.safetensors")
TAPS_FILE = Path("/home/kv/work/basalt-dflash2/data/corpus/taps.npz")
CONFIG_JSON = Path("/home/kv/work/models/dflash2-fn/config.json")

# ---------------------------------------------------------------------------
# Tap file loading (custom binary format, not standard .npz)
# ---------------------------------------------------------------------------

def load_taps_file(path: Path) -> tuple[np.ndarray, torch.Tensor, dict]:
    """Load the basalt --dump-taps binary file.

    Returns (tokens[N] int32, taps[N, 5, 2560] torch.bfloat16 (CPU), meta dict).

    The taps are kept in BF16 — their on-file dtype. Loading them as F32 would
    upcast the frozen 2.5 GB target head to F32 per sample in the batch loop
    (32 × 2.5 GB) and OOM. mmap keeps the 2.4 GB file out of RAM until the
    per-record copies happen.
    """
    import struct
    size = path.stat().st_size
    if size < 20:
        raise ValueError(f"file too small: {size} bytes")
    with open(path, "rb") as f:
        magic = f.read(4)
        version, n_taps, n_embd, dtype = struct.unpack("<IIII", f.read(16))
    if magic != b"TAPS":
        raise ValueError(f"bad magic: {magic!r}")
    if version != 1:
        raise ValueError(f"unsupported version: {version}")
    if dtype != 1:
        raise ValueError(f"unsupported dtype: {dtype} (expected 1=BF16)")

    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=20)
    off = 0
    n_bytes = mm.nbytes
    tokens_list, taps_list = [], []
    while off < n_bytes:
        n_tokens = struct.unpack_from("<I", mm, off)[0]
        off += 4
        n_tok_bytes = 4 * n_tokens
        n_tap_elems = n_tokens * n_taps * n_embd
        n_tap_bytes = 2 * n_tap_elems
        if off + n_tok_bytes + n_tap_bytes > n_bytes:
            raise ValueError(f"truncated record at offset {off}")
        tokens = np.frombuffer(mm[off:off + n_tok_bytes], dtype=np.int32).copy()
        off += n_tok_bytes
        taps_raw = np.frombuffer(mm[off:off + n_tap_bytes], dtype=np.uint16).copy()
        off += n_tap_bytes
        # View the raw bytes as BF16 directly (bit-level reinterpret, lossless)
        taps_bf16 = torch.from_numpy(taps_raw).view(torch.bfloat16)
        taps_list.append(taps_bf16.reshape(n_tokens, n_taps, n_embd))
        tokens_list.append(tokens)

    tokens = np.concatenate(tokens_list)
    taps = torch.cat(taps_list)  # [N, 5, 2560] bf16
    meta = {"n_taps": n_taps, "n_embd": n_embd, "n_tokens": len(tokens)}
    return tokens, taps, meta


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# Safetensors name → PyTorch parameter name mapping.
# The only difference: nn.Embedding adds .weight suffix.
TENSOR_NAME_MAP = {
    "candidate_selector.predecessor_codebook": "candidate_selector.predecessor_codebook.weight",
    "candidate_selector.successor_codebook": "candidate_selector.successor_codebook.weight",
}


def load_drafter(model_path: Path, config) -> DFlash2DraftModel:
    """Load the DFlash2 drafter weights from safetensors into the PyTorch model."""
    from safetensors import safe_open

    model = DFlash2DraftModel(config)
    model_params = dict(model.named_parameters())

    with safe_open(model_path, framework="pt") as f:
        keys = set(f.keys())
        for key in keys:
            # Map safetensors name → PyTorch parameter name
            pt_name = TENSOR_NAME_MAP.get(key, key)
            if pt_name not in model_params:
                raise ValueError(f"Unexpected tensor in checkpoint: {key} → {pt_name}")
            tensor = f.get_tensor(key)  # read using the ORIGINAL file name
            model_params[pt_name].data.copy_(tensor)

    # Verify all parameters were loaded
    loaded_pt_names = {TENSOR_NAME_MAP.get(k, k) for k in keys}
    model_keys = set(model_params.keys())
    missing = model_keys - loaded_pt_names
    unexpected = loaded_pt_names - model_keys
    if missing:
        raise ValueError(f"Missing parameters: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected tensors: {unexpected}")

    # The checkpoint is BF16; the model was constructed in F32, so copy_ above
    # upcast every weight. Cast back to BF16 — the whole forward must run in
    # BF16 (an F32 model would upcast the frozen 2.5 GB target head per sample).
    model = model.to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded drafter: {len(keys)} tensors, {n_params:,} params (bf16)")
    return model


def load_target_embed_head(path: Path, device: str = "cpu") -> tuple[nn.Parameter, nn.Parameter]:
    """Load the frozen target embedding + head from safetensors.
    
    Handles the trailing null bytes in the header (the file was written by a
    non-standard tool that pads the header to 8-byte alignment).
    """
    import json
    import struct

    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        f.seek(8)
        raw_header = f.read(header_size)
        # Strip trailing null bytes
        json_end = raw_header.rfind(b"}") + 1
        header = json.loads(raw_header[:json_end])
        data_start = 8 + header_size

        def read_tensor(key: str) -> torch.Tensor:
            info = header[key]
            dtype_str = info["dtype"]
            shape = info["shape"]
            offsets = info["data_offsets"]
            # BF16 = 2 bytes per element
            n_bytes = (offsets[1] - offsets[0])
            raw = f.read(n_bytes) if offsets[0] == f.tell() - data_start else None
            if raw is None:
                f.seek(data_start + offsets[0])
                raw = f.read(n_bytes)
            # BF16 → F32: reinterpret upper 16 bits (via numpy to avoid non-writable buffer warning)
            u16 = torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy())
            f32 = (u16.to(torch.int32) << 16).view(torch.float32)
            return f32.to(torch.bfloat16).reshape(shape).to(device)

        embed = nn.Parameter(read_tensor("embed.weight"), requires_grad=False)
        head = nn.Parameter(read_tensor("head.weight"), requires_grad=False)

    print(f"Loaded target embed: {embed.shape}, head: {head.shape} (frozen)")
    return embed, head


# ---------------------------------------------------------------------------
# Training forward pass
# ---------------------------------------------------------------------------

class _RoPEBatchWrapper(nn.Module):
    """Wraps Qwen3RotaryEmbedding to add the batch dimension.
    
    Transformers 5.x returns [seq_len, head_dim]; the reference code expects
    [batch, seq_len, head_dim]. This wrapper adds the batch dim.
    """
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states, position_ids):
        cos, sin = self.inner(hidden_states, position_ids)
        bsz = hidden_states.shape[0]
        # transformers 5.12.x returns [batch, seq, dim]; earlier 5.x returned
        # [seq, dim]. Normalize to [batch, seq, dim] either way.
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).expand(bsz, -1, -1)
            sin = sin.unsqueeze(0).expand(bsz, -1, -1)
        return cos, sin


class DFlash2Trainer(nn.Module):
    """Wraps the drafter + frozen target head for the training loop.

    Forward: given context taps [W, 5, 2560] + anchor token id + block token ids [7],
    computes the 7-position logits via the drafter + target head.
    """

    def __init__(self, drafter: DFlash2DraftModel, target_embed: nn.Parameter,
                 target_head: nn.Parameter, config):
        super().__init__()
        self.drafter = drafter
        # Patch RoPE: transformers 5.x returns [seq, dim], reference expects [batch, seq, dim]
        self.drafter.rotary_emb = _RoPEBatchWrapper(self.drafter.rotary_emb)
        self.target_embed = target_embed
        self.target_head = target_head
        self.config = config
        self.mask_token_id = config.dflash_config["mask_token_id"]
        self.block_size = config.dflash_config["block_size"]  # 8

    def forward(self, context_taps: torch.Tensor, anchor_ids: torch.Tensor,
                target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 7-position logits for a batch of samples.

        Args:
            context_taps: [B, W, 5, 2560] — per-sample context windows
            anchor_ids: [B] — per-sample last committed token id
            target_ids: [B, 7] — target token ids (loss targets + the real
                tokens shown at the unmasked block positions)

        Returns:
            logits: [B, 7, vocab_size]
            loss_mask: [B, 7] bool — True where the block position is the
                mask token (i.e. a loss target; the unmasked positions show
                the real token to the model and are excluded from the loss)

        The drafter's forward is fully batch-compatible: fc + norm, the 5
        decoder layers (SDPA attention with a [1,1,q,k] causal mask, dynamic
        conv, MLP) and RoPE all broadcast over the batch dim, so one call
        processes all B samples (no per-sample Python loop).
        """
        B, W = context_taps.shape[0], context_taps.shape[1]
        device = context_taps.device

        # 1. Concatenate context taps: [B, W, 5, 2560] → [B, W, 12800]
        #    The drafter.forward() applies fc + hidden_norm internally.
        context_concat = context_taps.reshape(B, W, -1)  # [B, W, 12800]

        # 2. Build the block embeddings: [B, 8] ids (anchor + partial mask).
        #    Partial masking (the DFlash2 training objective): each sample
        #    sees k of the 7 real tokens at random distinct positions,
        #    k ~ Uniform(1..6). k=7 (all real) would leave no loss targets;
        #    k=0 (all masked, the v1 recipe) is excluded — the model must
        #    see at least one real token in the block. The remaining 7-k
        #    positions stay the mask token and are the only loss targets.
        n_draft = self.block_size - 1  # 7
        k = torch.randint(1, n_draft, (B,), device=device)  # real tokens/sample
        # argsort of uniform noise = a random permutation of 0..6 per row;
        # the first k_i of row i are the unmasked (real-token) positions.
        order = torch.rand(B, n_draft, device=device).argsort(dim=1)  # [B, 7]
        is_real = torch.zeros(B, n_draft, dtype=torch.bool, device=device)
        is_real.scatter_(1, order, torch.arange(n_draft, device=device)
                         .unsqueeze(0) < k.unsqueeze(1))  # [B, 7]
        block_ids = torch.full((B, self.block_size), self.mask_token_id,
                               dtype=torch.long, device=device)
        block_ids[:, 0] = anchor_ids
        block_ids[:, 1:] = torch.where(is_real, target_ids, block_ids[:, 1:])
        noise_embedding = F.embedding(block_ids, self.target_embed)  # [B, 8, 2560]

        # 3. Position ids for the full sequence (context + block).
        #    Every sample starts at position 0 → identical rows, [B, W+block].
        #    2D [batch, seq] — transformers 5.12.x RoPE expects that shape.
        position_ids = torch.arange(W + self.block_size, device=device) \
            .unsqueeze(0).expand(B, -1)

        # 4. Forward through the drafter (5 layers), batched
        #    target_hidden is the raw concat [B, W, 12800]; fc + norm inside
        hidden = self.drafter(
            position_ids=position_ids,
            noise_embedding=noise_embedding,     # [B, 8, 2560]
            target_hidden=context_concat,         # [B, W, 12800]
        )  # [B, 8, 2560]

        # 5. Take the 7 draft positions (drop the anchor at position 0)
        draft_hidden = hidden[:, 1:self.block_size, :]  # [B, 7, 2560]

        # 6. Apply the frozen target head (it's a raw weight matrix, not a module).
        #    Both sides are BF16, so .to() is a no-op — no per-sample F32 copy.
        logits = F.linear(draft_hidden, self.target_head.to(draft_hidden.dtype))  # [B, 7, vocab]

        return logits, ~is_real


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TapDataset(Dataset):
    """Sliding-window dataset over the taps file.

    Each sample at position p (where p >= W and p+7 < N):
        context_taps = taps[p-W : p]  → [W, 5, 2560]
        anchor = tokens[p-1]
        target = tokens[p : p+7]     → [7]
    """

    def __init__(self, tokens: np.ndarray, taps: np.ndarray, window: int = 2048,
                 block_size: int = 8):
        self.tokens = tokens
        self.taps = taps
        self.window = window
        self.block_size = block_size
        # Valid positions: p from W to N-7 (inclusive)
        self.start = window
        self.end = len(tokens) - block_size + 1  # exclusive
        self.n_samples = max(0, self.end - self.start)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        p = self.start + idx
        # Context: the W tokens before position p
        ctx_start = p - self.window
        context_taps = self.taps[ctx_start:p]  # [W, 5, 2560]
        anchor = self.tokens[p - 1]            # scalar
        target = self.tokens[p:p + self.block_size - 1]  # [7]
        return {
            "context_taps": context_taps,
            "anchor": anchor,
            "target": target,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate a batch of samples (all have the same context window size).

    context_taps may be GPU-resident tensors (taps preloaded to the device) or
    CPU numpy arrays (dry-run synthetic); stack accordingly so no per-batch
    CPU->GPU copy is needed in the GPU-resident case. Taps stay BF16.
    """
    ctx = [s["context_taps"] for s in batch]
    if torch.is_tensor(ctx[0]):
        context_taps = torch.stack(ctx)  # [B, W, 5, 2560] on the taps' device
    else:
        context_taps = torch.from_numpy(np.stack(ctx)).to(torch.bfloat16)
    return {
        "context_taps": context_taps,
        "anchor": torch.tensor([s["anchor"] for s in batch], dtype=torch.long),  # [B]
        "target": torch.from_numpy(
            np.stack([s["target"] for s in batch])
        ).long(),  # [B, 7]
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    # --- Load config ---
    checkpoint_dir = Path(args.checkpoint)
    config = load_fn_config(checkpoint_dir / "config.json")
    assert_config_invariants(config)
    print(f"Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"vocab={config.vocab_size}", flush=True)

    # --- Load model (BF16 end-to-end; the checkpoint is BF16) ---
    drafter = load_drafter(checkpoint_dir / "model.safetensors", config)
    drafter = drafter.to(device)

    # --- Load frozen target embed + head (BF16, frozen) ---
    target_embed, target_head = load_target_embed_head(args.target_head, device)

    # --- Build trainer ---
    trainer = DFlash2Trainer(drafter, target_embed, target_head, config).to(device)

    window = args.window
    block_size = config.dflash_config["block_size"]  # 8

    # --- Load data ---
    taps_path = Path(args.taps)
    if args.dry_run and not taps_path.exists():
        # Generate synthetic taps for the CPU dry-run (verifies the loop, not real data)
        print(f"Taps file not found ({taps_path}) — generating synthetic data for dry-run")
        n_syn = window + block_size + 10
        tokens = np.random.randint(0, 248077, size=n_syn, dtype=np.int32)
        taps = torch.randn(n_syn, 5, 2560).to(torch.bfloat16) * 0.1
        meta = {"n_taps": 5, "n_embd": 2560, "n_tokens": n_syn, "synthetic": True}
    else:
        print(f"Loading taps from {taps_path}...", flush=True)
        tokens, taps, meta = load_taps_file(taps_path)
        print(f"  tokens: {tokens.shape}, taps: {tuple(taps.shape)} {str(taps.dtype)}", flush=True)

    # --- Corpus manifest (validation only; the tokens live in the taps file) ---
    if args.corpus and Path(args.corpus).exists():
        import json as _json
        with open(args.corpus) as f:
            manifest = _json.load(f)
        mt = manifest.get("total_tokens")
        if mt is not None and mt != len(tokens):
            print(f"  WARNING: manifest total_tokens={mt} != taps tokens={len(tokens)}")
        else:
            print(f"  corpus: {Path(args.corpus).name} ({manifest.get('num_prompts')} prompts, "
                  f"{mt} tokens) — matches taps")

    if args.dry_run and not meta.get("synthetic", False):
        # Small subset for CPU dry-run
        window = min(window, 64)  # small context for CPU
        n_max = min(len(tokens), window + block_size + 10)
        tokens = tokens[:n_max]
        taps = taps[:n_max]
        print(f"  Dry-run: using {n_max} tokens, window={window}")

    dataset = TapDataset(tokens, taps, window=window, block_size=block_size)
    print(f"Dataset: {len(dataset)} samples (window={window}, block={block_size})", flush=True)

    if len(dataset) == 0:
        print("ERROR: no training samples (not enough tokens for the window)")
        sys.exit(1)

    if not args.dry_run and device.type == "cuda":
        # Preload the taps to the GPU once. They are already BF16 (their on-file
        # dtype), so this is lossless. Keeping the context BF16 keeps the whole
        # forward in BF16 — a float32 context would upcast the frozen 2.5 GB
        # target head to float32 on every sample in the batch loop (2.5 GB x 32)
        # and OOM. GPU-resident slices also avoid any per-batch CPU->GPU copy.
        dataset.taps = dataset.taps.to(device)
        print(f"  taps on {device}: {tuple(dataset.taps.shape)} "
              f"({dataset.taps.numel() * 2 / 1e9:.2f} GB, bf16)", flush=True)

    # For dry-run, limit to a few samples
    if args.dry_run:
        n_train = min(3, len(dataset))
        # Create a subset dataset
        indices = list(range(n_train))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"  Dry-run: {len(dataset)} samples")

    # Shuffled training order (seeded generator → reproducible across runs).
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size if not args.dry_run else min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        generator=gen,
    )

    # --- Optimizer ---
    # Param groups: codebooks at lower LR (fragile, large)
    codebook_params = []
    other_params = []
    for name, param in drafter.named_parameters():
        if "codebook" in name:
            codebook_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": other_params, "lr": args.lr},
        {"params": codebook_params, "lr": args.lr * 0.1},  # 10× lower for codebooks
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # --- LR schedule: warmup to peak, then cosine decay to 0.1× peak ---
    # LambdaLR multiplies each group's base LR by lr_lambda(step), so the
    # codebook group tracks the same shape at 0.1× (1e-5 → 1e-6 for the
    # codebooks when the peak is 1e-4).
    WARMUP_STEPS = 100
    total_steps = len(loader) * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training loop ---
    trainer.train()
    step = 0
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_loss_t = torch.zeros((), device=device)  # accumulate on-device; no per-step sync
        epoch_steps = 0
        for batch in loader:
            context_taps = batch["context_taps"].to(device)  # [B, W, 5, 2560]
            anchor = batch["anchor"].to(device)             # [B]
            target = batch["target"].to(device)             # [B, 7]

            # One batched forward: the drafter's 5-layer forward is fully
            # batch-compatible (linears, SDPA, RoPE, dynamic conv all broadcast
            # over B), so all B samples run in a single pass — no per-sample
            # Python loop, no GPU idle time between launches.
            logits, loss_mask = trainer(context_taps, anchor, target)  # [B, 7, vocab], [B, 7]

            # Cross-entropy on the MASKED positions only (the partial-masking
            # objective): each sample contributes 7-k_i targets (k_i real
            # tokens were shown to the model). In F32 for numerical stability
            # (a log-softmax over a 248K vocab in BF16 loses too much
            # precision); the forward stays BF16 end-to-end. Every sample has
            # ≥1 masked position (k ≤ 6), so the masked set is never empty.
            loss = F.cross_entropy(
                logits[loss_mask].float().reshape(-1, logits.shape[-1]),  # [n_masked, vocab]
                target[loss_mask],                                        # [n_masked]
            )

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(drafter.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()
            epoch_loss_t += loss.detach()
            epoch_steps += 1
            step += 1

            if step % args.log_every == 0 or (args.dry_run and step <= 3):
                elapsed = time.perf_counter() - t0
                rate = step / elapsed if elapsed > 0 else 0.0
                eta = (total_steps - step) / rate if rate > 0 else 0.0
                gpu_gb = torch.cuda.memory_allocated(device) / 1e9 \
                    if device.type == "cuda" else 0.0
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  step {step}/{total_steps} | loss={loss.item():.4f} | "
                      f"lr={lr_now:.2e} | {1000.0 / rate:.0f} ms/step | "
                      f"ETA={eta / 60:.1f}min | GPU={gpu_gb:.1f}GB",
                      flush=True)

            if args.max_steps > 0 and step >= args.max_steps:
                break

        avg_loss = (epoch_loss_t / max(1, epoch_steps)).item()
        print(f"  epoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f} "
              f"({epoch_steps} steps)", flush=True)
        if args.max_steps > 0 and step >= args.max_steps:
            break

    total_time = time.perf_counter() - t0
    print(f"\nDone. {step} steps in {total_time:.1f}s", flush=True)

    # --- Save (if not dry-run) ---
    if not args.dry_run:
        out_path = Path(args.output)
        if out_path.suffix == ".safetensors":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_drafter(drafter, out_path)
        else:
            # Directory output: model.safetensors + config.json (drop-in checkpoint dir)
            out_path.mkdir(parents=True, exist_ok=True)
            save_drafter(drafter, out_path / "model.safetensors")
            import shutil
            cfg_src = checkpoint_dir / "config.json"
            if cfg_src.exists():
                shutil.copy2(cfg_src, out_path / "config.json")
        print(f"Saved to {out_path}", flush=True)


def save_drafter(model: DFlash2DraftModel, path: Path) -> None:
    """Save the drafter weights in the same safetensors format as the checkpoint."""
    from safetensors.torch import save_file

    # Reverse the name mapping: PyTorch name → safetensors name
    reverse_map = {v: k for k, v in TENSOR_NAME_MAP.items()}
    state = {}
    for name, param in model.named_parameters():
        st_name = reverse_map.get(name, name)
        state[st_name] = param.data.contiguous()

    save_file(state, str(path))
    print(f"  {len(state)} tensors saved", flush=True)


# ---------------------------------------------------------------------------
# Dry-run test
# ---------------------------------------------------------------------------

def dry_run(checkpoint: str = str(Path(MODEL_SAFETENSORS).parent),
            target_head: str = str(TARGET_EMBED_HEAD)) -> None:
    """Quick CPU test: load model + head, run 3 training steps, print loss."""
    print("=" * 60)
    print("DRY-RUN: DFlash2 retrain (CPU, small tensors)")
    print("=" * 60)

    args = argparse.Namespace(
        device="cpu",
        dry_run=True,
        batch_size=2,
        epochs=1,
        lr=1e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        window=64,
        log_every=1,
        max_steps=0,
        seed=42,
        taps=str(TAPS_FILE),
        corpus="",
        checkpoint=checkpoint,
        target_head=target_head,
        output="/tmp/dflash2-dryrun.safetensors",
    )
    train(args)

    print("\n" + "=" * 60)
    print("DRY-RUN PASSED")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DFlash2 retrain")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick CPU test (small context, 3 steps)")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device (default: cuda:0; set CUDA_VISIBLE_DEVICES for the physical GPU)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate (codebooks get 0.1× this); 100-step "
                             "warmup then cosine decay to 0.1× peak")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=2048,
                        help="Context window size (default: 2048 = sliding_window)")
    parser.add_argument("--taps", default=str(TAPS_FILE),
                        help="Taps file (binary TAPS v1, BF16)")
    parser.add_argument("--corpus", default="",
                        help="Corpus manifest JSON (validation only)")
    parser.add_argument("--checkpoint", default=str(Path(MODEL_SAFETENSORS).parent),
                        help="Drafter checkpoint dir (model.safetensors + config.json)")
    parser.add_argument("--target-head", default=str(TARGET_EMBED_HEAD),
                        help="Frozen target embed+head safetensors")
    parser.add_argument("--seed", type=int, default=42,
                        help="Shuffle seed for the training DataLoader (default 42)")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after N steps (0 = run full epochs; for quick tests)")
    parser.add_argument("--output", default="/home/kv/work/basalt-dflash2/retrain/output/model.safetensors")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.checkpoint, args.target_head)
    else:
        train(args)


if __name__ == "__main__":
    main()
