#!/usr/bin/env python3
"""DFlash2 retrain — fine-tune the 603M drafter on basalt target taps.

Adapted from z-lab/dflash (open-source PyTorch reference). Trains the 5-layer
DFlash2 drafter to predict the next 7 tokens given the target's 5-layer
hidden-state taps + the anchor token. The target's embedding + head are frozen.

Usage:
    # Dry-run (CPU, small context, 3 steps):
    python retrain_dflash2.py --dry-run

    # Full training (GPU):
    python retrain_dflash2.py --epochs 10 --batch-size 32 --lr 1e-5
"""
from __future__ import annotations

import argparse
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

def load_taps_file(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the basalt --dump-taps binary file.

    Returns (tokens[N] int32, taps[N, 5, 2560] float32, meta dict).
    """
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20:
        raise ValueError(f"file too small: {len(data)} bytes")
    magic = data[:4]
    if magic != b"TAPS":
        raise ValueError(f"bad magic: {magic!r}")
    import struct
    version, n_taps, n_embd, dtype = struct.unpack_from("<IIII", data, 4)
    if version != 1:
        raise ValueError(f"unsupported version: {version}")
    if dtype != 1:
        raise ValueError(f"unsupported dtype: {dtype} (expected 1=BF16)")

    off = 20
    tokens_list, taps_list = [], []
    while off < len(data):
        n_tokens = struct.unpack_from("<I", data, off)[0]
        off += 4
        n_tok_bytes = 4 * n_tokens
        n_tap_elems = n_tokens * n_taps * n_embd
        n_tap_bytes = 2 * n_tap_elems
        if off + n_tok_bytes + n_tap_bytes > len(data):
            raise ValueError(f"truncated record at offset {off}")
        tokens = np.frombuffer(data, dtype=np.int32, count=n_tokens, offset=off).copy()
        off += n_tok_bytes
        taps_raw = np.frombuffer(data, dtype=np.uint16, count=n_tap_elems, offset=off)
        off += n_tap_bytes
        # BF16 → F32: upper 16 bits of F32
        taps_f32 = (taps_raw.astype(np.uint32) << 16).view(np.float32)
        taps_list.append(taps_f32.reshape(n_tokens, n_taps, n_embd))
        tokens_list.append(tokens)

    tokens = np.concatenate(tokens_list)
    taps = np.concatenate(taps_list)
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded drafter: {len(keys)} tensors, {n_params:,} params")
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
        return cos.unsqueeze(0).expand(bsz, -1, -1), sin.unsqueeze(0).expand(bsz, -1, -1)


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

    def forward(self, context_taps: torch.Tensor, anchor_id: torch.Tensor,
                target_ids: torch.Tensor) -> torch.Tensor:
        """Compute 7-position logits.

        Args:
            context_taps: [W, 5, 2560] — the context window's 5-layer taps
            anchor_id: scalar int — the last committed token id
            target_ids: [7] int — the 7 target token ids (for loss; not used in forward)

        Returns:
            logits: [7, vocab_size]
        """
        W = context_taps.shape[0]
        device = context_taps.device

        # 1. Concatenate context taps: [W, 5, 2560] → [W, 12800]
        #    The drafter.forward() applies fc + hidden_norm internally.
        context_concat = context_taps.reshape(W, -1)  # [W, 12800]

        # 2. Build the block embedding: [anchor, mask, mask, ..., mask] → [8, 2560]
        block_ids = torch.full((self.block_size,), self.mask_token_id,
                               dtype=torch.long, device=device)
        block_ids[0] = anchor_id
        noise_embedding = F.embedding(block_ids, self.target_embed)  # [8, 2560]

        # 3. Position ids for the full sequence (context + block)
        position_ids = torch.arange(W + self.block_size, device=device)

        # 4. Forward through the drafter (5 layers)
        #    target_hidden is the raw concat [W, 12800]; fc + norm applied inside
        hidden = self.drafter(
            position_ids=position_ids,
            noise_embedding=noise_embedding.unsqueeze(0),  # [1, 8, 2560]
            target_hidden=context_concat.unsqueeze(0),      # [1, W, 12800]
        )  # [1, 8, 2560]

        # 5. Take the 7 draft positions (drop the anchor at position 0)
        draft_hidden = hidden[0, 1:self.block_size, :]  # [7, 2560]

        # 6. Apply the frozen target head (it's a raw weight matrix, not a module)
        logits = F.linear(draft_hidden, self.target_head.to(draft_hidden.dtype))  # [7, 248320]

        return logits


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
    """Collate a batch of samples (all have the same context window size)."""
    return {
        "context_taps": torch.from_numpy(
            np.stack([s["context_taps"] for s in batch])
        ).float(),  # [B, W, 5, 2560]
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
    print(f"Device: {device}")

    # --- Load config ---
    config = load_fn_config(CONFIG_JSON)
    assert_config_invariants(config)
    print(f"Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"vocab={config.vocab_size}")

    # --- Load model ---
    drafter = load_drafter(MODEL_SAFETENSORS, config)
    drafter = drafter.to(device)

    # --- Load frozen target embed + head ---
    target_embed, target_head = load_target_embed_head(TARGET_EMBED_HEAD, device)

    # --- Build trainer ---
    trainer = DFlash2Trainer(drafter, target_embed, target_head, config).to(device)

    window = args.window
    block_size = config.dflash_config["block_size"]  # 8

    # --- Load data ---
    if args.dry_run and not TAPS_FILE.exists():
        # Generate synthetic taps for the CPU dry-run (verifies the loop, not real data)
        print(f"Taps file not found ({TAPS_FILE}) — generating synthetic data for dry-run")
        n_syn = window + block_size + 10
        tokens = np.random.randint(0, 248077, size=n_syn, dtype=np.int32)
        taps = np.random.randn(n_syn, 5, 2560).astype(np.float32) * 0.1
        meta = {"n_taps": 5, "n_embd": 2560, "n_tokens": n_syn, "synthetic": True}
    else:
        print(f"Loading taps from {TAPS_FILE}...")
        tokens, taps, meta = load_taps_file(TAPS_FILE)
        print(f"  tokens: {tokens.shape}, taps: {taps.shape}")

    if args.dry_run and not meta.get("synthetic", False):
        # Small subset for CPU dry-run
        window = min(window, 64)  # small context for CPU
        n_max = min(len(tokens), window + block_size + 10)
        tokens = tokens[:n_max]
        taps = taps[:n_max]
        print(f"  Dry-run: using {n_max} tokens, window={window}")

    dataset = TapDataset(tokens, taps, window=window, block_size=block_size)
    print(f"Dataset: {len(dataset)} samples (window={window}, block={block_size})")

    if len(dataset) == 0:
        print("ERROR: no training samples (not enough tokens for the window)")
        sys.exit(1)

    # For dry-run, limit to a few samples
    if args.dry_run:
        n_train = min(3, len(dataset))
        # Create a subset dataset
        indices = list(range(n_train))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"  Dry-run: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size if not args.dry_run else min(args.batch_size, len(dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
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

    # --- Training loop ---
    trainer.train()
    step = 0
    total_steps = len(loader) * args.epochs
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in loader:
            context_taps = batch["context_taps"].to(device)  # [B, W, 5, 2560]
            anchor = batch["anchor"].to(device)             # [B]
            target = batch["target"].to(device)             # [B, 7]

            # Process each sample in the batch (batch dim is outer)
            # For efficiency on GPU, we process the whole batch; on CPU dry-run
            # the batch is tiny.
            B = context_taps.shape[0]
            all_logits = []
            for b in range(B):
                logits_b = trainer(
                    context_taps[b],  # [W, 5, 2560]
                    anchor[b],        # scalar
                    target[b],        # [7] (not used in forward, just carried)
                )
                all_logits.append(logits_b)
            logits = torch.stack(all_logits)  # [B, 7, vocab]

            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),  # [B*7, vocab]
                target.reshape(-1),                    # [B*7]
            )

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(drafter.parameters(), args.grad_clip)

            optimizer.step()
            epoch_loss += loss.item()
            step += 1

            if step % args.log_every == 0 or (args.dry_run and step <= 3):
                elapsed = time.perf_counter() - t0
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  step {step}/{total_steps} | loss={loss.item():.4f} | "
                      f"lr={lr_now:.2e} | {elapsed:.1f}s")

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"  epoch {epoch+1}/{args.epochs} | avg_loss={avg_loss:.4f}")

    total_time = time.perf_counter() - t0
    print(f"\nDone. {step} steps in {total_time:.1f}s")

    # --- Save (if not dry-run) ---
    if not args.dry_run:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_drafter(drafter, out_path)
        print(f"Saved to {out_path}")


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
    print(f"  {len(state)} tensors saved")


# ---------------------------------------------------------------------------
# Dry-run test
# ---------------------------------------------------------------------------

def dry_run() -> None:
    """Quick CPU test: load model + head, run 3 training steps, print loss."""
    print("=" * 60)
    print("DRY-RUN: DFlash2 retrain (CPU, small tensors)")
    print("=" * 60)

    args = argparse.Namespace(
        device="cpu",
        dry_run=True,
        batch_size=2,
        epochs=1,
        lr=1e-5,
        weight_decay=0.01,
        grad_clip=1.0,
        window=64,
        log_every=1,
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
    parser.add_argument("--device", default="cuda:1",
                        help="Torch device (default: cuda:1)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (codebooks get 0.1× this)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=2048,
                        help="Context window size (default: 2048 = sliding_window)")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output", default="/home/kv/work/basalt-dflash2/retrain/output/model.safetensors")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        train(args)


if __name__ == "__main__":
    main()
