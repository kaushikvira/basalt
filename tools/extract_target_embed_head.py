"""Extract the target's embedding + head (dequantized to BF16) from the NVFP4+dflash2 pack.

The training script (retrain_dflash2.py) needs the target's embedding table + head as a
sidecar file. This script reads the .basalt pack, extracts:
  - token_embd.weight: Q8_0, bytes laid out as per-token rows [n_vocab, n_embd]
  - output.weight: Q8_0, bytes laid out as per-token rows [n_vocab, n_embd]

and saves both dequantized to a single safetensors file (BF16, [248320, 2560]).

Note: the meta JSON declares the logical ggml shape [n_embd, n_vocab], but the byte
layout is per-token rows (the engine's embedding_gather / NativeEmbed read
`table + token * row_bytes`), so the flat dequantized array is reshaped to
[n_vocab, n_embd] directly - no transpose.

Usage:
    python3 tools/extract_target_embed_head.py \
        --pack /home/kv/models/basalt-swift/Swift-1.5-FN-NVFP4-dflash2.basalt \
        --out /home/kv/models/basalt-swift/target-embed-head.bf16.safetensors
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Reuse the basalt tools' dequantizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggufq


def read_basalt_sections(pack_path: str) -> dict[str, tuple[int, int]]:
    """Read all section (offset, size) pairs from a .basalt file."""
    sections = {}
    with open(pack_path, "rb") as f:
        header = f.read(64)
        magic = header[:8]
        if magic != b"BASALTMF":
            raise ValueError(f"{pack_path}: not a .basalt model file (magic={magic!r})")
        n_sections = struct.unpack("<I", header[12:16])[0]
        for i in range(n_sections):
            entry = f.read(128)
            name = entry[:64].rstrip(b"\0").decode()
            off, size = struct.unpack("<QQ", entry[64:80])
            sections[name] = (off, size)
    return sections


def read_meta(pack_path: str, meta_off: int, meta_size: int) -> dict:
    """Read the meta JSON section."""
    with open(pack_path, "rb") as f:
        f.seek(meta_off)
        return json.loads(f.read(meta_size))


def read_tensor_q8_0(pack_path: str, gguf_off: int, tensor_off: int, n_bytes: int) -> np.ndarray:
    """Read a Q8_0 tensor from the gguf section and dequantize to f32."""
    print(f"  reading {n_bytes / 1e6:.0f} MB from offset {gguf_off + tensor_off} ...", flush=True)
    t0 = time.time()
    with open(pack_path, "rb") as f:
        f.seek(gguf_off + tensor_off)
        raw = f.read(n_bytes)
    if len(raw) < n_bytes:
        raise ValueError(f"short read: got {len(raw)} of {n_bytes}")
    print(f"  read in {time.time() - t0:.1f}s", flush=True)
    
    arr = np.frombuffer(raw, dtype=np.uint8)
    f32 = ggufq.dequantize(arr, ggufq.GGMLQuantizationType.Q8_0)
    return f32


def f32_to_bf16_uint16(f32: np.ndarray) -> np.ndarray:
    """Convert f32 array to BF16 (as uint16). Round to nearest even."""
    u32 = f32.view(np.uint32)
    u32 = (u32 + np.uint32(0x7fff) + ((u32 >> 16) & np.uint32(1))) >> 16
    return u32.astype(np.uint16)


def write_safetensors_bf16(path: str, tensors: dict[str, np.ndarray]) -> None:
    """Write a safetensors file with BF16 tensors (uint16 arrays).
    
    Format: 8-byte LE u64 header_size + JSON header (padded to 8) + tensor data.
    """
    data_parts = []
    header_entries = {}
    data_pos = 0
    for name, arr in tensors.items():
        raw = arr.tobytes()
        pad = (-data_pos) % 8
        if pad:
            data_parts.append(b"\0" * pad)
            data_pos += pad
        header_entries[name] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [data_pos, data_pos + len(raw)],
        }
        data_parts.append(raw)
        data_pos += len(raw)

    header_json = json.dumps(header_entries, separators=(",", ":")).encode("utf-8")
    header_pad = (-len(header_json)) % 8
    header_json += b"\0" * header_pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for part in data_parts:
            f.write(part)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True, help="the .basalt pack file")
    ap.add_argument("--out", required=True, help="output safetensors path")
    args = ap.parse_args()

    print(f"Reading pack: {args.pack}", flush=True)
    sections = read_basalt_sections(args.pack)
    
    if "gguf" not in sections:
        raise ValueError("no 'gguf' section in the pack")
    if "meta" not in sections:
        raise ValueError("no 'meta' section in the pack")
    
    gguf_off, gguf_size = sections["gguf"]
    meta_off, meta_size = sections["meta"]
    print(f"  gguf: offset={gguf_off} size={gguf_size / 1e9:.2f} GB", flush=True)
    
    meta = read_meta(args.pack, meta_off, meta_size)
    tensors_meta = meta["tensors"]
    
    # Find the embedding and head
    for name in ["token_embd.weight", "output.weight"]:
        if name not in tensors_meta:
            raise ValueError(f"{name} not found in the pack's meta")
        print(f"  {name}: type={tensors_meta[name]['type']} shape={tensors_meta[name]['shape']}", flush=True)
    
    # The pack's bytes are per-token rows: [n_vocab, n_embd] = [248320, 2560].
    # Both are Q8_0 in the NVFP4 pack.

    # Read and dequantize the embedding
    print("\nExtracting embedding (Q8_0 → BF16) ...", flush=True)
    t0 = time.time()
    emb_meta = tensors_meta["token_embd.weight"]
    emb_f32 = read_tensor_q8_0(args.pack, gguf_off, emb_meta["offset"], emb_meta["bytes"])
    emb_f32 = emb_f32.reshape(248320, 2560)  # per-token rows, as the engine reads them
    emb_bf16 = f32_to_bf16_uint16(emb_f32)
    del emb_f32
    print(f"  embedding: {emb_bf16.shape} ({emb_bf16.nbytes / 1e6:.0f} MB) in {time.time() - t0:.1f}s", flush=True)
    
    # Read and dequantize the head
    print("\nExtracting head (Q8_0 → BF16) ...", flush=True)
    t0 = time.time()
    head_meta = tensors_meta["output.weight"]
    head_f32 = read_tensor_q8_0(args.pack, gguf_off, head_meta["offset"], head_meta["bytes"])
    head_f32 = head_f32.reshape(248320, 2560)  # per-token rows, as the engine reads them
    head_bf16 = f32_to_bf16_uint16(head_f32)
    del head_f32
    print(f"  head: {head_bf16.shape} ({head_bf16.nbytes / 1e6:.0f} MB) in {time.time() - t0:.1f}s", flush=True)
    
    # Verify shapes
    assert emb_bf16.shape == (248320, 2560), f"unexpected embedding shape: {emb_bf16.shape}"
    assert head_bf16.shape == (248320, 2560), f"unexpected head shape: {head_bf16.shape}"
    
    # Write the safetensors file
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {out_path} ...", flush=True)
    t0 = time.time()
    write_safetensors_bf16(str(out_path), {
        "embed.weight": emb_bf16,
        "head.weight": head_bf16,
    })
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  written {size_mb:.0f} MB in {time.time() - t0:.1f}s", flush=True)
    
    # Verify: reload and check
    print("\nVerifying output ...", flush=True)
    with open(out_path, "rb") as f:
        hs = struct.unpack("<Q", f.read(8))[0]
        hjson = json.loads(f.read(hs).rstrip(b"\0"))
        for name, m in hjson.items():
            print(f"  {name}: dtype={m['dtype']} shape={m['shape']}", flush=True)
            assert m["dtype"] == "BF16"
            assert m["shape"] == [248320, 2560]
    print("  OK", flush=True)
    
    # Spot-check: compare a few values
    print("\nSpot-check (first 4 values of each) ...", flush=True)
    # Re-read a small slice from the pack for comparison
    with open(out_path, "rb") as f:
        hs = struct.unpack("<Q", f.read(8))[0]
        hjson = json.loads(f.read(hs).rstrip(b"\0"))
        # Read the first 8 BF16 values of embed.weight
        e_off, e_end = hjson["embed.weight"]["data_offsets"]
        f.seek(e_off)
        e_bytes = f.read(8)
        e_vals = np.frombuffer(e_bytes, dtype=np.uint16).astype(np.uint32) << 16
        e_f32 = e_vals.view(np.float32)
        print(f"  embed.weight[:4] = {e_f32}", flush=True)
        # Read the first 8 BF16 values of head.weight
        h_off, h_end = hjson["head.weight"]["data_offsets"]
        f.seek(h_off)
        h_bytes = f.read(8)
        h_vals = np.frombuffer(h_bytes, dtype=np.uint16).astype(np.uint32) << 16
        h_f32 = h_vals.view(np.float32)
        print(f"  head.weight[:4]  = {h_f32}", flush=True)
    
    print(f"\nDone. Output: {out_path} ({size_mb:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
