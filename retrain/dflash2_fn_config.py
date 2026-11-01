"""Flash Next DFlash2 config — built from the checkpoint's config.json.

The Qwen3Config from transformers accepts all the DFlash2-specific fields
(dflash_config dict, num_target_layers, layer_types, etc.) as extra kwargs.
No custom config class is needed.
"""
import json
from pathlib import Path

from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

CONFIG_PATH = Path("/home/kv/work/models/dflash2-fn/config.json")


def load_fn_config(path: Path | str = CONFIG_PATH) -> Qwen3Config:
    """Load the Flash Next DFlash2 config from its config.json."""
    with open(path) as f:
        raw = json.load(f)
    return Qwen3Config(**raw)


def assert_config_invariants(cfg: Qwen3Config) -> None:
    """Sanity-check the dimensions match the Flash Next checkpoint."""
    dc = cfg.dflash_config
    assert cfg.hidden_size == 2560, f"hidden_size={cfg.hidden_size}"
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.intermediate_size == 7680
    assert cfg.num_hidden_layers == 5
    assert cfg.vocab_size == 248320
    assert cfg.max_position_embeddings == 262144
    assert cfg.rms_norm_eps == 1e-6
    assert cfg.sliding_window == 2048
    assert cfg.num_target_layers == 48
    assert dc["block_size"] == 8
    assert dc["conv_group_size"] == 16
    assert dc["conv_kernel_size"] == 2
    assert dc["mask_token_id"] == 248077
    assert dc["selector_rank"] == 256
    assert dc["selector_top_k"] == 16
    assert dc["target_layer_ids"] == [5, 15, 25, 35, 45]
    # RoPE theta: the config.json carries it in rope_parameters
    rope = getattr(cfg, "rope_parameters", {})
    assert rope.get("rope_theta", 10000) == 10000, f"rope_theta={rope.get('rope_theta')}"


if __name__ == "__main__":
    cfg = load_fn_config()
    assert_config_invariants(cfg)
    print("Config OK:")
    print(f"  hidden_size={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
          f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads}, "
          f"head_dim={cfg.head_dim}")
    print(f"  vocab={cfg.vocab_size}, sliding_window={cfg.sliding_window}")
    print(f"  dflash_config={cfg.dflash_config}")
    print(f"  rope_parameters={cfg.rope_parameters}")
