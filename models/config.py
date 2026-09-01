from __future__ import annotations

from copy import deepcopy


# These are decoder-only GPT configurations. "50M" is intentionally sized close
# to the label rather than preserving the old ~34M configuration.
MODEL_CONFIGS = {
    "50M": dict(
        vocab_size=50257,
        context_length=2048,
        emb_dim=448,
        n_heads=7,
        n_layers=12,
        drop_rate=0.0,
        qkv_bias=False,
        tie_weights=True,
    ),
    "100M": dict(
        vocab_size=50257,
        context_length=2048,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.0,
        qkv_bias=False,
        tie_weights=True,
    ),
    "1.2B": dict(
        vocab_size=50257,
        context_length=2048,
        emb_dim=2048,
        n_heads=16,
        n_layers=24,
        drop_rate=0.0,
        qkv_bias=False,
        tie_weights=True,
    ),
}


def get_model_config(
    name: str,
    *,
    vocab_size: int | None = None,
    context_length: int | None = None,
) -> dict:
    if name not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODEL_CONFIGS)}")

    cfg = deepcopy(MODEL_CONFIGS[name])

    if vocab_size is not None:
        cfg["vocab_size"] = int(vocab_size)
    if context_length is not None:
        cfg["context_length"] = int(context_length)

    if cfg["emb_dim"] % cfg["n_heads"] != 0:
        raise ValueError("emb_dim must be divisible by n_heads")

    return cfg
