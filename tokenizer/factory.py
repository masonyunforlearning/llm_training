from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken
import yaml


@dataclass(frozen=True)
class TokenizerSpec:
    type: str
    name: str
    vocab_size: int
    eos_token: str
    eos_id: int


def load_tokenizer_config(path: str | Path = "configs/tokenizer.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_tokenizer(
    name: str | None = None,
    config_path: str | Path = "configs/tokenizer.yaml",
):
    cfg = load_tokenizer_config(config_path)
    token_cfg = cfg.get("tokenizer", {})

    tokenizer_type = token_cfg.get("type", "tiktoken")
    if tokenizer_type != "tiktoken":
        raise ValueError(
            f"Unsupported tokenizer type: {tokenizer_type}. "
            "This project currently implements tiktoken-backed tokenizers."
        )

    encoding_name = name or token_cfg.get("name", "gpt2")
    tokenizer = tiktoken.get_encoding(encoding_name)
    return tokenizer


def get_tokenizer_spec(
    name: str | None = None,
    config_path: str | Path = "configs/tokenizer.yaml",
) -> TokenizerSpec:
    cfg = load_tokenizer_config(config_path)
    token_cfg = cfg.get("tokenizer", {})
    tokenizer = build_tokenizer(name=name, config_path=config_path)

    eos_token = token_cfg.get("eos", "<|endoftext|>")
    if eos_token in tokenizer.special_tokens_set:
        eos_id = tokenizer.encode_single_token(eos_token)
    else:
        # cl100k_base does not expose GPT-2's endoftext as a normal special token.
        # Its canonical EOT token is exposed directly by tiktoken.
        eos_id = getattr(tokenizer, "eot_token", None)
        if eos_id is None:
            raise ValueError(f"Could not determine EOS id for tokenizer {tokenizer.name!r}")

    return TokenizerSpec(
        type=token_cfg.get("type", "tiktoken"),
        name=tokenizer.name,
        vocab_size=tokenizer.n_vocab,
        eos_token=eos_token,
        eos_id=int(eos_id),
    )


def encode_text(tokenizer, text: str) -> list[int]:
    # Keep special-token handling explicit. Dataset text is treated as ordinary text.
    return tokenizer.encode(text, disallowed_special=())


def encode_with_eos(tokenizer, text: str, eos_id: int) -> list[int]:
    return encode_text(tokenizer, text) + [eos_id]
