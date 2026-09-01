"""Backward-compatible model factory.

The model implementation now lives in the modular files under models/.
"""

from models.GPTModel import GPTModel
from models.config import MODEL_CONFIGS, get_model_config


def build_model(
    name: str,
    *,
    vocab_size: int | None = None,
    context_length: int | None = None,
) -> GPTModel:
    cfg = get_model_config(
        name,
        vocab_size=vocab_size,
        context_length=context_length,
    )
    return GPTModel(cfg)


__all__ = ["GPTModel", "MODEL_CONFIGS", "get_model_config", "build_model"]
