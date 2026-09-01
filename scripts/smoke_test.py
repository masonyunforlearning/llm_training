"""Offline structural smoke test for Phase A-C."""
from pathlib import Path
import sys

# 프로젝트 루트를 Python import path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.model import build_model
from tokenizer.factory import get_tokenizer_spec
from training.utils import count_parameters


def main():
    spec = get_tokenizer_spec("gpt2")
    model = build_model(
        "50M",
        vocab_size=spec.vocab_size,
        context_length=64,
    )

    x = torch.randint(0, spec.vocab_size, (2, 64))
    y = torch.randint(0, spec.vocab_size, (2, 64))

    logits, loss = model(x, y)
    loss.backward()

    assert logits.shape == (2, 64, spec.vocab_size)
    assert torch.isfinite(loss)

    print("Tokenizer:", spec.name, spec.vocab_size)
    print("Parameters:", count_parameters(model))
    print("Loss:", float(loss))
    print("Phase A-C model smoke test: PASSED")


if __name__ == "__main__":
    main()
