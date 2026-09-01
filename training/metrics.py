from __future__ import annotations

import json
import math
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **record):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def perplexity_from_loss(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")
