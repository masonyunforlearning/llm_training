from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scaler,
    state: dict,
    model_config: dict,
    tokenizer_name: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "state": state,
        "model_config": model_config,
        "tokenizer_name": tokenizer_name,
        "rng_state": capture_rng_state(),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer=None,
    scaler=None,
    map_location="cpu",
    restore_rng: bool = True,
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])

    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])

    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])

    if restore_rng and payload.get("rng_state") is not None:
        restore_rng_state(payload["rng_state"])

    return payload
