import math
import random

import numpy as np
import torch


def seed_everything(seed=123, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def count_parameters(model, unique: bool = True):
    if not unique:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    seen = set()
    total = 0
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def cosine_lr(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)

    progress = min(
        1.0,
        (step - warmup_steps) / max(1, total_steps - warmup_steps),
    )
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))
