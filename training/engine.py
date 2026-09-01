from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Callable

import torch

from training.checkpoint import load_checkpoint, save_checkpoint
from training.metrics import JsonlLogger, perplexity_from_loss
from training.utils import cosine_lr


@dataclass
class TrainingState:
    update_step: int = 0
    micro_step: int = 0
    tokens_seen: int = 0
    best_eval_loss: float | None = None

    def to_dict(self):
        return asdict(self)


def resolve_amp(device: torch.device, amp: str):
    amp = amp.lower()
    if device.type != "cuda" or amp == "none":
        return None, False

    if amp == "auto":
        amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    if amp == "bf16":
        return torch.bfloat16, False
    if amp == "fp16":
        return torch.float16, True

    raise ValueError(f"Unsupported amp mode: {amp}")


def evaluate(model, dataloader, device, amp_dtype, max_batches: int):
    model.eval()
    total_loss = 0.0
    batches = 0

    autocast = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )

    with torch.no_grad():
        for input_ids, targets in dataloader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast:
                _, loss = model(input_ids, targets)

            total_loss += float(loss.detach().cpu())
            batches += 1
            if batches >= max_batches:
                break

    model.train()
    if batches == 0:
        return None
    return total_loss / batches


def train_token_budget(
    *,
    model,
    train_loader,
    optimizer,
    device: torch.device,
    target_tokens: int,
    grad_accum: int,
    base_lr: float,
    warmup_steps: int,
    amp: str = "auto",
    max_grad_norm: float = 1.0,
    log_interval: int = 10,
    checkpoint_interval: int = 500,
    checkpoint_path: str | None = None,
    resume_path: str | None = None,
    model_config: dict | None = None,
    tokenizer_name: str = "gpt2",
    eval_loader=None,
    eval_interval: int = 500,
    eval_batches: int = 20,
    metrics_path: str | None = None,
    on_log: Callable[[dict], None] | None = None,
):
    if grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")

    amp_dtype, use_scaler = resolve_amp(device, amp)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler,
    ) if device.type == "cuda" else None

    state = TrainingState()

    if resume_path:
        payload = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
        )
        state_data = payload.get("state", {})
        state = TrainingState(**{
            key: state_data[key]
            for key in TrainingState.__dataclass_fields__
            if key in state_data
        })
        print(
            f"Resumed from {resume_path}: "
            f"update={state.update_step}, tokens={state.tokens_seen}"
        )

    seq_tokens_per_update = (
        train_loader.batch_size * model.cfg["context_length"] * grad_accum
    )
    total_updates = max(
        1,
        (target_tokens + seq_tokens_per_update - 1) // seq_tokens_per_update,
    )

    logger = JsonlLogger(metrics_path) if metrics_path else None
    model.train()
    optimizer.zero_grad(set_to_none=True)

    data_iter = iter(train_loader)
    update_loss_sum = 0.0
    update_start = time.perf_counter()

    while state.tokens_seen < target_tokens:
        for _ in range(grad_accum):
            input_ids, targets = next(data_iter)
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            autocast = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )

            with autocast:
                _, loss = model(input_ids, targets)
                scaled_loss = loss / grad_accum

            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            update_loss_sum += float(loss.detach().cpu())
            state.micro_step += 1
            state.tokens_seen += input_ids.numel()

        lr = cosine_lr(
            state.update_step,
            total_updates,
            base_lr,
            warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
            if max_grad_norm and max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )
            else:
                grad_norm = torch.tensor(0.0, device=device)
            scaler.step(optimizer)
            scaler.update()
        else:
            if max_grad_norm and max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )
            else:
                grad_norm = torch.tensor(0.0, device=device)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        state.update_step += 1

        elapsed = max(time.perf_counter() - update_start, 1e-9)
        tokens_per_sec = (
            train_loader.batch_size * model.cfg["context_length"] * grad_accum
        ) / elapsed
        mean_loss = update_loss_sum / grad_accum

        record = {
            "update": state.update_step,
            "micro_step": state.micro_step,
            "tokens_seen": state.tokens_seen,
            "loss": mean_loss,
            "lr": lr,
            "grad_norm": float(grad_norm.detach().cpu()),
            "tokens_per_sec": tokens_per_sec,
        }

        if device.type == "cuda":
            record["max_memory_gb"] = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            )

        if eval_loader is not None and state.update_step % eval_interval == 0:
            eval_loss = evaluate(
                model,
                eval_loader,
                device,
                amp_dtype,
                eval_batches,
            )
            if eval_loss is not None:
                record["eval_loss"] = eval_loss
                record["eval_ppl"] = perplexity_from_loss(eval_loss)
                if (
                    state.best_eval_loss is None
                    or eval_loss < state.best_eval_loss
                ):
                    state.best_eval_loss = eval_loss

        if logger is not None:
            logger.log(**record)
        if on_log is not None:
            on_log(record)

        if state.update_step % log_interval == 0 or state.tokens_seen >= target_tokens:
            print(
                "[UPDATE {update:07d}] tokens={tokens_seen:,} "
                "loss={loss:.4f} lr={lr:.3e} tok/s={tokens_per_sec:.1f}".format(
                    **record
                )
            )

        if (
            checkpoint_path
            and state.update_step % checkpoint_interval == 0
        ):
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                state=state.to_dict(),
                model_config=model_config or model.cfg,
                tokenizer_name=tokenizer_name,
            )

        update_loss_sum = 0.0
        update_start = time.perf_counter()

    if checkpoint_path:
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=state.to_dict(),
            model_config=model_config or model.cfg,
            tokenizer_name=tokenizer_name,
        )

    return state
