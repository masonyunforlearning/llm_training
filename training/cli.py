from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.pipeline import build_pretraining_dataloader, normalize_mixture
from models.model import build_model
from tokenizer.factory import get_tokenizer_spec
from training.engine import train_token_budget
from training.utils import count_parameters, seed_everything


def parse_mixture(value: str) -> dict[str, float]:
    try:
        parsed = {
            key.strip(): float(weight)
            for key, weight in (
                item.split(":", 1) for item in value.split(",") if item.strip()
            )
        }
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "Mixture must look like 'ko:0.3,en:0.7'"
        ) from e

    return normalize_mixture(parsed)


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--mixture", required=True, type=parse_mixture)
    parser.add_argument("--train-tokens", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=123)
    return parser

"""
Author: Mason Yun 
Description: main training llm
"""
def run_training(args, model_name: str):
    seed_everything(args.seed)

    tokenizer_spec = get_tokenizer_spec(args.tokenizer)
    model = build_model(
        model_name,
        vocab_size=tokenizer_spec.vocab_size,
        context_length=args.seq_len,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("=" * 72)
    print(f"Model              : {model_name}")
    print(f"Parameters         : {count_parameters(model):,}")
    print(f"Tokenizer          : {tokenizer_spec.name}")
    print(f"Vocab size         : {tokenizer_spec.vocab_size:,}")
    print(f"Sequence length    : {args.seq_len}")
    print(f"Mixture            : {args.mixture}")
    print(f"Target tokens      : {args.train_tokens:,}")
    print(f"Device             : {device}")
    if device.type == "cuda":
        print(f"GPU                : {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    train_loader = build_pretraining_dataloader(
        args.mixture,
        tokenizer_name=tokenizer_spec.name,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        datasets_config=args.datasets_config,
        seed=args.seed,
        section="datasets",
    )

    # Validation is enabled only when every active language has an explicitly
    # configured validation source. This avoids silently evaluating on the
    # training stream.
    from data.config import load_yaml
    registry = load_yaml(args.datasets_config)
    validation_registry = registry.get("validation", {})
    has_validation = all(validation_registry.get(lang) for lang in args.mixture)

    eval_loader = None
    if has_validation:
        eval_loader = build_pretraining_dataloader(
            args.mixture,
            tokenizer_name=tokenizer_spec.name,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            datasets_config=args.datasets_config,
            seed=args.seed + 999_999,
            section="validation",
        )
        print("Validation         : enabled (explicit validation section)")
    else:
        print("Validation         : disabled (no complete explicit validation sources)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )

    return train_token_budget(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        target_tokens=args.train_tokens,
        grad_accum=args.grad_accum,
        base_lr=args.lr,
        warmup_steps=args.warmup_steps,
        amp=args.amp,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=str(output_dir / "latest.pt"),
        resume_path=args.resume,
        model_config=model.cfg,
        tokenizer_name=tokenizer_spec.name,
        eval_loader=eval_loader,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        metrics_path=str(output_dir / "metrics.jsonl"),
    )
