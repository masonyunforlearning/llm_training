from __future__ import annotations

import random
from collections.abc import Callable, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from data.config import build_language_sources
from tokenizer.factory import build_tokenizer, get_tokenizer_spec


DocumentFactory = Callable[[int, int], Iterator]


class PretrainingDataset(IterableDataset):
    """Infinite token stream with language-level weighted sampling.

    A sequence is formed by concatenating documents and inserting one EOS token
    after every document. Sequences may cross document boundaries by design.
    """

    def __init__(
        self,
        language_sources: dict[str, tuple[list[DocumentFactory], list[float]]],
        language_weights: dict[str, float],
        *,
        tokenizer_name: str,
        seq_len: int,
        seed: int = 123,
    ):
        super().__init__()

        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")

        self.language_sources = language_sources
        self.languages = list(language_weights)
        self.language_weights = [float(language_weights[x]) for x in self.languages]
        self.tokenizer_name = tokenizer_name
        self.seq_len = int(seq_len)
        self.seed = int(seed)

        if set(self.languages) != set(language_sources):
            raise ValueError(
                "language_sources and language_weights must contain the same languages"
            )
        if any(w <= 0 for w in self.language_weights):
            raise ValueError("All active language weights must be > 0")

    def _make_language_iterators(self, worker_id: int, num_workers: int):
        result = {}
        for language, (factories, weights) in self.language_sources.items():
            result[language] = {
                "factories": factories,
                "weights": weights,
                "iterators": [f(worker_id, num_workers) for f in factories],
            }
        return result

    @staticmethod
    def _next_document(state, rng: random.Random, worker_id: int, num_workers: int):
        index = rng.choices(
            range(len(state["iterators"])),
            weights=state["weights"],
            k=1,
        )[0]
        try:
            return next(state["iterators"][index])
        except StopIteration:
            state["iterators"][index] = state["factories"][index](worker_id, num_workers)
            return next(state["iterators"][index])

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        rng = random.Random(self.seed + worker_id)
        tokenizer = build_tokenizer(self.tokenizer_name)
        eos_id = get_tokenizer_spec(self.tokenizer_name).eos_id

        states = self._make_language_iterators(worker_id, num_workers)
        buffer: list[int] = []

        while True:
            language = rng.choices(
                self.languages,
                weights=self.language_weights,
                k=1,
            )[0]
            doc = self._next_document(states[language], rng, worker_id, num_workers)

            ids = tokenizer.encode(doc.text, disallowed_special=())
            if not ids:
                continue

            buffer.extend(ids)
            buffer.append(eos_id)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                del buffer[: self.seq_len]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


def normalize_mixture(mixture: dict[str, float]) -> dict[str, float]:
    if not mixture:
        raise ValueError("Mixture cannot be empty")
    if any(v < 0 for v in mixture.values()):
        raise ValueError("Mixture weights cannot be negative")
    total = sum(mixture.values())
    if total <= 0:
        raise ValueError("Mixture weight sum must be positive")
    return {k: float(v) / total for k, v in mixture.items() if v > 0}


def build_pretraining_dataset(
    mixture: dict[str, float],
    *,
    tokenizer_name: str,
    seq_len: int,
    datasets_config: str = "configs/datasets.yaml",
    seed: int = 123,
    section: str = "datasets",
) -> PretrainingDataset:
    mixture = normalize_mixture(mixture)
    language_sources = build_language_sources(
        list(mixture),
        config_path=datasets_config,
        section=section,
        seed=seed,
    )
    return PretrainingDataset(
        language_sources,
        mixture,
        tokenizer_name=tokenizer_name,
        seq_len=seq_len,
        seed=seed,
    )


def build_pretraining_dataloader(
    mixture: dict[str, float],
    *,
    tokenizer_name: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    datasets_config: str = "configs/datasets.yaml",
    seed: int = 123,
    section: str = "datasets",
) -> DataLoader:
    dataset = build_pretraining_dataset(
        mixture,
        tokenizer_name=tokenizer_name,
        seq_len=seq_len,
        datasets_config=datasets_config,
        seed=seed,
        section=section,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
