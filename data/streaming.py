from __future__ import annotations

import itertools
from typing import Iterator

from datasets import load_dataset

from .adapter import Document, normalize_record


def stream_dataset(
    name: str,
    split: str = "train",
    text_field: str = "text",
    seed: int = 123,
    shuffle_buffer: int = 10_000,

    *,
    config_name: str | None = None,
    revision: str | None = None,

    worker_id: int = 0,
    num_workers: int = 1,

    skip: int = 0,
    take: int | None = None,

    source_name: str | None = None,
) -> Iterator[Document]:

    kwargs = {
        "path": name,
        "split": split,
        "streaming": True,
    }

    # Hugging Face dataset configuration
    if config_name is not None:
        kwargs["name"] = config_name

    # Dataset revision pinning
    if revision is not None:
        kwargs["revision"] = revision

    ds = load_dataset(**kwargs)

    # ------------------------------------------------------------
    # Worker sharding
    # ------------------------------------------------------------

    if num_workers > 1:
        try:
            ds = ds.shard(
                num_shards=num_workers,
                index=worker_id,
            )

        except Exception:
            ds = itertools.islice(
                ds,
                worker_id,
                None,
                num_workers,
            )

    # ------------------------------------------------------------
    # Deterministic streaming shuffle
    # ------------------------------------------------------------

    if shuffle_buffer and shuffle_buffer > 1:
        ds = ds.shuffle(
            seed=seed + worker_id,
            buffer_size=shuffle_buffer,
        )

    # ------------------------------------------------------------
    # Dataset partitioning
    # ------------------------------------------------------------

    if skip > 0:
        ds = ds.skip(skip)

    if take is not None:
        ds = ds.take(take)

    # ------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------

    dataset_source = source_name or name

    for record in ds:

        doc = normalize_record(
            record,
            text_field,
            dataset_source,
        )

        if doc is not None:
            yield doc