from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .streaming import stream_dataset


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dataset_registry(path: str | Path = "configs/datasets.yaml") -> dict[str, Any]:
    cfg = load_yaml(path)
    if "datasets" not in cfg:
        raise ValueError(f"{path} does not contain a top-level 'datasets' key")
    return cfg


def get_language_entries(
    language: str,
    *,
    config_path: str | Path = "configs/datasets.yaml",
    section: str = "datasets",
) -> list[dict[str, Any]]:
    cfg = load_yaml(config_path)
    entries = cfg.get(section, {}).get(language, [])
    if not entries:
        raise ValueError(
            f"No dataset entries configured for language={language!r} "
            f"in section={section!r} of {config_path}"
        )
    return list(entries)


def make_document_source(entry: dict[str, Any], *, seed: int):
    entry = dict(entry)

    def source(worker_id: int = 0, num_workers: int = 1):
        return stream_dataset(
            name=entry["name"],
            split=entry.get("split", "train"),
            text_field=entry.get("text_field", "text"),
            seed=seed,
            shuffle_buffer=int(entry.get("shuffle_buffer", 10_000)),
            config_name=entry.get("config"),
            revision=entry.get("revision"),
            worker_id=worker_id,
            num_workers=num_workers,
            skip=int(entry.get("skip", 0)),
            take=entry.get("take"),
        )

    return source, float(entry.get("weight", 1.0))


def build_language_sources(
    languages: list[str],
    *,
    config_path: str | Path = "configs/datasets.yaml",
    section: str = "datasets",
    seed: int = 123,
):
    sources = {}
    for language_index, language in enumerate(languages):
        entries = get_language_entries(
            language,
            config_path=config_path,
            section=section,
        )
        factories = []
        weights = []
        for entry_index, entry in enumerate(entries):
            factory, weight = make_document_source(
                entry,
                seed=seed + language_index * 10_000 + entry_index,
            )
            factories.append(factory)
            weights.append(weight)
        sources[language] = (factories, weights)
    return sources
