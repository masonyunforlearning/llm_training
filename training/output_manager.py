from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_model_safetensors(path: Path, model: torch.nn.Module) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    tmp = path.with_suffix(path.suffix + '.tmp')
    save_file(state, str(tmp))
    os.replace(tmp, path)
    return path


def copy_tokenizer_artifact(tokenizer_name: str, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # tiktoken names are registry identifiers, not portable files. Persist metadata instead.
    p = out_dir / 'tokenizer_manifest.json'
    atomic_json(p, {'tokenizer_name': tokenizer_name})
    return {'tokenizer_manifest': str(p.name)}


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def write_manifest(output_dir: Path, *, run: dict[str, Any], files: dict[str, Path]) -> None:
    payload = {'version': 1, 'run': run, 'files': {k: {'path': str(v.relative_to(output_dir)), 'bytes': file_size(v)} for k,v in files.items() if v.exists()}}
    atomic_json(output_dir / 'manifest.json', payload)


def retain_milestones(dir_: Path, keep: int) -> None:
    if keep < 0:
        return
    files = sorted(dir_.glob('model_tokens_*.safetensors'), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[keep:]:
        p.unlink(missing_ok=True)


def atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    with open(src, 'rb') as r, open(tmp, 'wb') as w:
        shutil.copyfileobj(r, w, length=16 * 1024 * 1024)
        w.flush(); os.fsync(w.fileno())
    os.replace(tmp, dst)
