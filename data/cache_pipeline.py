from __future__ import annotations

import json, os, random, shutil
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from data.config import get_language_entries
from data.streaming import stream_dataset
from tokenizer.factory import build_tokenizer, get_tokenizer_spec

DTYPE = np.uint32
BYTES_PER_TOKEN = np.dtype(DTYPE).itemsize


def _atomic_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@dataclass
class SourceProgress:
    documents_seen: int = 0
    tokens_written: int = 0


@dataclass
class ProducerState:
    shard_id: int = 0
    source_progress: dict[str, SourceProgress] = field(default_factory=dict)
    rng_state: object | None = None
    total_tokens_written: int = 0
    total_documents_seen: int = 0

    def to_dict(self):
        d = asdict(self)
        # random state is tuple and JSON cannot preserve tuple exactly; repr is portable enough here.
        d['rng_state'] = repr(self.rng_state) if self.rng_state is not None else None
        return d


class RollingTokenCache:
    def __init__(self, root: str | Path, max_bytes: int, shard_bytes: int):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes); self.shard_bytes = int(shard_bytes)

    def shard_path(self, shard_id: int) -> Path:
        return self.root / f'shard_{shard_id:08d}.bin'

    def meta_path(self, shard_id: int) -> Path:
        return self.root / f'shard_{shard_id:08d}.json'

    def complete_shards(self):
        return sorted(self.root.glob('shard_*.json'))

    def total_bytes(self):
        return sum(p.stat().st_size for p in self.root.glob('shard_*.bin'))

    def list_ids(self):
        return sorted(int(p.stem.split('_')[1]) for p in self.complete_shards())

    def write_shard(self, shard_id: int, tokens: list[int], meta: dict):
        path = self.shard_path(shard_id); tmp = path.with_suffix('.bin.tmp')
        np.asarray(tokens, dtype=DTYPE).tofile(tmp)
        os.replace(tmp, path)
        meta.update({'shard_id': shard_id, 'tokens': len(tokens), 'bytes': len(tokens)*BYTES_PER_TOKEN})
        _atomic_json(self.meta_path(shard_id), meta)

    def has_complete(self, shard_id: int) -> bool:
        return self.shard_path(shard_id).exists() and self.meta_path(shard_id).exists()

    def discard_from(self, shard_id: int) -> None:
        """Discard completed shards with id >= shard_id. Used on resume to remove data
        produced after the last durable producer state.
        """
        for i in self.list_ids():
            if i >= shard_id:
                self.shard_path(i).unlink(missing_ok=True)
                self.meta_path(i).unlink(missing_ok=True)

    def prune_before(self, protected_shard_id: int):
        # Never delete the shard containing the latest durable consumer checkpoint.
        while self.total_bytes() > self.max_bytes:
            ids = self.list_ids()
            candidates = [i for i in ids if i < protected_shard_id]
            if not candidates:
                break
            i = candidates[0]
            self.shard_path(i).unlink(missing_ok=True)
            self.meta_path(i).unlink(missing_ok=True)


class StreamingTokenProducer:
    """Deterministic producer: source choice is RNG-driven, each source itself is sequential.

    A completed local shard is immutable. Resume never rewrites a completed shard.
    """
    def __init__(self, mixture: dict[str,float], tokenizer_name: str, cache_dir: str | Path,
                 datasets_config='configs/datasets.yaml', seed=123, shard_bytes=512*1024**2,
                 max_cache_bytes=10*1024**3, rank=0, world_size=1, state: dict | None=None):
        self.mixture = {k: float(v) for k,v in mixture.items() if v > 0}
        total=sum(self.mixture.values()); self.mixture={k:v/total for k,v in self.mixture.items()}
        self.languages=list(self.mixture); self.weights=[self.mixture[x] for x in self.languages]
        self.tokenizer_name=tokenizer_name; self.tokenizer=build_tokenizer(tokenizer_name)
        self.eos_id=get_tokenizer_spec(tokenizer_name).eos_id
        self.datasets_config=datasets_config; self.rank=rank; self.world_size=world_size
        self.cache=RollingTokenCache(cache_dir,max_cache_bytes,shard_bytes)
        self.rng=random.Random(seed + rank * 1_000_003)
        self.entries=[]; self.progress={}; self.iterators={}
        for lang in self.languages:
            for idx,e in enumerate(get_language_entries(lang, config_path=datasets_config, section='datasets')):
                key=f'{lang}:{idx}'
                ee=dict(e); ee['_key']=key; ee['_language']=lang
                self.entries.append(ee); self.progress[key]=SourceProgress()
        self.state=ProducerState(source_progress=self.progress, rng_state=self.rng.getstate())
        if state:
            self.restore(state)
        else:
            ids = self.cache.list_ids()
            if ids:
                self.state.shard_id = max(ids) + 1

    def restore(self, state: dict):
        self.state.shard_id=int(state.get('shard_id',0))
        self.state.total_tokens_written=int(state.get('total_tokens_written',0))
        self.state.total_documents_seen=int(state.get('total_documents_seen',0))
        for k,v in state.get('source_progress',{}).items():
            self.progress[k]=SourceProgress(**v)
        import ast
        rs=state.get('rng_state')
        if rs: self.rng.setstate(ast.literal_eval(rs))
        self.state.source_progress=self.progress

    def snapshot(self):
        self.state.source_progress=self.progress; self.state.rng_state=self.rng.getstate()
        return self.state.to_dict()

    def _choose_entry(self):
        lang=self.rng.choices(self.languages, weights=self.weights, k=1)[0]
        candidates=[e for e in self.entries if e['_language']==lang]
        ws=[float(e.get('weight',1.0)) for e in candidates]
        return self.rng.choices(candidates, weights=ws, k=1)[0]

    def _iterator(self, entry):
        key=entry['_key']
        if key not in self.iterators:
            p=self.progress[key]
            # DDP sharding first, then deterministic sequential resume per rank.
            self.iterators[key]=stream_dataset(name=entry['name'], split=entry.get('split','train'),
                text_field=entry.get('text_field','text'), seed=123, shuffle_buffer=0,
                config_name=entry.get('config'), revision=entry.get('revision'),
                worker_id=self.rank, num_workers=self.world_size, skip=p.documents_seen,
                source_name=key)
        return self.iterators[key]

    def produce_one_shard(self):
        shard_id=self.state.shard_id
        tokens=[]; docs_by_source={}
        target_tokens=max(1,self.cache.shard_bytes//BYTES_PER_TOKEN)
        while len(tokens) < target_tokens:
            e=self._choose_entry(); key=e['_key']
            try: doc=next(self._iterator(e))
            except StopIteration:
                self.iterators.pop(key,None); continue
            ids=self.tokenizer.encode(doc.text, disallowed_special=())
            self.progress[key].documents_seen += 1; self.state.total_documents_seen += 1
            if not ids: continue
            ids.append(self.eos_id)
            tokens.extend(ids); self.progress[key].tokens_written += len(ids); self.state.total_tokens_written += len(ids)
            docs_by_source[key]=docs_by_source.get(key,0)+1
        # Advance the producer position before snapshot so metadata means 'next shard to write'.
        self.state.shard_id += 1
        self.cache.write_shard(shard_id, tokens, {'documents_by_source':docs_by_source, 'producer_state_after': self.snapshot()})
        return shard_id, len(tokens)

    def fill_until(self, min_bytes: int, protected_shard_id: int):
        while self.cache.total_bytes() < min_bytes:
            self.produce_one_shard()
            self.cache.prune_before(protected_shard_id)


class TokenShardDataset(IterableDataset):
    def __init__(self, cache_dir, seq_len, start_shard_id=0, start_token_offset=0,
                 rank=0, world_size=1, follow=True):
        self.root=Path(cache_dir); self.seq_len=int(seq_len); self.start_shard_id=int(start_shard_id)
        self.start_token_offset=int(start_token_offset); self.rank=rank; self.world_size=world_size; self.follow=follow

    def _ids(self): return sorted(int(p.stem.split('_')[1]) for p in self.root.glob('shard_*.json'))

    def __iter__(self):
        # rank consumes complete shards round-robin; producer writes rank-local caches by default.
        for sid in self._ids():
            if sid < self.start_shard_id: continue
            p=self.root/f'shard_{sid:08d}.bin'
            arr=np.memmap(p,dtype=DTYPE,mode='r')
            off=self.start_token_offset if sid==self.start_shard_id else 0
            while off+self.seq_len+1 <= len(arr):
                chunk=np.asarray(arr[off:off+self.seq_len+1],dtype=np.int64)
                yield torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:]), sid, off
                off += self.seq_len



def export_resume_anchor(cache_dir: str | Path, output_path: str | Path, shard_id: int, token_offset: int) -> dict:
    """Persist the unread suffix of the active shard.

    This makes a checkpoint portable across Kaggle sessions without exporting the full rolling cache.
    The anchor is raw uint32 tokens and is exact; its worst-case size is one shard per DDP rank.
    """
    cache_dir = Path(cache_dir); output_path = Path(output_path)
    src = cache_dir / f'shard_{int(shard_id):08d}.bin'
    if not src.exists():
        raise FileNotFoundError(f'Cannot export resume anchor; missing {src}')
    total_tokens = src.stat().st_size // BYTES_PER_TOKEN
    if token_offset < 0 or token_offset > total_tokens:
        raise ValueError(f'Invalid token offset {token_offset} for shard with {total_tokens} tokens')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + '.tmp')
    with open(src, 'rb') as r, open(tmp, 'wb') as w:
        r.seek(int(token_offset) * BYTES_PER_TOKEN)
        shutil.copyfileobj(r, w, length=16 * 1024 * 1024)
        w.flush(); os.fsync(w.fileno())
    os.replace(tmp, output_path)
    meta = {
        'version': 1, 'virtual_shard_id': int(shard_id),
        'start_token_offset': int(token_offset),
        'tokens': int(total_tokens - token_offset), 'dtype': 'uint32',
    }
    _atomic_json(output_path.with_suffix(output_path.suffix + '.json'), meta)
    return meta


def restore_resume_anchor(cache_dir: str | Path, anchor_path: str | Path) -> tuple[int, int]:
    cache_dir = Path(cache_dir); anchor_path = Path(anchor_path)
    meta_path = anchor_path.with_suffix(anchor_path.suffix + '.json')
    if not anchor_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f'Missing portable resume anchor: {anchor_path}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    sid = int(meta['virtual_shard_id'])
    dst = cache_dir / f'shard_{sid:08d}.bin'
    dst_meta = cache_dir / f'shard_{sid:08d}.json'
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix('.bin.tmp')
    # Anchor contains only unread tokens, so resume from offset zero after restore.
    with open(anchor_path, 'rb') as r, open(tmp, 'wb') as w:
        shutil.copyfileobj(r, w, length=16 * 1024 * 1024)
        w.flush(); os.fsync(w.fileno())
    os.replace(tmp, dst)
    _atomic_json(dst_meta, {'shard_id': sid, 'tokens': int(meta['tokens']), 'bytes': dst.stat().st_size, 'portable_anchor': True})
    return sid, 0

def inspect_cache(cache_dir):
    c=RollingTokenCache(cache_dir, 1, 1)
    out=[]
    for p in c.complete_shards():
        with open(p,encoding='utf-8') as f: out.append(json.load(f))
    return out
