"""Download/tokenize/cache/resume test. No model training."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from data.cache_pipeline import StreamingTokenProducer, TokenShardDataset, inspect_cache
from data.pipeline import normalize_mixture


def parse_mix(s):
    return normalize_mixture({k.strip():float(v) for k,v in (x.split(':',1) for x in s.split(','))})

p=argparse.ArgumentParser()
p.add_argument('--mixture',default='ko:0.3,en:0.7')
p.add_argument('--cache-dir',default='token_cache_test')
p.add_argument('--datasets-config',default='configs/datasets.yaml')
p.add_argument('--tokenizer',default='cl100k_base')
p.add_argument('--shard-mb',type=int,default=64)
p.add_argument('--cache-gb',type=float,default=1.0)
p.add_argument('--fill-shards',type=int,default=2)
p.add_argument('--seq-len',type=int,default=2048)
p.add_argument('--resume-state',default=None)
a=p.parse_args(); mix=parse_mix(a.mixture)
state=None
if a.resume_state and Path(a.resume_state).exists(): state=json.loads(Path(a.resume_state).read_text())
prod=StreamingTokenProducer(mix,a.tokenizer,a.cache_dir,a.datasets_config,123,a.shard_mb*1024**2,int(a.cache_gb*1024**3),0,1,state)
for _ in range(a.fill_shards):
    sid,n=prod.produce_one_shard(); print(f'produced shard={sid} tokens={n:,}')
Path(a.cache_dir,'producer_state.json').write_text(json.dumps(prod.snapshot(),ensure_ascii=False,indent=2))
print('CACHE:')
for x in inspect_cache(a.cache_dir): print(x['shard_id'], x['tokens'], x['bytes'])
ds=TokenShardDataset(a.cache_dir,a.seq_len)
it=iter(ds)
for i in range(3):
    x,y,sid,off=next(it); print(f'sample={i} shard={sid} offset={off} x={tuple(x.shape)} y={tuple(y.shape)}')
print('state:', Path(a.cache_dir,'producer_state.json'))
