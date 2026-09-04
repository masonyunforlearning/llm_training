from __future__ import annotations

import argparse, json, os, time, signal
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data.cache_pipeline import (StreamingTokenProducer, TokenShardDataset,
                                 export_resume_anchor, restore_resume_anchor)
from data.pipeline import normalize_mixture
from models.model import build_model
from tokenizer.factory import get_tokenizer_spec
from training.ddp import init_distributed, is_main, barrier, cleanup
from training.checkpoint import save_checkpoint, load_checkpoint
from training.output_manager import atomic_json, save_model_safetensors, retain_milestones, write_manifest, copy_tokenizer_artifact
from training.utils import seed_everything, count_parameters


def parse_mix(s: str) -> dict[str, float]:
    return normalize_mixture({k.strip(): float(v) for k,v in (x.split(':',1) for x in s.split(','))})


def sidecar_path(d: Path, rank: int) -> Path: return d / 'resume' / f'latest.rank{rank}.data.json'
def anchor_path(d: Path, rank: int) -> Path: return d / 'resume' / 'anchors' / f'rank{rank}.anchor.bin'

def save_data_state(output: Path, rank: int, sid: int, off: int, producer: StreamingTokenProducer, cache_dir: Path, portable: bool):
    anchor = None
    if portable:
        anchor = export_resume_anchor(cache_dir, anchor_path(output, rank), sid, off)
    atomic_json(sidecar_path(output, rank), {'version': 3, 'shard_id': int(sid), 'token_offset': int(off), 'producer': producer.snapshot(), 'anchor': anchor})

def load_data_state(output: Path, rank: int) -> dict:
    p = sidecar_path(output, rank)
    if not p.exists(): raise FileNotFoundError(f'Missing DDP data sidecar: {p}')
    return json.loads(p.read_text(encoding='utf-8'))

p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--model', default='small'); p.add_argument('--mixture', default='ko:0.3,en:0.7')
p.add_argument('--train-tokens', type=int, required=True); p.add_argument('--seq-len', type=int, default=2048)
p.add_argument('--batch-size', type=int, default=1); p.add_argument('--grad-accum', type=int, default=8)
p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--tokenizer', default='cl100k_base')
p.add_argument('--datasets-config', default='configs/datasets.yaml'); p.add_argument('--cache-dir', default='/kaggle/working/token_cache')
p.add_argument('--cache-gb', type=float, default=10); p.add_argument('--shard-mb', type=int, default=128)
p.add_argument('--prefill-shards', type=int, default=4); p.add_argument('--output-dir', default='/kaggle/working/output')
p.add_argument('--checkpoint-interval', type=int, default=100); p.add_argument('--export-interval', type=int, default=500)
p.add_argument('--milestone-tokens', type=int, default=0); p.add_argument('--keep-milestones', type=int, default=2)
p.add_argument('--resume', action='store_true'); p.add_argument('--portable-resume', action=argparse.BooleanOptionalAction, default=True)
p.add_argument('--seed', type=int, default=123)
a=p.parse_args()

info=init_distributed(); seed_everything(a.seed+info.rank)
device=torch.device('cuda', info.local_rank) if torch.cuda.is_available() else torch.device('cpu')
if min(a.cache_gb,a.shard_mb) <= 0 or a.prefill_shards < 1: raise ValueError('invalid cache/shard/prefill values')

mix=parse_mix(a.mixture); output=Path(a.output_dir); cache_dir=Path(a.cache_dir)/f'rank_{info.rank:02d}'
spec=get_tokenizer_spec(a.tokenizer)
model=build_model(a.model,vocab_size=spec.vocab_size,context_length=a.seq_len).to(device); raw=model
opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(0.9,0.95),weight_decay=0.1)
scaler=torch.amp.GradScaler('cuda',enabled=True) if device.type=='cuda' else None
step=tokens_seen=start_shard=start_offset=0; resume_data=None
resume_pt=output/'resume'/'latest.pt'
if a.resume:
    if not resume_pt.exists(): raise FileNotFoundError(f'No resume checkpoint: {resume_pt}')
    payload=load_checkpoint(resume_pt,model=model,optimizer=opt,scaler=scaler,map_location=device,restore_rng=False)
    resume_data=load_data_state(output,info.rank)
    start_shard=int(resume_data['shard_id']); start_offset=int(resume_data['token_offset'])
    step=int(payload.get('state',{}).get('update_step',0)); tokens_seen=int(payload.get('state',{}).get('tokens_seen',0))

producer=StreamingTokenProducer(mix,a.tokenizer,cache_dir,a.datasets_config,a.seed,a.shard_mb*1024**2,int(a.cache_gb*1024**3),info.rank,info.world_size,resume_data.get('producer') if resume_data else None)
if a.resume:
    producer.cache.discard_from(producer.state.shard_id)
    if not producer.cache.has_complete(start_shard):
        if a.portable_resume:
            start_shard,start_offset=restore_resume_anchor(cache_dir,anchor_path(output,info.rank))
        else:
            raise RuntimeError('Resume shard missing and portable resume is disabled')

min_prefill=min(a.prefill_shards*producer.cache.shard_bytes,producer.cache.max_bytes)
producer.fill_until(min_prefill, protected_shard_id=start_shard)
if info.enabled: model=DDP(model,device_ids=[info.local_rank],output_device=info.local_rank,find_unused_parameters=False)

def make_it(sid,off):
    return iter(DataLoader(TokenShardDataset(cache_dir,a.seq_len,sid,off),batch_size=a.batch_size,num_workers=0,pin_memory=device.type=='cuda'))
it=make_it(start_shard,start_offset); last_shard,last_offset=start_shard,start_offset
stop_requested=False
def _stop(sig,frame):
    global stop_requested; stop_requested=True
signal.signal(signal.SIGINT,_stop); signal.signal(signal.SIGTERM,_stop)

def durable_checkpoint(final=False):
    save_data_state(output,info.rank,last_shard,last_offset,producer,cache_dir,a.portable_resume)
    barrier(info)
    if is_main(info):
        save_checkpoint(resume_pt,model=raw,optimizer=opt,scaler=scaler,state={'update_step':step,'tokens_seen':tokens_seen},model_config=raw.cfg,tokenizer_name=a.tokenizer)
        copy_tokenizer_artifact(a.tokenizer,output/'tokenizer')
        if final or step % a.export_interval == 0:
            save_model_safetensors(output/'model'/'latest.safetensors',raw)
        if a.milestone_tokens and tokens_seen and tokens_seen % a.milestone_tokens < a.batch_size*a.seq_len*a.grad_accum*info.world_size:
            mp=output/'milestones'/f'model_tokens_{tokens_seen:012d}.safetensors'; save_model_safetensors(mp,raw); retain_milestones(output/'milestones',a.keep_milestones)
        write_manifest(output,run={'tokens_seen':tokens_seen,'update_step':step,'world_size':info.world_size,'portable_resume':a.portable_resume},files={'resume_latest':resume_pt,'model_latest':output/'model'/'latest.safetensors','tokenizer':output/'tokenizer'/'tokenizer_manifest.json'})
    barrier(info)
    producer.cache.prune_before(last_shard)

if is_main(info):
    print(f'DDP world={info.world_size} params={count_parameters(raw):,} cache={a.cache_gb}GB shard={a.shard_mb}MB portable_resume={a.portable_resume}')
    if a.resume: print(f'RESUME step={step} tokens={tokens_seen:,} shard={start_shard} offset={start_offset}')
model.train(); opt.zero_grad(set_to_none=True)
try:
    while tokens_seen < a.train_tokens and not stop_requested:
        loss_sum=0.; t0=time.perf_counter()
        for _ in range(a.grad_accum):
            try: x,y,sids,offs=next(it)
            except StopIteration:
                producer.fill_until(min_prefill,protected_shard_id=last_shard); it=make_it(last_shard,last_offset); x,y,sids,offs=next(it)
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                _,loss=model(x,y); loss=loss/a.grad_accum
            (scaler.scale(loss).backward() if scaler else loss.backward())
            loss_sum+=float(loss.detach())*a.grad_accum; last_shard=int(sids[-1]); last_offset=int(offs[-1])+a.seq_len; tokens_seen+=x.numel()*info.world_size
        if scaler: scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if scaler: scaler.step(opt); scaler.update()
        else: opt.step()
        opt.zero_grad(set_to_none=True); step+=1
        if is_main(info):
            dt=max(time.perf_counter()-t0,1e-9); print(f'[UPDATE {step:07d}] tokens={tokens_seen:,} loss={loss_sum/a.grad_accum:.4f} tok/s={(a.batch_size*a.seq_len*a.grad_accum*info.world_size)/dt:.1f}')
        if step % a.checkpoint_interval==0: durable_checkpoint()
finally:
    if step>0:
        try: durable_checkpoint(final=True)
        except Exception as e:
            if is_main(info): print(f'WARNING final checkpoint failed: {e}')
    barrier(info); cleanup(info)
