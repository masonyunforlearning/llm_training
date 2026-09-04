from __future__ import annotations
import os, torch
import torch.distributed as dist
from dataclasses import dataclass

@dataclass
class DistInfo:
    enabled: bool=False; rank:int=0; world_size:int=1; local_rank:int=0

def init_distributed() -> DistInfo:
    ws=int(os.environ.get('WORLD_SIZE','1'))
    if ws<=1: return DistInfo()
    local=int(os.environ.get('LOCAL_RANK','0')); rank=int(os.environ.get('RANK','0'))
    torch.cuda.set_device(local)
    dist.init_process_group(backend='nccl', init_method='env://')
    return DistInfo(True,rank,ws,local)

def is_main(info): return info.rank==0

def barrier(info):
    if info.enabled: dist.barrier()

def cleanup(info):
    if info.enabled and dist.is_initialized(): dist.destroy_process_group()
