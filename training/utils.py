import math, random, numpy as np, torch

def seed_everything(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cosine_lr(step,total_steps,base_lr,warmup_steps):
    if step<warmup_steps: return base_lr*(step+1)/max(1,warmup_steps)
    p=min(1.0,(step-warmup_steps)/max(1,total_steps-warmup_steps))
    return 0.5*base_lr*(1+math.cos(math.pi*p))
