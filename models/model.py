import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_CONFIGS = {
    "50M":  dict(vocab_size=50257, context_length=2048, emb_dim=384,  n_heads=6,  n_layers=8,  drop_rate=0.0, qkv_bias=False),
    "100M": dict(vocab_size=50257, context_length=2048, emb_dim=768,  n_heads=12, n_layers=12, drop_rate=0.0, qkv_bias=False),
    "1.2B": dict(vocab_size=50257, context_length=2048, emb_dim=2048, n_heads=16, n_layers=24, drop_rate=0.0, qkv_bias=False),
}

class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.shift = nn.Parameter(torch.zeros(d))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.scale * (x-mean) / torch.sqrt(var+self.eps) + self.shift

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg["emb_dim"], cfg["n_heads"]
        assert d % h == 0
        self.h, self.head_dim = h, d//h
        self.q = nn.Linear(d,d,bias=cfg["qkv_bias"])
        self.k = nn.Linear(d,d,bias=cfg["qkv_bias"])
        self.v = nn.Linear(d,d,bias=cfg["qkv_bias"])
        self.proj = nn.Linear(d,d)
        self.drop = nn.Dropout(cfg["drop_rate"])
    def forward(self,x):
        b,s,d=x.shape
        q=self.q(x).view(b,s,self.h,self.head_dim).transpose(1,2)
        k=self.k(x).view(b,s,self.h,self.head_dim).transpose(1,2)
        v=self.v(x).view(b,s,self.h,self.head_dim).transpose(1,2)
        if hasattr(F,"scaled_dot_product_attention"):
            y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.drop.p if self.training else 0.0)
        else:
            score=q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
            mask=torch.triu(torch.ones(s,s,device=x.device,dtype=torch.bool),1)
            score=score.masked_fill(mask,float("-inf"))
            y=self.drop(torch.softmax(score,-1))@v
        return self.proj(y.transpose(1,2).contiguous().view(b,s,d))

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        d=cfg["emb_dim"]
        self.net=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.norm1=LayerNorm(cfg["emb_dim"]); self.att=MultiHeadAttention(cfg)
        self.norm2=LayerNorm(cfg["emb_dim"]); self.ff=FeedForward(cfg)
    def forward(self,x):
        x=x+self.att(self.norm1(x))
        return x+self.ff(self.norm2(x))

class GPTModel(nn.Module):
    def __init__(self,cfg,tie_weights=True):
        super().__init__()
        self.cfg=cfg
        self.tok_emb=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb=nn.Dropout(cfg["drop_rate"])
        self.blocks=nn.Sequential(*(TransformerBlock(cfg) for _ in range(cfg["n_layers"])))
        self.final_norm=LayerNorm(cfg["emb_dim"])
        self.out_head=nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)
        if tie_weights: self.out_head.weight=self.tok_emb.weight
    def forward(self,idx,targets=None):
        b,s=idx.shape
        if s>self.cfg["context_length"]: raise ValueError("sequence too long")
        pos=torch.arange(s,device=idx.device)
        x=self.drop_emb(self.tok_emb(idx)+self.pos_emb(pos))
        x=self.blocks(x); logits=self.out_head(self.final_norm(x))
        if targets is None: return logits
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1))
        return logits,loss

def build_model(name): return GPTModel(MODEL_CONFIGS[name])
