from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=True)
        self.dropout_p = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape

        queries = self.W_query(x).view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = self.W_key(x).view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = self.W_value(x).view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            context = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = queries @ keys.transpose(-2, -1)
            scores = scores / math.sqrt(self.head_dim)
            mask = torch.triu(
                torch.ones(
                    num_tokens,
                    num_tokens,
                    device=x.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                weights = F.dropout(weights, p=self.dropout_p)
            context = weights @ values

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, num_tokens, self.d_out)
        return self.out_proj(context)
