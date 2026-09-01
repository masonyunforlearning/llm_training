from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.norm.LayerNorm import LayerNorm
from models.transformerblock.TransformerBlock import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = dict(cfg)

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg.get("drop_rate", 0.0))

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        if cfg.get("tie_weights", True):
            self.out_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.cfg["context_length"]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context_length "
                f"{self.cfg['context_length']}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.drop_emb(x)

        for block in self.trf_blocks:
            x = block(x)

        logits = self.out_head(self.final_norm(x))

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return logits, loss
