from __future__ import annotations

import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        target_queries: torch.Tensor,
        context_tokens: torch.Tensor,
        target_mask: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if target_queries.shape[1] == 0:
            return target_queries
        out = self.decoder(
            target_queries,
            context_tokens,
            tgt_key_padding_mask=~target_mask,
            memory_key_padding_mask=~context_mask,
        )
        return self.norm(out) * target_mask.unsqueeze(-1).to(out.dtype)
