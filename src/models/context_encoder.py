from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PatchPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, grid_size: Sequence[int]) -> None:
        super().__init__()
        self.register_buffer("grid_size", torch.tensor(grid_size, dtype=torch.float32), persistent=False)
        self.net = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, patch_coords: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(self.grid_size.to(patch_coords.device) - 1.0, min=1.0)
        return self.net(patch_coords.float() / denom.view(1, 1, 3))


class ContextEncoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == 0:
            return tokens
        x = self.blocks(tokens, src_key_padding_mask=~valid_mask)
        return self.norm(x) * valid_mask.unsqueeze(-1).to(tokens.dtype)
