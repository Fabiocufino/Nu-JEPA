from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.masking.block3d_masking import MaskResult

from .context_encoder import ContextEncoder, PatchPositionalEncoding
from .predictor import JEPAPredictor
from .sparse_patch_embed import SparsePatchEmbed, _grid_size
from .target_encoder import TargetEncoder


def _pad(items: List[torch.Tensor], trailing_shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((x.shape[0] for x in items), default=0)
    out = torch.zeros((len(items), max_len, *trailing_shape), dtype=dtype, device=device)
    mask = torch.zeros((len(items), max_len), dtype=torch.bool, device=device)
    for i, value in enumerate(items):
        n = value.shape[0]
        if n:
            out[i, :n] = value.to(device=device, dtype=dtype)
            mask[i, :n] = True
    return out, mask


class NuJEPA(nn.Module):
    def __init__(
        self,
        *,
        detector_size: Sequence[int] = (48, 48, 200),
        patch_size: Sequence[int] = (12, 12, 10),
        in_channels: int = 1,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_spconv: bool = True,
        ema_decay: float = 0.996,
    ) -> None:
        super().__init__()
        self.grid_size = _grid_size(detector_size, patch_size)
        self.ema_decay = float(ema_decay)
        self.patch_embed = SparsePatchEmbed(
            detector_size=detector_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_spconv=use_spconv,
        )
        self.target_patch_embed = copy.deepcopy(self.patch_embed)
        self.pos_embed = PatchPositionalEncoding(embed_dim, self.grid_size)
        self.context_encoder = ContextEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.target_encoder = TargetEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            depth=max(1, depth // 2),
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self._init_target_encoder()

    def _init_target_encoder(self) -> None:
        self.target_patch_embed.load_state_dict(self.patch_embed.state_dict())
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_patch_embed.parameters():
            param.requires_grad = False
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for target, source in zip(self.target_patch_embed.parameters(), self.patch_embed.parameters()):
            target.data.mul_(self.ema_decay).add_(source.data, alpha=1.0 - self.ema_decay)
        for target, source in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target.data.mul_(self.ema_decay).add_(source.data, alpha=1.0 - self.ema_decay)

    def _select_tokens(
        self,
        embedded: List[Dict[str, torch.Tensor]],
        masks: List[MaskResult],
        which: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_list: List[torch.Tensor] = []
        coord_list: List[torch.Tensor] = []
        device = next(self.parameters()).device
        for item, mask in zip(embedded, masks):
            ids = item["patch_ids"].detach().cpu()
            wanted = mask.context_patch_ids if which == "context" else mask.target_patch_ids
            wanted_set = set(int(x) for x in wanted.detach().cpu().tolist())
            keep = torch.tensor([int(x) in wanted_set for x in ids.tolist()], dtype=torch.bool, device=item["tokens"].device)
            token_list.append(item["tokens"][keep])
            coord_list.append(item["patch_coords"][keep])
        tokens, valid = _pad(token_list, (embedded[0]["tokens"].shape[-1],), torch.float32, device)
        coords, _ = _pad(coord_list, (3,), torch.long, device)
        return tokens, coords, valid

    def forward(self, batch: Dict[str, object], masks: List[MaskResult]) -> Dict[str, torch.Tensor]:
        embedded = self.patch_embed(batch["events"], batch["patches"])
        if not embedded:
            raise RuntimeError("Empty batch")
        context_tokens, context_coords, context_mask = self._select_tokens(embedded, masks, "context")
        context_tokens = context_tokens + self.pos_embed(context_coords)

        context_latent = self.context_encoder(context_tokens, context_mask)
        with torch.no_grad():
            target_embedded = self.target_patch_embed(batch["events"], batch["patches"])
            target_tokens, target_coords, target_mask = self._select_tokens(target_embedded, masks, "target")
            target_pos = self.pos_embed(target_coords)
            target_latent = self.target_encoder(target_tokens + target_pos, target_mask)
            target_latent = F.layer_norm(target_latent, target_latent.shape[-1:])
        target_pos = self.pos_embed(target_coords)
        pred = self.predictor(target_pos, context_latent, target_mask, context_mask)
        return {
            "pred": pred,
            "target": target_latent.detach(),
            "target_mask": target_mask,
            "context_mask": context_mask,
        }

    @torch.no_grad()
    def encode_events(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        """Encode all active patch tokens and return mean-pooled event embeddings."""
        embedded = self.patch_embed(batch["events"], batch["patches"])
        if not embedded:
            raise RuntimeError("Empty batch")
        device = next(self.parameters()).device
        token_list = [item["tokens"] for item in embedded]
        coord_list = [item["patch_coords"] for item in embedded]
        tokens, valid_mask = _pad(token_list, (embedded[0]["tokens"].shape[-1],), torch.float32, device)
        coords, _ = _pad(coord_list, (3,), torch.long, device)
        tokens = tokens + self.pos_embed(coords)
        patch_latents = self.context_encoder(tokens, valid_mask)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(patch_latents.dtype)
        event_latents = (patch_latents * valid_mask.unsqueeze(-1).to(patch_latents.dtype)).sum(dim=1) / denom
        return {
            "event_latents": event_latents,
            "patch_latents": patch_latents,
            "patch_mask": valid_mask,
        }
