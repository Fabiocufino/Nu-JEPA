from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from src.data.sparse_preprocessing import SparseVoxelEvent, PatchSummary, as_tuple3

try:
    from spconv.pytorch import SparseConv3d, SparseConvTensor

    HAS_SPCONV = True
except Exception:
    SparseConv3d = None
    SparseConvTensor = None
    HAS_SPCONV = False


def _grid_size(detector_size: Sequence[int], patch_size: Sequence[int]) -> tuple[int, int, int]:
    d = torch.tensor(as_tuple3(detector_size), dtype=torch.long)
    p = torch.tensor(as_tuple3(patch_size), dtype=torch.long)
    g = torch.div(d + p - 1, p, rounding_mode="floor")
    return int(g[0]), int(g[1]), int(g[2])


class SparsePatchEmbed(nn.Module):
    """Sparse 3D patch embedding with an spconv path and a pure-PyTorch fallback."""

    def __init__(
        self,
        *,
        detector_size: Sequence[int] = (48, 48, 200),
        patch_size: Sequence[int] = (12, 12, 10),
        in_channels: int = 1,
        embed_dim: int = 128,
        use_spconv: bool = True,
    ) -> None:
        super().__init__()
        self.detector_size = as_tuple3(detector_size)
        self.patch_size = as_tuple3(patch_size)
        self.grid_size = _grid_size(self.detector_size, self.patch_size)
        self.embed_dim = int(embed_dim)
        self.use_spconv = bool(use_spconv and HAS_SPCONV)
        if use_spconv and not HAS_SPCONV:
            print("[SparsePatchEmbed] spconv is not available; using aggregated PyTorch patch embedding.")
        if self.use_spconv:
            self.sparse_conv = SparseConv3d(
                in_channels,
                embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                padding=0,
                bias=True,
            )
        self.fallback = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        if self.use_spconv:
            for param in self.fallback.parameters():
                param.requires_grad = False

    def _pack_by_event(
        self,
        batch_size: int,
        features: torch.Tensor,
        indices: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        gx, gy, gz = self.grid_size
        out: List[Dict[str, torch.Tensor]] = []
        for b in range(batch_size):
            keep = indices[:, 0].long() == b
            coords = indices[keep, 1:4].long()
            tokens = features[keep]
            if coords.numel() == 0:
                ids = torch.empty((0,), dtype=torch.long, device=features.device)
            else:
                ids = coords[:, 0] * (gy * gz) + coords[:, 1] * gz + coords[:, 2]
                order = torch.argsort(ids)
                ids = ids[order]
                coords = coords[order]
                tokens = tokens[order]
            out.append({"tokens": tokens, "patch_coords": coords, "patch_ids": ids})
        return out

    def _forward_spconv(self, events: List[SparseVoxelEvent]) -> List[Dict[str, torch.Tensor]]:
        device = next(self.parameters()).device
        indices = []
        features = []
        for b, event in enumerate(events):
            if event.coords.numel() == 0:
                continue
            batch_col = torch.full((event.coords.shape[0], 1), b, dtype=torch.long)
            indices.append(torch.cat([batch_col, event.coords.cpu().long()], dim=1))
            features.append(event.feats.cpu().float())
        if not indices:
            empty_f = torch.empty((0, self.embed_dim), device=device)
            empty_i = torch.empty((0, 4), dtype=torch.long, device=device)
            return self._pack_by_event(len(events), empty_f, empty_i)
        idx = torch.cat(indices, dim=0).to(device=device, dtype=torch.int32)
        feat = torch.cat(features, dim=0).to(device=device)
        tensor = SparseConvTensor(
            features=feat,
            indices=idx,
            spatial_shape=list(self.detector_size),
            batch_size=len(events),
        )
        embedded = self.sparse_conv(tensor)
        return self._pack_by_event(len(events), embedded.features, embedded.indices.long())

    def _forward_fallback(self, patches: List[PatchSummary]) -> List[Dict[str, torch.Tensor]]:
        device = next(self.parameters()).device
        out: List[Dict[str, torch.Tensor]] = []
        for summary in patches:
            features = summary.features.to(device)
            tokens = self.fallback(features) if features.numel() else torch.empty((0, self.fallback[-1].out_features), device=device)
            out.append(
                {
                    "tokens": tokens,
                    "patch_coords": summary.patch_coords.to(device),
                    "patch_ids": summary.patch_ids.to(device),
                }
            )
        return out

    def forward(
        self,
        events: List[SparseVoxelEvent],
        patches: List[PatchSummary],
    ) -> List[Dict[str, torch.Tensor]]:
        if self.use_spconv:
            return self._forward_spconv(events)
        return self._forward_fallback(patches)
