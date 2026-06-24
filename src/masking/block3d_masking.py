from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch


@dataclass
class MaskResult:
    context_patch_ids: torch.Tensor
    target_patch_ids: torch.Tensor
    ignored_patch_ids: torch.Tensor
    context_mask: torch.Tensor
    target_mask: torch.Tensor
    target_blocks: List[torch.Tensor]
    stats: Dict[str, float]
    valid: bool = True
    reason: str = "ok"


def _flat_ids(coords: torch.Tensor, grid_size: Sequence[int]) -> torch.Tensor:
    gx, gy, gz = (int(v) for v in grid_size)
    return coords[:, 0].long() * (gy * gz) + coords[:, 1].long() * gz + coords[:, 2].long()


class Block3DMaskGenerator:
    """Sample non-overlapping active target blocks for sparse 3DCal patch tokens."""

    def __init__(
        self,
        *,
        grid_size: Sequence[int],
        num_target_blocks: int = 4,
        target_block_shape: Sequence[int] = (1, 1, 2),
        min_target_patches: int = 1,
        max_target_fraction: float = 0.45,
        min_context_patches: int = 2,
        seed: Optional[int] = None,
        max_attempts: int = 128,
    ) -> None:
        self.grid_size = tuple(int(x) for x in grid_size)
        self.num_target_blocks = int(num_target_blocks)
        self.target_block_shape = tuple(max(1, int(x)) for x in target_block_shape)
        self.min_target_patches = int(min_target_patches)
        self.max_target_fraction = float(max_target_fraction)
        self.min_context_patches = int(min_context_patches)
        self.max_attempts = int(max_attempts)
        self.rng = np.random.default_rng(seed)

    def _block_ids_around(self, center: torch.Tensor) -> torch.Tensor:
        center_np = center.cpu().numpy().astype(int)
        starts = []
        for c, size, grid in zip(center_np, self.target_block_shape, self.grid_size):
            lo = max(0, c - size // 2)
            hi = min(lo, grid - size)
            starts.append(max(0, hi))
        ranges = [torch.arange(starts[i], min(starts[i] + self.target_block_shape[i], self.grid_size[i])) for i in range(3)]
        xx, yy, zz = torch.meshgrid(ranges[0], ranges[1], ranges[2], indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
        return _flat_ids(coords, self.grid_size)

    def __call__(
        self,
        patch_coords: torch.Tensor,
        patch_ids: Optional[torch.Tensor] = None,
        patch_energy: Optional[torch.Tensor] = None,
    ) -> MaskResult:
        patch_coords = patch_coords.cpu().long()
        if patch_ids is None:
            patch_ids = _flat_ids(patch_coords, self.grid_size)
        patch_ids = patch_ids.cpu().long()
        n_active = int(patch_ids.numel())
        if n_active == 0:
            empty = torch.empty((0,), dtype=torch.long)
            stats = {
                "num_active_patches": 0.0,
                "num_context_patches": 0.0,
                "num_target_patches": 0.0,
                "target_fraction": 0.0,
                "overlap": 0.0,
                "valid": 0.0,
            }
            return MaskResult(
                empty,
                empty,
                empty,
                torch.zeros(0, dtype=torch.bool),
                torch.zeros(0, dtype=torch.bool),
                [],
                stats,
                valid=False,
                reason="no_active_patches",
            )

        active_set: Set[int] = set(int(x) for x in patch_ids.tolist())

        if n_active < self.min_context_patches + self.min_target_patches:
            context_mask = torch.ones(n_active, dtype=torch.bool)
            target_mask = torch.zeros(n_active, dtype=torch.bool)
            stats = {
                "num_active_patches": float(n_active),
                "num_context_patches": float(n_active),
                "num_target_patches": 0.0,
                "target_fraction": 0.0,
                "overlap": 0.0,
                "valid": 0.0,
            }
            if patch_energy is not None and patch_energy.numel() == n_active:
                total = float(patch_energy.sum().item())
                stats["context_energy_fraction"] = 1.0 if total > 0 else 0.0
                stats["target_energy_fraction"] = 0.0
            return MaskResult(
                context_patch_ids=patch_ids,
                target_patch_ids=torch.empty((0,), dtype=torch.long),
                ignored_patch_ids=torch.empty((0,), dtype=torch.long),
                context_mask=context_mask,
                target_mask=target_mask,
                target_blocks=[],
                stats=stats,
                valid=False,
                reason="too_few_active_patches",
            )

        max_target_slots = max(0, n_active - self.min_context_patches)
        max_targets = max(self.min_target_patches, int(np.floor(n_active * self.max_target_fraction)))
        max_targets = min(max_targets, max_target_slots)
        target_set: Set[int] = set()
        target_blocks: List[torch.Tensor] = []

        if patch_energy is not None and patch_energy.numel() == n_active and float(patch_energy.sum()) > 0:
            probs = (patch_energy.cpu().float() / patch_energy.sum().float()).numpy()
        else:
            probs = None

        candidate_indices = np.arange(n_active)
        for _ in range(self.num_target_blocks):
            accepted = False
            for _attempt in range(self.max_attempts):
                idx = int(self.rng.choice(candidate_indices, p=probs))
                block_ids = self._block_ids_around(patch_coords[idx])
                active_block = [int(x) for x in block_ids.tolist() if int(x) in active_set and int(x) not in target_set]
                if len(active_block) < self.min_target_patches:
                    continue
                if len(target_set) + len(active_block) > max_targets:
                    active_block = active_block[: max(0, max_targets - len(target_set))]
                if len(active_block) < self.min_target_patches:
                    continue
                target_set.update(active_block)
                target_blocks.append(torch.tensor(active_block, dtype=torch.long))
                accepted = True
                break
            if not accepted and len(target_set) >= self.min_target_patches:
                break

        if len(target_set) == 0 and n_active > self.min_context_patches:
            idx = int(self.rng.choice(candidate_indices, p=probs))
            target_set.add(int(patch_ids[idx]))

        context_set = active_set - target_set
        if len(context_set) < self.min_context_patches and n_active > self.min_context_patches:
            sorted_targets = list(target_set)
            while len(active_set - target_set) < self.min_context_patches and sorted_targets:
                target_set.remove(sorted_targets.pop())
            context_set = active_set - target_set

        target_mask = torch.tensor([int(pid) in target_set for pid in patch_ids.tolist()], dtype=torch.bool)
        context_mask = torch.tensor([int(pid) in context_set for pid in patch_ids.tolist()], dtype=torch.bool)
        overlap = bool((target_mask & context_mask).any().item())
        if overlap:
            raise RuntimeError("Context and target masks overlap; this should be impossible.")

        target_ids = patch_ids[target_mask]
        context_ids = patch_ids[context_mask]
        ignored_ids = patch_ids[~(target_mask | context_mask)]
        valid = target_ids.numel() >= self.min_target_patches and context_ids.numel() >= self.min_context_patches
        stats = {
            "num_active_patches": float(n_active),
            "num_context_patches": float(context_ids.numel()),
            "num_target_patches": float(target_ids.numel()),
            "target_fraction": float(target_ids.numel() / max(1, n_active)),
            "overlap": 0.0,
            "valid": 1.0 if valid else 0.0,
        }
        if patch_energy is not None and patch_energy.numel() == n_active:
            total = float(patch_energy.sum().item())
            stats["context_energy_fraction"] = float(patch_energy[context_mask].sum().item() / total) if total > 0 else 0.0
            stats["target_energy_fraction"] = float(patch_energy[target_mask].sum().item() / total) if total > 0 else 0.0

        return MaskResult(
            context_patch_ids=context_ids,
            target_patch_ids=target_ids,
            ignored_patch_ids=ignored_ids,
            context_mask=context_mask,
            target_mask=target_mask,
            target_blocks=target_blocks,
            stats=stats,
            valid=valid,
            reason="ok" if valid else "insufficient_context_or_target",
        )
