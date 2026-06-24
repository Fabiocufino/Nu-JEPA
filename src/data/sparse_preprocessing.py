from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class SparseVoxelEvent:
    coords: torch.Tensor
    feats: torch.Tensor
    energy: torch.Tensor
    raw_hits: np.ndarray
    path: Optional[str] = None
    event_id: Optional[int] = None


@dataclass
class PatchSummary:
    patch_coords: torch.Tensor
    patch_ids: torch.Tensor
    features: torch.Tensor
    energy: torch.Tensor
    voxel_counts: torch.Tensor
    voxel_patch_ids: torch.Tensor


def as_tuple3(value: Sequence[int] | torch.Tensor | np.ndarray) -> Tuple[int, int, int]:
    if len(value) != 3:
        raise ValueError(f"Expected a length-3 value, got {value}")
    return int(value[0]), int(value[1]), int(value[2])


def _find_numeric_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist())
    if arr.ndim == 1:
        if arr.size == 0:
            return arr.reshape(0, 0).astype(np.float32)
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected reco_hits to be 2D, got shape {arr.shape}")
    if arr.shape[1] < 5:
        raise ValueError(
            "Expected reco_hits columns [x, y, z, module, energy, ...]; "
            f"got shape {arr.shape}"
        )
    return arr.astype(np.float32, copy=False)


def extract_reco_hits(npz: Any, key: str = "reco_hits") -> np.ndarray:
    if key not in npz:
        keys = list(getattr(npz, "files", []))
        raise KeyError(f"Missing key {key!r}. Available keys: {keys}")
    return _find_numeric_array(npz[key])


def inspect_npz(path: str, reco_hits_key: str = "reco_hits") -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        out: Dict[str, Any] = {"path": path, "keys": list(npz.files)}
        for key in npz.files:
            arr = np.asarray(npz[key])
            out[key] = {"shape": arr.shape, "dtype": str(arr.dtype)}
        if reco_hits_key in npz:
            hits = extract_reco_hits(npz, reco_hits_key)
            out["reco_hits_summary"] = {
                "shape": hits.shape,
                "coord_min": hits[:, :3].min(axis=0).tolist() if hits.size else [None] * 3,
                "coord_max": hits[:, :3].max(axis=0).tolist() if hits.size else [None] * 3,
                "energy_min": float(hits[:, 4].min()) if hits.size else None,
                "energy_max": float(hits[:, 4].max()) if hits.size else None,
                "energy_sum": float(hits[:, 4].sum()) if hits.size else 0.0,
            }
        return out


def hits_to_sparse_event(
    hits: np.ndarray,
    *,
    detector_size: Sequence[int] = (48, 48, 200),
    coord_mode: str = "discrete",
    energy_scale: float = 10.0,
    energy_transform: str = "log1p",
    clip: bool = True,
    path: Optional[str] = None,
    event_id: Optional[int] = None,
) -> SparseVoxelEvent:
    detector = torch.tensor(as_tuple3(detector_size), dtype=torch.long)
    if hits.size == 0:
        return SparseVoxelEvent(
            coords=torch.empty((0, 3), dtype=torch.long),
            feats=torch.empty((0, 1), dtype=torch.float32),
            energy=torch.empty((0,), dtype=torch.float32),
            raw_hits=hits,
            path=path,
            event_id=event_id,
        )

    coords_np = hits[:, :3]
    if coord_mode == "auto":
        integer_like = np.allclose(coords_np, np.round(coords_np), atol=1e-4)
        non_negative = bool((coords_np >= -1e-4).all())
        coord_mode = "discrete" if integer_like and non_negative else "physical"

    if coord_mode == "discrete":
        coords = torch.from_numpy(np.rint(coords_np).astype(np.int64, copy=False))
    elif coord_mode == "physical":
        mins = torch.from_numpy(coords_np.min(axis=0)).float()
        maxs = torch.from_numpy(coords_np.max(axis=0)).float()
        denom = torch.clamp(maxs - mins, min=1e-6)
        scaled = (torch.from_numpy(coords_np).float() - mins) / denom
        coords = torch.floor(scaled * detector.float()).long()
    else:
        raise ValueError(f"Unknown coord_mode={coord_mode!r}")

    energy = torch.from_numpy(hits[:, 4].astype(np.float32, copy=False))
    keep = energy > 0
    if clip:
        in_bounds = ((coords >= 0) & (coords < detector.view(1, 3))).all(dim=1)
        keep = keep & in_bounds
    coords = coords[keep]
    energy = energy[keep]

    feats = energy.reshape(-1, 1) * float(energy_scale)
    if energy_transform == "log1p":
        feats = torch.log1p(torch.clamp(feats, min=0.0))
    elif energy_transform == "sqrt":
        feats = torch.sqrt(torch.clamp(feats, min=0.0))
    elif energy_transform in ("identity", None):
        pass
    else:
        raise ValueError(f"Unknown energy_transform={energy_transform!r}")

    return SparseVoxelEvent(
        coords=coords.long(),
        feats=feats.float(),
        energy=energy.float(),
        raw_hits=hits,
        path=path,
        event_id=event_id,
    )


def voxel_to_patch_ids(
    coords: torch.Tensor,
    *,
    detector_size: Sequence[int],
    patch_size: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    detector = torch.tensor(as_tuple3(detector_size), dtype=torch.long, device=coords.device)
    patch = torch.tensor(as_tuple3(patch_size), dtype=torch.long, device=coords.device)
    grid = torch.div(detector + patch - 1, patch, rounding_mode="floor")
    patch_coords = torch.div(coords.long(), patch.view(1, 3), rounding_mode="floor")
    patch_coords = torch.minimum(patch_coords, (grid - 1).view(1, 3))
    patch_ids = patch_coords[:, 0] * (grid[1] * grid[2]) + patch_coords[:, 1] * grid[2] + patch_coords[:, 2]
    return patch_coords.long(), patch_ids.long(), (int(grid[0]), int(grid[1]), int(grid[2]))


def summarize_active_patches(
    event: SparseVoxelEvent,
    *,
    detector_size: Sequence[int],
    patch_size: Sequence[int],
    min_voxels_per_patch: int = 1,
) -> PatchSummary:
    if event.coords.numel() == 0:
        return PatchSummary(
            patch_coords=torch.empty((0, 3), dtype=torch.long),
            patch_ids=torch.empty((0,), dtype=torch.long),
            features=torch.empty((0, 4), dtype=torch.float32),
            energy=torch.empty((0,), dtype=torch.float32),
            voxel_counts=torch.empty((0,), dtype=torch.long),
            voxel_patch_ids=torch.empty((0,), dtype=torch.long),
        )

    voxel_patch_coords, voxel_patch_ids, grid = voxel_to_patch_ids(
        event.coords, detector_size=detector_size, patch_size=patch_size
    )
    unique_ids, inverse = torch.unique(voxel_patch_ids, sorted=True, return_inverse=True)
    n_patches = unique_ids.numel()
    counts = torch.bincount(inverse, minlength=n_patches)
    energy = torch.zeros(n_patches, dtype=event.energy.dtype, device=event.energy.device)
    energy.scatter_add_(0, inverse, event.energy)
    feat_sum = torch.zeros(n_patches, dtype=event.feats.dtype, device=event.feats.device)
    feat_sum.scatter_add_(0, inverse, event.feats[:, 0])
    feat_mean = feat_sum / counts.clamp_min(1).to(feat_sum.dtype)
    feat_max = torch.full((n_patches,), -torch.inf, dtype=event.feats.dtype, device=event.feats.device)
    if hasattr(feat_max, "scatter_reduce_"):
        feat_max.scatter_reduce_(0, inverse, event.feats[:, 0], reduce="amax", include_self=True)
    else:
        for idx in range(n_patches):
            feat_max[idx] = event.feats[inverse == idx, 0].max()
    feat_count = torch.log1p(counts.float())
    features = torch.stack([feat_sum, feat_mean, feat_max, feat_count], dim=1).float()

    gx, gy, gz = grid
    patch_coords = torch.stack(
        [
            unique_ids // (gy * gz),
            (unique_ids // gz) % gy,
            unique_ids % gz,
        ],
        dim=1,
    ).long()
    keep = counts >= int(min_voxels_per_patch)
    return PatchSummary(
        patch_coords=patch_coords[keep].cpu(),
        patch_ids=unique_ids[keep].cpu(),
        features=features[keep].cpu(),
        energy=energy[keep].cpu(),
        voxel_counts=counts[keep].cpu(),
        voxel_patch_ids=voxel_patch_ids.cpu(),
    )


def split_event_by_patch_mask(
    event: SparseVoxelEvent,
    *,
    detector_size: Sequence[int],
    patch_size: Sequence[int],
    target_patch_ids: Iterable[int],
) -> Dict[str, torch.Tensor]:
    target_ids = torch.as_tensor(list(target_patch_ids), dtype=torch.long)
    _, voxel_patch_ids, _ = voxel_to_patch_ids(event.coords, detector_size=detector_size, patch_size=patch_size)
    is_target = torch.isin(voxel_patch_ids.cpu(), target_ids).to(event.coords.device)
    is_context = ~is_target
    return {
        "context_voxel_mask": is_context.cpu(),
        "target_voxel_mask": is_target.cpu(),
        "ignored_voxel_mask": torch.zeros_like(is_target, dtype=torch.bool).cpu(),
        "voxel_patch_ids": voxel_patch_ids.cpu(),
    }
