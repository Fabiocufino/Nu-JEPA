from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .sparse_preprocessing import (
    SparseVoxelEvent,
    extract_reco_hits,
    hits_to_sparse_event,
    inspect_npz,
    summarize_active_patches,
)


class FASERCalNPZDataset(Dataset):
    """Map-style dataset for sparse FASERCal `.npz` events."""

    def __init__(
        self,
        root: str | Sequence[str],
        *,
        reco_hits_key: str = "reco_hits",
        recursive: bool = False,
        max_events: Optional[int] = None,
        detector_size: Sequence[int] = (48, 48, 200),
        patch_size: Sequence[int] = (12, 12, 10),
        coord_mode: str = "discrete",
        energy_scale: float = 10.0,
        energy_transform: str = "log1p",
        min_voxels_per_patch: int = 1,
        debug_first_n: int = 0,
    ) -> None:
        if isinstance(root, (str, Path)):
            self.roots = [Path(root)]
        else:
            self.roots = [Path(path) for path in root]
        self.root = self.roots[0] if len(self.roots) == 1 else None
        self.reco_hits_key = reco_hits_key
        self.detector_size = tuple(int(x) for x in detector_size)
        self.patch_size = tuple(int(x) for x in patch_size)
        self.coord_mode = coord_mode
        self.energy_scale = float(energy_scale)
        self.energy_transform = energy_transform
        self.min_voxels_per_patch = int(min_voxels_per_patch)
        self.debug_first_n = int(debug_first_n)

        pattern = "**/*.npz" if recursive else "*.npz"
        self.files = sorted(str(p) for root_path in self.roots for p in root_path.glob(pattern))
        if max_events is not None:
            self.files = self.files[: int(max_events)]

    def __len__(self) -> int:
        return len(self.files)

    def describe_first(self, n: int = 3) -> List[Dict[str, Any]]:
        return [inspect_npz(path, self.reco_hits_key) for path in self.files[:n]]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        with np.load(path, allow_pickle=True) as npz:
            hits = extract_reco_hits(npz, self.reco_hits_key)
            event_id = int(npz["event_id"].item()) if "event_id" in npz and np.asarray(npz["event_id"]).size == 1 else idx
        event = hits_to_sparse_event(
            hits,
            detector_size=self.detector_size,
            coord_mode=self.coord_mode,
            energy_scale=self.energy_scale,
            energy_transform=self.energy_transform,
            path=path,
            event_id=event_id,
        )
        patches = summarize_active_patches(
            event,
            detector_size=self.detector_size,
            patch_size=self.patch_size,
            min_voxels_per_patch=self.min_voxels_per_patch,
        )
        if idx < self.debug_first_n:
            print(
                f"[FASERCalNPZDataset] idx={idx} voxels={event.coords.shape[0]} "
                f"patches={patches.patch_coords.shape[0]} energy={event.energy.sum().item():.4g} path={path}"
            )
        return {"event": event, "patches": patches, "path": path, "event_id": event_id}


def faser_sparse_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "events": [item["event"] for item in batch],
        "patches": [item["patches"] for item in batch],
        "paths": [item["path"] for item in batch],
        "event_ids": torch.tensor([item["event_id"] for item in batch], dtype=torch.long),
    }


def list_npz_files(root: str, recursive: bool = False) -> List[str]:
    root_path = Path(root)
    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted(str(p) for p in root_path.glob(pattern))
