from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import torch

from src.data.sparse_preprocessing import SparseVoxelEvent, PatchSummary, split_event_by_patch_mask
from src.masking.block3d_masking import MaskResult


def _plotly():
    try:
        import plotly.graph_objects as go
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Plotly is required for visualization notebooks. Install plotly first.") from exc
    return go


def _detector_view_coords(coords: torch.Tensor):
    return coords[:, 2], coords[:, 0], coords[:, 1]


def _detector_view_scene() -> Dict[str, object]:
    return dict(
        xaxis_title="z (detector depth)",
        yaxis_title="x",
        zaxis_title="y",
        aspectmode="data",
        camera=dict(
            eye=dict(x=0.0, y=-2.2, z=0.7),
            up=dict(x=0.0, y=0.0, z=1.0),
        ),
    )


def scatter_event(
    event: SparseVoxelEvent,
    *,
    title: str = "FASERCal reco_hits",
    mask: Optional[torch.Tensor] = None,
    color: str = "energy",
):
    go = _plotly()
    coords = event.coords.cpu()
    energy = event.energy.cpu()
    if mask is not None:
        mask = mask.cpu().bool()
        coords = coords[mask]
        energy = energy[mask]
    marker_color = energy if color == "energy" else color
    plot_x, plot_y, plot_z = _detector_view_coords(coords)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=plot_x,
                y=plot_y,
                z=plot_z,
                mode="markers",
                marker=dict(size=3, color=marker_color, colorscale="Viridis", opacity=0.8),
                name=title,
            )
        ]
    )
    fig.update_layout(title=title, scene=_detector_view_scene())
    return fig


def plot_masked_event(
    event: SparseVoxelEvent,
    mask: MaskResult,
    *,
    detector_size: Sequence[int],
    patch_size: Sequence[int],
    title: str = "3DCal context/target mask",
):
    go = _plotly()
    split = split_event_by_patch_mask(
        event,
        detector_size=detector_size,
        patch_size=patch_size,
        target_patch_ids=mask.target_patch_ids.tolist(),
    )
    coords = event.coords.cpu()
    context = split["context_voxel_mask"].bool()
    target = split["target_voxel_mask"].bool()
    ignored = split["ignored_voxel_mask"].bool()

    traces = []
    for name, keep, color, size in [
        ("context", context, "rgb(35, 116, 171)", 3),
        ("target", target, "rgb(214, 73, 51)", 5),
        ("ignored", ignored, "rgb(140, 140, 140)", 2),
    ]:
        if keep.sum() == 0:
            continue
        c = coords[keep]
        plot_x, plot_y, plot_z = _detector_view_coords(c)
        traces.append(
            go.Scatter3d(
                x=plot_x,
                y=plot_y,
                z=plot_z,
                mode="markers",
                marker=dict(size=size, color=color, opacity=0.82),
                name=name,
            )
        )

    fig = go.Figure(data=traces)
    for block in mask.target_blocks:
        for patch_id in block.tolist():
            fig.add_trace(_patch_wireframe_trace(patch_id, detector_size, patch_size))
    fig.update_layout(title=title, scene=_detector_view_scene())
    return fig


def _patch_wireframe_trace(patch_id: int, detector_size: Sequence[int], patch_size: Sequence[int]):
    go = _plotly()
    gx = (int(detector_size[0]) + int(patch_size[0]) - 1) // int(patch_size[0])
    gy = (int(detector_size[1]) + int(patch_size[1]) - 1) // int(patch_size[1])
    gz = (int(detector_size[2]) + int(patch_size[2]) - 1) // int(patch_size[2])
    x = patch_id // (gy * gz)
    y = (patch_id // gz) % gy
    z = patch_id % gz
    x0, y0, z0 = x * patch_size[0], y * patch_size[1], z * patch_size[2]
    x1, y1, z1 = min(x0 + patch_size[0], detector_size[0]), min(y0 + patch_size[1], detector_size[1]), min(z0 + patch_size[2], detector_size[2])
    phys_xs = [x0, x1, x1, x0, x0, None, x0, x1, x1, x0, x0, None, x0, x0, None, x1, x1, None, x1, x1, None, x0, x0]
    phys_ys = [y0, y0, y1, y1, y0, None, y0, y0, y1, y1, y0, None, y0, y0, None, y0, y0, None, y1, y1, None, y1, y1]
    phys_zs = [z0, z0, z0, z0, z0, None, z1, z1, z1, z1, z1, None, z0, z1, None, z0, z1, None, z0, z1, None, z0, z1]
    return go.Scatter3d(
        x=phys_zs,
        y=phys_xs,
        z=phys_ys,
        mode="lines",
        line=dict(color="rgb(214, 73, 51)", width=4),
        name="target patch boundary",
        showlegend=False,
    )


def mask_sanity_table(event: SparseVoxelEvent, patches: PatchSummary, mask: MaskResult) -> Dict[str, float]:
    total_energy = float(event.energy.sum().item())
    context_patch_set = set(int(x) for x in mask.context_patch_ids.tolist())
    target_patch_set = set(int(x) for x in mask.target_patch_ids.tolist())
    overlap = context_patch_set & target_patch_set
    return {
        "active_voxels": float(event.coords.shape[0]),
        "active_patches": float(patches.patch_ids.numel()),
        "context_patches": float(mask.context_patch_ids.numel()),
        "target_patches": float(mask.target_patch_ids.numel()),
        "overlap_patches": float(len(overlap)),
        "total_energy": total_energy,
        "context_energy_fraction": float(mask.stats.get("context_energy_fraction", 0.0)),
        "target_energy_fraction": float(mask.stats.get("target_energy_fraction", 0.0)),
    }
