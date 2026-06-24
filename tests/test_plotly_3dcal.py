import numpy as np
import pytest
import torch

pytest.importorskip("plotly")

from src.data.sparse_preprocessing import SparseVoxelEvent
from src.masking.block3d_masking import MaskResult
from src.visualization.plotly_3dcal import plot_masked_event, scatter_event


def _event():
    coords = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 20],
            [13, 2, 11],
        ],
        dtype=torch.long,
    )
    energy = torch.tensor([0.4, 0.8, 1.2], dtype=torch.float32)
    return SparseVoxelEvent(
        coords=coords,
        feats=energy.reshape(-1, 1),
        energy=energy,
        raw_hits=np.empty((0, 6), dtype=np.float32),
    )


def test_scatter_event_displays_detector_depth_left_to_right():
    fig = scatter_event(_event())
    trace = fig.data[0]

    assert list(trace.x) == [3, 20, 11]
    assert list(trace.y) == [1, 4, 13]
    assert list(trace.z) == [2, 5, 2]
    assert fig.layout.scene.xaxis.title.text == "z (detector depth)"


def test_masked_event_wireframe_displays_detector_depth_left_to_right():
    target_patch_id = 81  # physical patch x=[12,24], y=[0,12], z=[10,20]
    mask = MaskResult(
        context_patch_ids=torch.tensor([0, 20], dtype=torch.long),
        target_patch_ids=torch.tensor([target_patch_id], dtype=torch.long),
        ignored_patch_ids=torch.empty((0,), dtype=torch.long),
        context_mask=torch.tensor([True, True, False]),
        target_mask=torch.tensor([False, False, True]),
        target_blocks=[torch.tensor([target_patch_id], dtype=torch.long)],
        stats={},
    )

    fig = plot_masked_event(
        _event(),
        mask,
        detector_size=(48, 48, 200),
        patch_size=(12, 12, 10),
    )
    wireframe = fig.data[-1]

    assert {v for v in wireframe.x if v is not None} == {10, 20}
    assert {v for v in wireframe.y if v is not None} == {12, 24}
    assert {v for v in wireframe.z if v is not None} == {0, 12}
    assert fig.layout.scene.xaxis.title.text == "z (detector depth)"
