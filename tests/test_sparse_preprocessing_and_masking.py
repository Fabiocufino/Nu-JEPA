import numpy as np
import torch

from src.data.sparse_preprocessing import hits_to_sparse_event, summarize_active_patches
from src.masking.block3d_masking import Block3DMaskGenerator


def _synthetic_hits():
    return np.array(
        [
            [1, 1, 1, 0, 0.4, 0],
            [2, 1, 2, 0, 0.3, 0],
            [14, 15, 21, 1, 1.2, 0],
            [15, 15, 22, 1, 0.8, 0],
            [30, 20, 80, 4, 2.0, 0],
            [31, 20, 81, 4, 1.5, 0],
        ],
        dtype=np.float32,
    )


def test_sparse_hits_to_active_patches():
    event = hits_to_sparse_event(_synthetic_hits(), detector_size=(48, 48, 200))
    patches = summarize_active_patches(event, detector_size=(48, 48, 200), patch_size=(12, 12, 10))
    assert event.coords.shape == (6, 3)
    assert patches.patch_ids.numel() == 3
    assert torch.isclose(patches.energy.sum(), event.energy.sum())


def test_block_mask_has_no_overlap_and_nonempty_target():
    event = hits_to_sparse_event(_synthetic_hits(), detector_size=(48, 48, 200))
    patches = summarize_active_patches(event, detector_size=(48, 48, 200), patch_size=(12, 12, 10))
    masker = Block3DMaskGenerator(
        grid_size=(4, 4, 20),
        num_target_blocks=2,
        target_block_shape=(1, 1, 2),
        seed=3,
    )
    mask = masker(patches.patch_coords, patches.patch_ids, patches.energy)
    assert mask.target_patch_ids.numel() > 0
    assert mask.context_patch_ids.numel() > 0
    assert set(mask.target_patch_ids.tolist()).isdisjoint(set(mask.context_patch_ids.tolist()))
    assert mask.stats["overlap"] == 0.0


def test_block_mask_marks_events_without_enough_active_patches_untrainable():
    masker = Block3DMaskGenerator(
        grid_size=(4, 4, 20),
        num_target_blocks=2,
        target_block_shape=(1, 1, 2),
        min_target_patches=1,
        min_context_patches=2,
        seed=3,
    )

    empty = masker(torch.empty((0, 3), dtype=torch.long))
    assert not empty.valid
    assert empty.reason == "no_active_patches"
    assert empty.context_patch_ids.numel() == 0
    assert empty.target_patch_ids.numel() == 0

    one_patch = masker(torch.tensor([[0, 0, 0]], dtype=torch.long))
    assert not one_patch.valid
    assert one_patch.reason == "too_few_active_patches"
    assert one_patch.context_patch_ids.numel() == 1
    assert one_patch.target_patch_ids.numel() == 0

    two_patches = masker(torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.long))
    assert not two_patches.valid
    assert two_patches.context_patch_ids.numel() == 2
    assert two_patches.target_patch_ids.numel() == 0

    three_patches = masker(torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=torch.long))
    assert three_patches.valid
    assert three_patches.context_patch_ids.numel() >= 2
    assert three_patches.target_patch_ids.numel() >= 1
