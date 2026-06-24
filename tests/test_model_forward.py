import numpy as np
import torch

from src.data.sparse_preprocessing import hits_to_sparse_event, summarize_active_patches
from src.masking.block3d_masking import Block3DMaskGenerator
from src.models.nu_jepa import NuJEPA
from src.training.losses import jepa_cosine_loss


def test_nu_jepa_forward_on_synthetic_sparse_event():
    hits = np.array(
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
    event = hits_to_sparse_event(hits, detector_size=(48, 48, 200))
    patches = summarize_active_patches(event, detector_size=(48, 48, 200), patch_size=(12, 12, 10))
    masker = Block3DMaskGenerator(
        grid_size=(4, 4, 20),
        num_target_blocks=1,
        target_block_shape=(1, 1, 2),
        seed=1,
    )
    mask = masker(patches.patch_coords, patches.patch_ids, patches.energy)
    model = NuJEPA(
        detector_size=(48, 48, 200),
        patch_size=(12, 12, 10),
        embed_dim=32,
        depth=1,
        num_heads=4,
        use_spconv=False,
    )
    out = model({"events": [event], "patches": [patches]}, [mask])
    loss = jepa_cosine_loss(out["pred"], out["target"], out["target_mask"])
    assert torch.isfinite(loss)
    assert out["pred"].shape == out["target"].shape

    encoded = model.encode_events({"events": [event], "patches": [patches]})
    assert encoded["event_latents"].shape == (1, 32)
    assert encoded["patch_latents"].shape[0] == 1
    assert encoded["patch_mask"].sum().item() == patches.patch_ids.numel()
    assert torch.isfinite(encoded["event_latents"]).all()
