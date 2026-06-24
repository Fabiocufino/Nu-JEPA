from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.sparse_preprocessing import SparseVoxelEvent, summarize_active_patches
from src.masking.block3d_masking import Block3DMaskGenerator
from src.models.nu_jepa import NuJEPA
from src.training.config import load_config
from src.training.losses import jepa_cosine_loss
from src.training.train_nu_jepa import resolve_device, set_global_seed


def _synthetic_event(
    *,
    detector_size: List[int],
    patch_size: List[int],
    active_patches: int,
    voxels_per_patch: int,
) -> SparseVoxelEvent:
    grid = [(int(d) + int(p) - 1) // int(p) for d, p in zip(detector_size, patch_size)]
    coords = []
    patch_volume = int(patch_size[0]) * int(patch_size[1]) * int(patch_size[2])
    voxels_per_patch = min(int(voxels_per_patch), patch_volume)
    target_voxels = active_patches * voxels_per_patch
    for x in range(grid[0]):
        for y in range(grid[1]):
            for z in range(grid[2]):
                base = [x * int(patch_size[0]), y * int(patch_size[1]), z * int(patch_size[2])]
                for offset in range(voxels_per_patch):
                    dx = offset // (int(patch_size[1]) * int(patch_size[2]))
                    rem = offset % (int(patch_size[1]) * int(patch_size[2]))
                    dy = rem // int(patch_size[2])
                    dz = rem % int(patch_size[2])
                    coords.append([base[0] + dx, base[1] + dy, base[2] + dz])
                if len(coords) >= target_voxels:
                    break
            if len(coords) >= target_voxels:
                break
        if len(coords) >= target_voxels:
            break
    coords = coords[:target_voxels]
    coord_tensor = torch.tensor(coords, dtype=torch.long)
    energy = torch.ones((len(coords),), dtype=torch.float32)
    feats = torch.log1p(energy.reshape(-1, 1) * 10.0)
    return SparseVoxelEvent(
        coords=coord_tensor,
        feats=feats,
        energy=energy,
        raw_hits=torch.empty((0, 6)).numpy(),
    )


def _make_batch(cfg: Dict[str, Any], batch_size: int, active_patches: int, voxels_per_patch: int) -> tuple[Dict[str, object], list]:
    detector_size = cfg["data"]["detector_size"]
    patch_size = cfg["data"]["patch_size"]
    events = [
        _synthetic_event(
            detector_size=detector_size,
            patch_size=patch_size,
            active_patches=active_patches,
            voxels_per_patch=voxels_per_patch,
        )
        for _ in range(batch_size)
    ]
    patches = [
        summarize_active_patches(
            event,
            detector_size=detector_size,
            patch_size=patch_size,
            min_voxels_per_patch=cfg["data"]["min_voxels_per_patch"],
        )
        for event in events
    ]
    grid = tuple((int(d) + int(p) - 1) // int(p) for d, p in zip(detector_size, patch_size))
    masker = Block3DMaskGenerator(grid_size=grid, **cfg["masking"])
    masks = [masker(summary.patch_coords, summary.patch_ids, summary.energy) for summary in patches]
    batch = {
        "events": events,
        "patches": patches,
        "paths": ["synthetic"] * batch_size,
        "event_ids": torch.arange(batch_size, dtype=torch.long),
    }
    return batch, masks


def benchmark(args: argparse.Namespace) -> Dict[str, object]:
    cfg = load_config(args.config)
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    model_cfg = dict(cfg["model"])
    if args.embed_dim is not None:
        model_cfg["embed_dim"] = args.embed_dim
    if args.depth is not None:
        model_cfg["depth"] = args.depth
    if args.num_heads is not None:
        model_cfg["num_heads"] = args.num_heads
    if args.no_spconv:
        model_cfg["use_spconv"] = False
    model = NuJEPA(**model_cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    grid = model.grid_size
    max_tokens = int(grid[0] * grid[1] * grid[2])
    active_patches = args.active_patches or max_tokens
    if active_patches > max_tokens:
        raise ValueError(f"active_patches={active_patches} exceeds grid capacity {max_tokens}")
    batch, masks = _make_batch(cfg, args.batch_size, active_patches, args.voxels_per_patch)
    if not all(mask.valid for mask in masks):
        raise RuntimeError("Synthetic benchmark produced an invalid mask; reduce min_context/min_target settings.")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    step_times: List[float] = []
    losses: List[float] = []
    total_steps = args.warmup_steps + args.steps
    for step in range(total_steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(batch, masks)
        loss = jepa_cosine_loss(out["pred"], out["target"], out["target_mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip_norm", 1.0)))
        optimizer.step()
        model.update_target_encoder()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if step >= args.warmup_steps:
            step_times.append(elapsed)
            losses.append(float(loss.detach().cpu().item()))
        print(f"[benchmark] step={step} loss={float(loss.detach().cpu().item()):.5f} seconds={elapsed:.4f}", flush=True)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    result = {
        "device": str(device),
        "use_spconv": bool(model.patch_embed.use_spconv),
        "batch_size": args.batch_size,
        "active_patches_per_event": active_patches,
        "voxels_per_patch": int(args.voxels_per_patch),
        "voxels_per_event": int(active_patches * args.voxels_per_patch),
        "max_patch_tokens_per_event": max_tokens,
        "num_params": int(num_params),
        "trainable_params": int(trainable_params),
        "model_config": model_cfg,
        "mean_step_seconds": float(sum(step_times) / len(step_times)),
        "events_per_second": float(args.batch_size / (sum(step_times) / len(step_times))),
        "peak_memory_mb": peak_memory_mb,
        "loss_first_measured": losses[0],
        "loss_last_measured": losses[-1],
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one NuJEPA training batch at a controlled token count.")
    parser.add_argument("--config", default="configs/nu_jepa_3dcal_12k_robust.yaml")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--active-patches", type=int, default=None)
    parser.add_argument("--voxels-per-patch", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-spconv", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    print(json.dumps(benchmark(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
