from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to read configs. Install pyyaml in this environment.") from exc

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except Exception:
    SummaryWriter = None
    HAS_TENSORBOARD = False

from src.data.faser_npz_dataset import FASERCalNPZDataset, faser_sparse_collate
from src.masking.block3d_masking import Block3DMaskGenerator, MaskResult
from src.models.nu_jepa import NuJEPA
from src.training.checkpointing import save_checkpoint
from src.training.config import load_config
from src.training.logging_utils import CSVLogger, log_tensorboard_scalars
from src.training.losses import jepa_cosine_loss


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def init_distributed(requested_device: str) -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
    else:
        device = resolve_device(requested_device)
    return {
        "enabled": enabled,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
        "device": device,
    }


def cleanup_distributed(info: Dict[str, Any]) -> None:
    if info["enabled"] and dist.is_initialized():
        dist.destroy_process_group()


def print_main(info: Dict[str, Any], *args: object, **kwargs: object) -> None:
    if info["is_main"]:
        print(*args, **kwargs)


def _model_for_state(model: torch.nn.Module) -> NuJEPA:
    return model.module if hasattr(model, "module") else model


def _dist_sum(value: float, device: torch.device, info: Dict[str, Any]) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if info["enabled"]:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _dist_max(value: float, device: torch.device, info: Dict[str, Any]) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if info["enabled"]:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _dist_min_int(value: int, device: torch.device, info: Dict[str, Any]) -> int:
    tensor = torch.tensor(int(value), dtype=torch.long, device=device)
    if info["enabled"]:
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def build_mask_generator(cfg: Dict[str, Any]) -> Block3DMaskGenerator:
    data_cfg = cfg["data"]
    detector = data_cfg["detector_size"]
    patch = data_cfg["patch_size"]
    grid = tuple((int(d) + int(p) - 1) // int(p) for d, p in zip(detector, patch))
    return Block3DMaskGenerator(grid_size=grid, **cfg["masking"])


def build_masks(batch: Dict[str, Any], generator: Block3DMaskGenerator) -> List[MaskResult]:
    return [
        generator(summary.patch_coords, summary.patch_ids, summary.energy)
        for summary in batch["patches"]
    ]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def train(config_path: str, max_steps: Optional[int] = None) -> None:
    cfg = load_config(config_path)
    seed = int(cfg["training"].get("seed", cfg.get("masking", {}).get("seed", 0)))
    dist_info = init_distributed(cfg["training"].get("device", "auto"))
    rank = int(dist_info["rank"])
    world_size = int(dist_info["world_size"])
    set_global_seed(seed + rank)
    device = dist_info["device"]
    data_args = dict(cfg["data"])
    dataset_path = data_args.pop("dataset_path")
    dataset = FASERCalNPZDataset(dataset_path, **data_args, debug_first_n=3 if dist_info["is_main"] else 0)
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No .npz files found under {cfg['data']['dataset_path']}. "
            "Place the event files there or update configs/nu_jepa_3dcal_baseline.yaml."
        )
    print_main(dist_info, f"[train] config={config_path}", flush=True)
    print_main(
        dist_info,
        f"[train] seed={seed} distributed={dist_info['enabled']} "
        f"world_size={world_size} rank={rank}",
        flush=True,
    )
    print_main(dist_info, f"[train] device={device} cuda_available={torch.cuda.is_available()}", flush=True)
    if device.type == "cuda":
        print_main(dist_info, f"[train] cuda_device={torch.cuda.get_device_name(device)}", flush=True)
    print_main(
        dist_info,
        f"[train] dataset_path={cfg['data']['dataset_path']} events={len(dataset)} "
        f"batch_size={cfg['training']['batch_size']} workers={cfg['training']['num_workers']}",
        flush=True,
    )
    if dist_info["is_main"]:
        for idx, info in enumerate(dataset.describe_first(3)):
            summary = info.get("reco_hits_summary", {})
            print(
                f"[train] inspect[{idx}] shape={summary.get('shape')} "
                f"coord_min={summary.get('coord_min')} coord_max={summary.get('coord_max')} "
                f"energy_sum={summary.get('energy_sum')} path={info.get('path')}",
                flush=True,
            )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=False,
    ) if dist_info["enabled"] else None
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=faser_sparse_collate,
        drop_last=False,
        generator=loader_generator if sampler is None else None,
        worker_init_fn=_seed_worker,
        pin_memory=bool(cfg["training"].get("pin_memory", device.type == "cuda")),
    )
    model = NuJEPA(**cfg["model"]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_main(dist_info, f"[train] model_params={num_params:,} trainable={num_trainable:,}", flush=True)
    if dist_info["enabled"]:
        model = DistributedDataParallel(
            model,
            device_ids=[int(dist_info["local_rank"])] if device.type == "cuda" else None,
            output_device=int(dist_info["local_rank"]) if device.type == "cuda" else None,
        )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    mask_generator = build_mask_generator(cfg)

    log_dir = Path(cfg["training"]["log_dir"])
    logger: Optional[CSVLogger] = None
    writer = None
    if dist_info["is_main"]:
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "config_resolved.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        with (log_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:
            handle.write(str(_model_for_state(model)))
            handle.write("\n")
            handle.write(f"\nmodel_params: {num_params}\ntrainable_params: {num_trainable}\n")
        with (log_dir / "run_info.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config_path": config_path,
                    "dataset_path": cfg["data"]["dataset_path"],
                    "num_events": len(dataset),
                    "device": str(device),
                    "distributed": bool(dist_info["enabled"]),
                    "world_size": world_size,
                    "seed": seed,
                    "model_params": num_params,
                    "trainable_params": num_trainable,
                    "tensorboard_available": HAS_TENSORBOARD,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        logger = CSVLogger(
            str(log_dir / "train_metrics.csv"),
            [
                "time",
                "epoch",
                "step",
                "loss",
                "grad_norm",
                "lr",
                "num_context_patches",
                "num_target_patches",
                "context_energy_fraction",
                "target_energy_fraction",
                "batch_events",
                "valid_events",
                "skipped_events",
                "trainable_event_fraction",
                "seconds_per_step",
                "events_per_second",
                "gpu_memory_allocated_mb",
                "gpu_memory_reserved_mb",
                "gpu_memory_peak_allocated_mb",
            ],
        )
        if bool(cfg["training"].get("tensorboard", True)):
            if not HAS_TENSORBOARD:
                print("[train] TensorBoard is not available; skipping SummaryWriter.", flush=True)
            else:
                tb_dir = Path(cfg["training"].get("tensorboard_dir", str(log_dir / "tensorboard")))
                writer = SummaryWriter(str(tb_dir))
    step = 0
    skipped_batches = 0
    skipped_events = 0
    printed_first_batch = False
    epoch = 0
    try:
        for epoch in range(int(cfg["training"]["epochs"])):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                masks = build_masks(batch, mask_generator)
                raw_batch_events = len(masks)
                batch, masks = _filter_trainable_samples(batch, masks)
                skipped_now = raw_batch_events - len(masks)
                local_has_valid = 1 if len(masks) > 0 else 0
                all_ranks_have_valid = _dist_min_int(local_has_valid, device, dist_info)
                global_batch_events = int(_dist_sum(raw_batch_events, device, dist_info))
                global_valid_events = int(_dist_sum(len(masks), device, dist_info))
                if all_ranks_have_valid == 0:
                    skipped_batches += 1
                    skipped_events += raw_batch_events
                    print_main(
                        dist_info,
                        f"[train] skip epoch={epoch} step={step} reason=rank_without_valid_context_target "
                        f"global_batch_events={global_batch_events} global_valid_events={global_valid_events}",
                        flush=True,
                    )
                    continue
                skipped_events += skipped_now
                global_skipped_events = int(_dist_sum(skipped_now, device, dist_info))
                if not printed_first_batch:
                    print_main(
                        dist_info,
                        f"[train] first_batch raw_events={raw_batch_events} skipped={skipped_now} "
                        f"{_batch_debug_summary(batch, masks)}",
                        flush=True,
                    )
                    printed_first_batch = True
                step_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                out = model(batch, masks)
                loss = jepa_cosine_loss(out["pred"], out["target"], out["target_mask"])
                if not torch.isfinite(loss).item():
                    raise RuntimeError(
                        f"Non-finite loss at epoch={epoch} step={step}: {float(loss.detach().cpu().item())}"
                    )
                loss.backward()
                grad_clip = float(cfg["training"].get("grad_clip_norm", 0.0) or 0.0)
                grad_norm = None
                if grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                        error_if_nonfinite=True,
                    )
                else:
                    grad_norm = _grad_norm(model)
                    if not math.isfinite(grad_norm):
                        raise RuntimeError(f"Non-finite gradient norm at epoch={epoch} step={step}: {grad_norm}")
                optimizer.step()
                _model_for_state(model).update_target_encoder()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds_per_step = time.perf_counter() - step_start
                global_seconds_per_step = _dist_max(seconds_per_step, device, dist_info)

                log_this_step = step % int(cfg["training"]["log_every"]) == 0
                is_final_step = max_steps is not None and step + 1 >= max_steps
                stats = _mean_mask_stats(masks)
                grad_norm_value = float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                loss_value = float(loss.detach().cpu().item())
                global_loss = _dist_sum(loss_value, device, dist_info) / world_size
                global_grad_norm = _dist_sum(grad_norm_value, device, dist_info) / world_size
                global_stats = {
                    key: _dist_sum(value, device, dist_info) / world_size
                    for key, value in stats.items()
                }
                lr = float(optimizer.param_groups[0]["lr"])
                trainable_fraction = global_valid_events / max(1, global_batch_events)
                events_per_second = global_valid_events / max(global_seconds_per_step, 1e-9)
                gpu_allocated = 0.0
                gpu_reserved = 0.0
                gpu_peak = 0.0
                if device.type == "cuda":
                    gpu_allocated = _dist_max(torch.cuda.memory_allocated(device) / 1024**2, device, dist_info)
                    gpu_reserved = _dist_max(torch.cuda.memory_reserved(device) / 1024**2, device, dist_info)
                    gpu_peak = _dist_max(torch.cuda.max_memory_allocated(device) / 1024**2, device, dist_info)
                if (log_this_step or is_final_step) and dist_info["is_main"]:
                    row = {
                        "time": time.time(),
                        "epoch": epoch,
                        "step": step,
                        "loss": global_loss,
                        "grad_norm": global_grad_norm,
                        "lr": lr,
                        **global_stats,
                        "batch_events": global_batch_events,
                        "valid_events": global_valid_events,
                        "skipped_events": global_skipped_events,
                        "trainable_event_fraction": trainable_fraction,
                        "seconds_per_step": global_seconds_per_step,
                        "events_per_second": events_per_second,
                        "gpu_memory_allocated_mb": gpu_allocated,
                        "gpu_memory_reserved_mb": gpu_reserved,
                        "gpu_memory_peak_allocated_mb": gpu_peak,
                    }
                    if logger is not None:
                        logger.log(row)
                    if writer is not None:
                        log_tensorboard_scalars(writer, row, step)
                    print(
                        f"epoch={epoch} step={step} loss={global_loss:.5f} "
                        f"grad_norm={global_grad_norm:.5f} lr={lr:.3g} "
                        f"events_per_second={events_per_second:.2f} stats={global_stats}",
                        flush=True,
                    )

                if dist_info["is_main"] and step > 0 and step % int(cfg["training"]["checkpoint_every"]) == 0:
                    save_checkpoint(
                        cfg["training"]["checkpoint_dir"],
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        step=step,
                        config=cfg,
                    )
                step += 1
                if max_steps is not None and step >= max_steps:
                    return
    finally:
        if logger is not None:
            logger.close()
        if writer is not None:
            writer.flush()
            writer.close()
        print_main(
            dist_info,
            f"[train] finished_steps={step} skipped_batches={skipped_batches} "
            f"skipped_events_local_rank0={skipped_events}",
            flush=True,
        )
        if dist_info["is_main"] and step > 0:
            save_checkpoint(
                cfg["training"]["checkpoint_dir"],
                model=model,
                optimizer=optimizer,
                epoch=max(0, epoch),
                step=step,
                config=cfg,
            )
        cleanup_distributed(dist_info)


def _mean_mask_stats(masks: List[MaskResult]) -> Dict[str, float]:
    keys = [
        "num_context_patches",
        "num_target_patches",
        "context_energy_fraction",
        "target_energy_fraction",
    ]
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(m.stats.get(key, 0.0)) for m in masks]
        out[key] = sum(vals) / max(1, len(vals))
    return out


def _filter_trainable_samples(batch: Dict[str, Any], masks: List[MaskResult]) -> tuple[Dict[str, Any], List[MaskResult]]:
    keep = [
        idx
        for idx, mask in enumerate(masks)
        if mask.valid and mask.target_patch_ids.numel() > 0 and mask.context_patch_ids.numel() > 0
    ]
    if len(keep) == len(masks):
        return batch, masks
    filtered = {
        "events": [batch["events"][idx] for idx in keep],
        "patches": [batch["patches"][idx] for idx in keep],
        "paths": [batch["paths"][idx] for idx in keep],
        "event_ids": batch["event_ids"][keep],
    }
    return filtered, [masks[idx] for idx in keep]


def _batch_debug_summary(batch: Dict[str, Any], masks: List[MaskResult]) -> Dict[str, object]:
    voxel_counts = [int(event.coords.shape[0]) for event in batch["events"]]
    patch_counts = [int(summary.patch_ids.numel()) for summary in batch["patches"]]
    target_counts = [int(mask.target_patch_ids.numel()) for mask in masks]
    context_counts = [int(mask.context_patch_ids.numel()) for mask in masks]
    overlaps = [
        len(set(mask.target_patch_ids.tolist()) & set(mask.context_patch_ids.tolist()))
        for mask in masks
    ]
    return {
        "event_ids": batch["event_ids"].tolist(),
        "voxels_min_max": [min(voxel_counts), max(voxel_counts)],
        "patches_min_max": [min(patch_counts), max(patch_counts)],
        "target_patches_min_max": [min(target_counts), max(target_counts)],
        "context_patches_min_max": [min(context_counts), max(context_counts)],
        "overlap_max": max(overlaps),
    }


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        norm = float(param.grad.detach().data.norm(2).cpu().item())
        total += norm * norm
    return math.sqrt(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nu_jepa_3dcal_baseline.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    train(args.config, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
