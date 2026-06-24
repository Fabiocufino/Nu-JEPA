from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(
    checkpoint_dir: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: Dict[str, Any],
) -> str:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    ckpt_path = path / f"nu_jepa_step_{step:08d}.pt"
    torch.save(
        {
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
        },
        ckpt_path,
    )
    latest = path / "latest.pt"
    torch.save({"path": str(ckpt_path), "step": step, "epoch": epoch}, latest)
    return str(ckpt_path)
