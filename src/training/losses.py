from __future__ import annotations

import torch
import torch.nn.functional as F


def jepa_cosine_loss(pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    if target_mask.sum() == 0:
        return pred.sum() * 0.0
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    loss = 1.0 - (pred_n * target_n).sum(dim=-1)
    return loss[target_mask].mean()
