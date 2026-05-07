from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def kl_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL(q(z|x)||N(0,I)) for diagonal Gaussian q."""
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def cosine_embedding_loss(pred: Tensor, target: Tensor) -> Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - (pred * target).sum(dim=-1).mean()


def mse_or_zero(pred: Tensor | None, target: Tensor | None, device: torch.device) -> Tensor:
    if pred is None or target is None:
        return torch.zeros((), device=device)
    return F.mse_loss(pred, target)
