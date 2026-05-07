from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class AdaptationResult:
    adapted_module: nn.Module
    support_losses: list[float]


class SemanticProjectionHead(nn.Module):
    """Small projection head used by first-order meta-adaptation."""

    def __init__(self, latent_dim: int, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class FirstOrderMetaAdapter:
    """Lightweight MAML-style adaptation utility.

    It adapts a projection module on a support set without modifying the base
    module. This is deliberately simple and robust for repository use; it can be
    replaced by a full higher-order MAML implementation if needed.
    """

    def __init__(self, module: nn.Module, inner_lr: float = 1e-2, steps: int = 5):
        self.module = module
        self.inner_lr = inner_lr
        self.steps = steps

    def adapt(self, z_support: Tensor, y_support: Tensor) -> AdaptationResult:
        adapted = deepcopy(self.module).to(z_support.device)
        opt = torch.optim.SGD(adapted.parameters(), lr=self.inner_lr)
        losses: list[float] = []
        for _ in range(self.steps):
            pred = adapted(z_support)
            loss = F.mse_loss(pred, y_support)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        return AdaptationResult(adapted_module=adapted, support_losses=losses)

    @staticmethod
    def query_loss(module: nn.Module, z_query: Tensor, y_query: Tensor) -> Tensor:
        return F.mse_loss(module(z_query), y_query)
