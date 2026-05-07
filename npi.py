from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class NPIPerturbationDynamics(nn.Module):
    """Neural Perturbational Inference-style effective connectivity module.

    This module estimates a directed effective-connectivity matrix and uses it
    to predict first-order temporal changes in regional EEG activity. It is a
    compact implementation of the perturbation-propagation idea used by the
    NPI+GNFC encoder.
    """

    def __init__(self, n_regions: int, init_scale: float = 0.02, learn_delay: bool = False):
        super().__init__()
        self.n_regions = n_regions
        self.effective_connectivity = nn.Parameter(init_scale * torch.randn(n_regions, n_regions))
        self.log_tau = nn.Parameter(torch.zeros(n_regions)) if learn_delay else None

    def normalized_connectivity(self) -> Tensor:
        # Remove self-loop dominance and stabilize propagation.
        w = self.effective_connectivity
        w = w - torch.diag_embed(torch.diagonal(w))
        return torch.tanh(w) / max(1, self.n_regions ** 0.5)

    def predict_derivative(self, regional_ts: Tensor) -> Tensor:
        """Predict dx/dt from regional time series.

        Args:
            regional_ts: [B, R, T]
        Returns:
            predicted derivative: [B, R, T-1]
        """
        x_prev = regional_ts[:, :, :-1]
        w = self.normalized_connectivity()
        propagated = torch.einsum("ij,bjt->bit", w, torch.tanh(x_prev))
        if self.log_tau is not None:
            tau = F.softplus(self.log_tau).view(1, -1, 1).clamp_min(1e-3)
            propagated = propagated / tau
        return propagated

    def npi_loss(self, regional_ts: Tensor) -> Tensor:
        dx = regional_ts[:, :, 1:] - regional_ts[:, :, :-1]
        dx_hat = self.predict_derivative(regional_ts)
        return F.mse_loss(dx_hat, dx)

    def forward(self, regional_ts: Tensor) -> tuple[Tensor, Tensor]:
        loss = self.npi_loss(regional_ts)
        return self.normalized_connectivity(), loss
