from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .geometry import GeometricFieldProjector
from .losses import kl_normal
from .npi import NPIPerturbationDynamics


@dataclass
class EncoderOutput:
    z: Tensor
    mu: Tensor
    logvar: Tensor
    regional_ts: Tensor
    field: Tensor
    coeffs: Tensor
    effective_connectivity: Tensor
    losses: Dict[str, Tensor]


class GeometricNeuralFieldEncoder(nn.Module):
    """Encode EEG into a geometry-constrained source-space neural field.

    Innovation implemented here:
    - NPI estimates directed perturbation propagation in regional time series.
    - The propagation-driven activity is not used as a standalone graph output;
      it drives coefficients of cortical geometric eigenmodes.
    - The model therefore produces a continuous cortical field F(v,t), rather
      than only a discrete effective-connectivity matrix.
    """

    def __init__(
        self,
        eeg_channels: int,
        n_regions: int,
        eigenmodes: Tensor,
        eigenvalues: Tensor,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.eeg_channels = eeg_channels
        self.n_regions = n_regions
        self.n_eigenmodes = int(eigenmodes.shape[1])
        self.latent_dim = latent_dim

        self.sensor_to_region = nn.Sequential(
            nn.Conv1d(eeg_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, n_regions, kernel_size=1),
        )
        self.npi = NPIPerturbationDynamics(n_regions=n_regions)
        self.field_projector = GeometricFieldProjector(eigenmodes=eigenmodes, eigenvalues=eigenvalues)

        # NPI-driven regional dynamics -> geometric eigenmode coefficients.
        self.coeff_head = nn.Sequential(
            nn.Linear(n_regions * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_eigenmodes),
        )
        self.temporal_pool = nn.GRU(
            input_size=self.n_eigenmodes,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.mu_head = nn.Linear(2 * hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(2 * hidden_dim, latent_dim)

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            eps = torch.randn_like(mu)
            return mu + torch.exp(0.5 * logvar) * eps
        return mu

    def forward(self, eeg: Tensor) -> EncoderOutput:
        """Encode EEG.

        Args:
            eeg: Tensor with shape [B, C, T].
        """
        if eeg.ndim != 3:
            raise ValueError("eeg must have shape [B, C, T].")
        regional_ts = self.sensor_to_region(eeg)  # [B, R, T]
        ec, npi_loss = self.npi(regional_ts)

        # NPI propagation is used as a dynamic driver for field coefficients.
        propagated = self.npi.predict_derivative(regional_ts)
        # Pad to original time length.
        propagated = F.pad(propagated, pad=(1, 0), mode="replicate")
        regional_features = torch.cat([regional_ts, propagated], dim=1).transpose(1, 2)  # [B,T,2R]
        coeffs = self.coeff_head(regional_features)  # [B,T,K]
        field = self.field_projector(coeffs)  # [B,T,V]

        encoded_seq, _ = self.temporal_pool(coeffs)
        pooled = encoded_seq.mean(dim=1)
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled).clamp(-8.0, 8.0)
        z = self._reparameterize(mu, logvar)

        losses = {
            "npi": npi_loss,
            "field": self.field_projector.smoothness_loss(coeffs),
            "kl": kl_normal(mu, logvar),
        }
        return EncoderOutput(
            z=z,
            mu=mu,
            logvar=logvar,
            regional_ts=regional_ts,
            field=field,
            coeffs=coeffs,
            effective_connectivity=ec,
            losses=losses,
        )
