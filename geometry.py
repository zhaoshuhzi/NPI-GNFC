from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class GeometryData:
    eigenmodes: Tensor  # [V, K]
    eigenvalues: Tensor  # [K]
    region_names: Optional[list[str]] = None

    @property
    def n_vertices(self) -> int:
        return int(self.eigenmodes.shape[0])

    @property
    def n_eigenmodes(self) -> int:
        return int(self.eigenmodes.shape[1])


def load_geometry_npz(path: str | Path, device: torch.device | str = "cpu") -> GeometryData:
    data = np.load(path, allow_pickle=True)
    if "eigenmodes" not in data or "eigenvalues" not in data:
        raise KeyError("Geometry NPZ must contain 'eigenmodes' [V,K] and 'eigenvalues' [K].")
    eigenmodes = torch.as_tensor(data["eigenmodes"], dtype=torch.float32, device=device)
    eigenvalues = torch.as_tensor(data["eigenvalues"], dtype=torch.float32, device=device)
    region_names = data["region_names"].tolist() if "region_names" in data else None
    return GeometryData(eigenmodes=eigenmodes, eigenvalues=eigenvalues, region_names=region_names)


def synthetic_geometry(n_vertices: int, n_eigenmodes: int, device: torch.device | str = "cpu") -> GeometryData:
    """Create orthonormal synthetic eigenmodes for smoke tests.

    This is not a substitute for HCP-derived cortical eigenmodes. It only makes
    the package runnable without external data.
    """
    x = torch.randn(n_vertices, n_eigenmodes, device=device)
    q, _ = torch.linalg.qr(x, mode="reduced")
    eigenvalues = torch.linspace(1.0, float(n_eigenmodes), n_eigenmodes, device=device)
    return GeometryData(eigenmodes=q, eigenvalues=eigenvalues)


class GeometricFieldProjector(torch.nn.Module):
    """Expand time-varying coefficients onto cortical geometric eigenmodes."""

    def __init__(self, eigenmodes: Tensor, eigenvalues: Tensor):
        super().__init__()
        if eigenmodes.ndim != 2:
            raise ValueError("eigenmodes must have shape [V, K].")
        if eigenvalues.ndim != 1 or eigenvalues.shape[0] != eigenmodes.shape[1]:
            raise ValueError("eigenvalues must have shape [K].")
        self.register_buffer("eigenmodes", eigenmodes.float())
        self.register_buffer("eigenvalues", eigenvalues.float().clamp_min(1e-6))

    def forward(self, coeffs: Tensor) -> Tensor:
        """Map coefficients to a neural field.

        Args:
            coeffs: Tensor with shape [B, T, K].
        Returns:
            field: Tensor with shape [B, T, V].
        """
        return torch.einsum("btk,vk->btv", coeffs, self.eigenmodes)

    def smoothness_loss(self, coeffs: Tensor) -> Tensor:
        """Dirichlet energy in the eigenmode domain.

        For Laplace-Beltrami eigenmodes, the Dirichlet energy is proportional to
        sum_k lambda_k * a_k^2.
        """
        return (coeffs.pow(2) * self.eigenvalues.view(1, 1, -1)).mean()
