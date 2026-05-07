"""
NPI + Geometric Neural Field Encoder
====================================

A GitHub-ready PyTorch implementation inspired by the schematic:

    EEG perturbation responses  -->  NPI encoder
    MRI cortical geometry       -->  geometric neural field constraints
    output                      -->  language-network field / boundary logits

This module intentionally avoids heavy geometric-learning dependencies
(torch-geometric, torch-scatter, etc.) so it can be dropped into a repository
and run with only PyTorch.

Author: <your name>
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def add_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Add self-loops to a graph edge index.

    Args:
        edge_index: Long tensor with shape [2, E].
        num_nodes: Number of graph vertices.

    Returns:
        Long tensor with shape [2, E + num_nodes].
    """
    device = edge_index.device
    loops = torch.arange(num_nodes, device=device, dtype=torch.long)
    loops = torch.stack([loops, loops], dim=0)
    return torch.cat([edge_index.long(), loops], dim=1)


def edges_from_faces(faces: Tensor, make_undirected: bool = True) -> Tensor:
    """Build an edge index from triangular mesh faces.

    Args:
        faces: Long tensor with shape [F, 3].
        make_undirected: If True, include both directions for every edge.

    Returns:
        edge_index: Long tensor with shape [2, E].
    """
    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError("faces must have shape [F, 3].")

    faces = faces.long()
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = torch.cat([e01, e12, e20], dim=0)

    if make_undirected:
        edges = torch.cat([edges, edges[:, [1, 0]]], dim=0)

    edges = torch.unique(edges, dim=0)
    return edges.t().contiguous()


def edge_length_weights(
    vertices: Tensor,
    edge_index: Tensor,
    sigma: Optional[float] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Compute geometry-aware edge weights from Euclidean edge lengths.

    Args:
        vertices: Tensor with shape [V, 3] or [B, V, 3].
        edge_index: Long tensor with shape [2, E].
        sigma: RBF bandwidth. If None, use median edge length.
        eps: Numerical stability constant.

    Returns:
        Edge weights with shape [E]. If vertices are batched, the mean length
        across the batch is used.
    """
    if vertices.ndim == 3:
        v = vertices.mean(dim=0)
    elif vertices.ndim == 2:
        v = vertices
    else:
        raise ValueError("vertices must have shape [V, 3] or [B, V, 3].")

    row, col = edge_index.long()
    lengths = torch.linalg.norm(v[row] - v[col], dim=-1)

    if sigma is None:
        sigma = torch.median(lengths.detach()).clamp_min(eps).item()

    weights = torch.exp(-(lengths**2) / (2.0 * sigma**2 + eps))
    return weights.clamp_min(eps)


def normalized_message_passing(
    x: Tensor,
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Symmetric normalized graph aggregation.

    Computes approximately D^{-1/2} A D^{-1/2} X without external dependencies.

    Args:
        x: Node features with shape [B, V, F].
        edge_index: Long tensor with shape [2, E], where edge row <- col.
        edge_weight: Optional edge weights with shape [E].
        eps: Numerical stability constant.

    Returns:
        Aggregated node features with shape [B, V, F].
    """
    if x.ndim != 3:
        raise ValueError("x must have shape [B, V, F].")

    bsz, num_nodes, feat_dim = x.shape
    row, col = edge_index.long()

    if edge_weight is None:
        edge_weight = torch.ones(row.numel(), device=x.device, dtype=x.dtype)
    else:
        edge_weight = edge_weight.to(device=x.device, dtype=x.dtype)

    deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
    deg.index_add_(0, row, edge_weight)
    deg_inv_sqrt = torch.rsqrt(deg.clamp_min(eps))

    norm = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    messages = x[:, col, :] * norm.view(1, -1, 1)

    out = torch.zeros(bsz, num_nodes, feat_dim, device=x.device, dtype=x.dtype)
    out.index_add_(1, row, messages)
    return out


def mask_to_boundary(mask: Tensor, edge_index: Tensor) -> Tensor:
    """Convert a binary vertex mask to a graph-boundary mask.

    A vertex is considered a boundary vertex if at least one of its neighbors
    has a different binary label.

    Args:
        mask: Boolean or 0/1 tensor with shape [B, V] or [V].
        edge_index: Long tensor with shape [2, E].

    Returns:
        Boundary mask with shape [B, V].
    """
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)

    mask_bool = mask.bool()
    bsz, num_nodes = mask_bool.shape
    row, col = edge_index.long()

    diff = mask_bool[:, row] != mask_bool[:, col]

    boundary = torch.zeros(bsz, num_nodes, device=mask.device, dtype=torch.bool)
    boundary.scatter_(1, row.unsqueeze(0).expand(bsz, -1), diff)
    boundary.scatter_(1, col.unsqueeze(0).expand(bsz, -1), diff)
    return boundary


# ---------------------------------------------------------------------
# Metrics: segmentation-style evaluation for language-network boundaries
# ---------------------------------------------------------------------


def dice_score(
    prediction: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Tensor:
    """Dice similarity coefficient. Higher is better. Range: [0, 1]."""
    if prediction.shape[-1:] == (1,):
        prediction = prediction.squeeze(-1)
    if target.shape[-1:] == (1,):
        target = target.squeeze(-1)

    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)

    pred = (prediction >= threshold).float()
    tgt = target.float()

    intersection = (pred * tgt).sum(dim=1)
    denom = pred.sum(dim=1) + tgt.sum(dim=1)
    return ((2.0 * intersection + eps) / (denom + eps)).mean()


def iou_score(
    prediction: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Tensor:
    """Intersection-over-Union, also known as the Jaccard index."""
    if prediction.shape[-1:] == (1,):
        prediction = prediction.squeeze(-1)
    if target.shape[-1:] == (1,):
        target = target.squeeze(-1)

    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)

    pred = (prediction >= threshold).float()
    tgt = target.float()

    intersection = (pred * tgt).sum(dim=1)
    union = pred.sum(dim=1) + tgt.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()


@torch.no_grad()
def hd95_distance(
    prediction: Tensor,
    target: Tensor,
    vertices: Tensor,
    edge_index: Tensor,
    threshold: float = 0.5,
) -> Tensor:
    """95th-percentile Hausdorff distance between predicted and target boundaries.

    Lower is better. Units follow the units of `vertices`, for example mm.
    """
    if prediction.shape[-1:] == (1,):
        prediction = prediction.squeeze(-1)
    if target.shape[-1:] == (1,):
        target = target.squeeze(-1)

    if prediction.min() < 0 or prediction.max() > 1:
        prediction = torch.sigmoid(prediction)

    pred_mask = prediction >= threshold
    tgt_mask = target.bool()

    pred_boundary = mask_to_boundary(pred_mask, edge_index)
    tgt_boundary = mask_to_boundary(tgt_mask, edge_index)

    bsz, _ = pred_mask.shape

    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0).expand(bsz, -1, -1)
    elif vertices.ndim != 3:
        raise ValueError("vertices must have shape [V, 3] or [B, V, 3].")

    distances = []
    for b in range(bsz):
        p = vertices[b, pred_boundary[b]]
        g = vertices[b, tgt_boundary[b]]

        if p.numel() == 0 or g.numel() == 0:
            continue

        d_pg = torch.cdist(p, g).min(dim=1).values
        d_gp = torch.cdist(g, p).min(dim=1).values
        d_all = torch.cat([d_pg, d_gp], dim=0)
        distances.append(torch.quantile(d_all, 0.95))

    if not distances:
        return torch.tensor(float("nan"), device=prediction.device)

    return torch.stack(distances).mean()


def boundary_metrics(
    prediction: Tensor,
    target: Tensor,
    vertices: Tensor,
    edge_index: Tensor,
    threshold: float = 0.5,
) -> Dict[str, Tensor]:
    """Compute Dice, IoU and HD95 for cortical language-network masks."""
    return {
        "dice": dice_score(prediction, target, threshold=threshold),
        "iou": iou_score(prediction, target, threshold=threshold),
        "hd95": hd95_distance(
            prediction,
            target,
            vertices=vertices,
            edge_index=edge_index,
            threshold=threshold,
        ),
    }


# ---------------------------------------------------------------------
# Neural-network blocks
# ---------------------------------------------------------------------


class MLP(nn.Module):
    """Small feed-forward network."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int = 2,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1.")

        layers = []
        current = in_dim
        for _ in range(depth - 1):
            layers += [
                nn.Linear(current, hidden_dim),
                activation(),
                nn.Dropout(dropout),
            ]
            current = hidden_dim

        layers.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TemporalPerturbationEncoder(nn.Module):
    """Encode EEG perturbation-response time series.

    Input:
        eeg: [B, C, T]
        stimulus: optional [B, 1, T]

    Output:
        embedding: [B, D]
    """

    def __init__(
        self,
        eeg_channels: int,
        hidden_dim: int = 128,
        depth: int = 4,
        kernel_size: int = 7,
        use_stimulus_channel: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_stimulus_channel = use_stimulus_channel
        in_channels = eeg_channels + int(use_stimulus_channel)

        layers = []
        channels = in_channels
        for i in range(depth):
            dilation = 2**i
            padding = dilation * (kernel_size - 1) // 2
            layers += [
                nn.Conv1d(
                    channels,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            channels = hidden_dim

        self.temporal_net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, eeg: Tensor, stimulus: Optional[Tensor] = None) -> Tensor:
        if eeg.ndim != 3:
            raise ValueError("eeg must have shape [B, C, T].")

        if self.use_stimulus_channel:
            if stimulus is None:
                stimulus = torch.zeros(
                    eeg.shape[0],
                    1,
                    eeg.shape[-1],
                    device=eeg.device,
                    dtype=eeg.dtype,
                )
            if stimulus.ndim != 3 or stimulus.shape[1] != 1:
                raise ValueError("stimulus must have shape [B, 1, T].")
            eeg = torch.cat([eeg, stimulus.to(eeg.dtype)], dim=1)

        h = self.temporal_net(eeg)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)


class GraphResidualBlock(nn.Module):
    """Residual graph-convolution block using normalized message passing."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_proj = nn.Linear(dim, dim)
        self.msg_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        m = normalized_message_passing(x, edge_index, edge_weight=edge_weight)
        h = self.self_proj(x) + self.msg_proj(m)
        x = x + self.dropout(F.gelu(h))
        x = x + self.dropout(self.ffn(x))
        return self.norm(x)


class GeometricNeuralFieldBlock(nn.Module):
    r"""Graph neural-field integration block.

    This block implements a learnable discretized neural-field update on the
    cortical surface graph:

        dφ/dt = D · Δ_G φ - A · φ + u

    where:
        φ   is the neural field on cortical vertices,
        Δ_G is a graph Laplacian-like operator,
        D   is a learnable diffusion coefficient,
        A   is a learnable damping coefficient,
        u   is the NPI-driven source term.
    """

    def __init__(
        self,
        dim: int,
        steps: int = 4,
        dt: float = 0.25,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be >= 1.")
        self.steps = steps
        self.dt = dt

        self.log_diffusion = nn.Parameter(torch.zeros(dim))
        self.log_damping = nn.Parameter(torch.full((dim,), -2.0))
        self.source_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        field: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        source: Optional[Tensor] = None,
    ) -> Tensor:
        diffusion = F.softplus(self.log_diffusion).view(1, 1, -1)
        damping = F.softplus(self.log_damping).view(1, 1, -1)

        if source is None:
            source = torch.zeros_like(field)

        source = self.source_gate(source) * source

        phi = field
        for _ in range(self.steps):
            neighbor_avg = normalized_message_passing(
                phi,
                edge_index,
                edge_weight=edge_weight,
            )
            graph_laplacian = neighbor_avg - phi
            dphi = diffusion * graph_laplacian - damping * phi + source
            phi = phi + self.dt * dphi
            phi = self.norm(phi)
            phi = self.dropout(phi)

        return phi


@dataclass
class EncoderOutput:
    """Container returned by NPIGeometricNeuralFieldEncoder."""

    latent: Tensor
    field: Tensor
    logits: Tensor
    probabilities: Tensor
    eeg_embedding: Tensor


class NPIGeometricNeuralFieldEncoder(nn.Module):
    """NPI + Geometric Neural Field encoder.

    This model fuses perturbation-response EEG information with MRI-derived
    cortical geometry. It can be used as an encoder for downstream language
    network boundary prediction, region classification, or latent representation
    learning.
    """

    def __init__(
        self,
        eeg_channels: int,
        vertex_feature_dim: int = 0,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        graph_depth: int = 3,
        field_steps: int = 4,
        output_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vertex_feature_dim = vertex_feature_dim

        self.eeg_encoder = TemporalPerturbationEncoder(
            eeg_channels=eeg_channels,
            hidden_dim=hidden_dim,
            use_stimulus_channel=True,
            dropout=dropout,
        )

        self.geometry_encoder = MLP(
            in_dim=3,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            depth=3,
            dropout=dropout,
        )

        if vertex_feature_dim > 0:
            self.vertex_feature_encoder = MLP(
                in_dim=vertex_feature_dim,
                hidden_dim=hidden_dim,
                out_dim=hidden_dim,
                depth=2,
                dropout=dropout,
            )
        else:
            self.vertex_feature_encoder = None

        self.eeg_to_vertex = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.graph_blocks = nn.ModuleList(
            [GraphResidualBlock(hidden_dim, dropout=dropout) for _ in range(graph_depth)]
        )

        self.source_head = MLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            depth=2,
            dropout=dropout,
        )

        self.neural_field = GeometricNeuralFieldBlock(
            dim=hidden_dim,
            steps=field_steps,
            dt=0.25,
            dropout=dropout,
        )

        self.attention_pool = nn.Linear(hidden_dim, 1)
        self.latent_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.boundary_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        eeg: Tensor,
        vertices: Tensor,
        edge_index: Tensor,
        stimulus: Optional[Tensor] = None,
        vertex_features: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> EncoderOutput:
        """Forward pass.

        Args:
            eeg: EEG perturbation-response data with shape [B, C, T].
            vertices: MRI-derived cortical coordinates with shape [V, 3] or
                [B, V, 3].
            edge_index: Cortical mesh/graph edges with shape [2, E].
            stimulus: Optional perturbation timing channel with shape [B, 1, T].
            vertex_features: Optional cortical vertex features with shape
                [B, V, Fv].
            edge_weight: Optional edge weights with shape [E]. If None, uniform
                graph weights are used.

        Returns:
            EncoderOutput containing latent code, neural field, vertex logits,
            probabilities, and EEG embedding.
        """
        if eeg.ndim != 3:
            raise ValueError("eeg must have shape [B, C, T].")

        bsz = eeg.shape[0]
        edge_index = edge_index.to(eeg.device).long()

        if vertices.ndim == 2:
            vertices = vertices.unsqueeze(0).expand(bsz, -1, -1)
        elif vertices.ndim != 3:
            raise ValueError("vertices must have shape [V, 3] or [B, V, 3].")

        vertices = vertices.to(device=eeg.device, dtype=eeg.dtype)
        num_vertices = vertices.shape[1]

        edge_index_loop = add_self_loops(edge_index, num_vertices)

        if edge_weight is not None:
            edge_weight = edge_weight.to(device=eeg.device, dtype=eeg.dtype)
            self_loop_weight = torch.ones(num_vertices, device=eeg.device, dtype=eeg.dtype)
            edge_weight_loop = torch.cat([edge_weight, self_loop_weight], dim=0)
        else:
            edge_weight_loop = None

        eeg_embedding = self.eeg_encoder(eeg, stimulus=stimulus)
        geom = self.geometry_encoder(vertices)

        if self.vertex_feature_encoder is not None:
            if vertex_features is None:
                raise ValueError(
                    "vertex_features must be provided when vertex_feature_dim > 0."
                )
            vertex_features = vertex_features.to(device=eeg.device, dtype=eeg.dtype)
            geom = geom + self.vertex_feature_encoder(vertex_features)

        eeg_condition = self.eeg_to_vertex(eeg_embedding).unsqueeze(1)
        eeg_condition = eeg_condition.expand(-1, num_vertices, -1)

        x = self.fusion(torch.cat([geom, eeg_condition], dim=-1))

        for block in self.graph_blocks:
            x = block(x, edge_index_loop, edge_weight=edge_weight_loop)

        source = self.source_head(x + eeg_condition)
        field = self.neural_field(
            x,
            edge_index_loop,
            edge_weight=edge_weight_loop,
            source=source,
        )

        logits = self.boundary_head(field)
        probabilities = torch.sigmoid(logits)

        attn = torch.softmax(self.attention_pool(field), dim=1)
        attn_pool = (attn * field).sum(dim=1)
        mean_pool = field.mean(dim=1)
        max_pool = field.max(dim=1).values

        latent = self.latent_head(
            torch.cat([eeg_embedding, attn_pool, mean_pool, max_pool], dim=-1)
        )

        return EncoderOutput(
            latent=latent,
            field=field,
            logits=logits,
            probabilities=probabilities,
            eeg_embedding=eeg_embedding,
        )


# ---------------------------------------------------------------------
# Optional losses
# ---------------------------------------------------------------------


def neural_field_smoothness_loss(
    field: Tensor,
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
) -> Tensor:
    """Geometry-aware smoothness loss on cortical mesh edges.

    Penalizes abrupt changes of the neural field across adjacent cortical
    vertices. This is useful as a continuous-field regularizer.
    """
    row, col = edge_index.long()
    diff = field[:, row, :] - field[:, col, :]

    if edge_weight is not None:
        w = edge_weight.to(device=field.device, dtype=field.dtype).view(1, -1, 1)
        diff = diff * w

    return (diff**2).mean()


def supervised_boundary_loss(
    logits: Tensor,
    target: Tensor,
    pos_weight: Optional[Tensor] = None,
    smooth_weight: float = 0.0,
    field: Optional[Tensor] = None,
    edge_index: Optional[Tensor] = None,
) -> Tensor:
    """Binary boundary/region prediction loss.

    Combines BCE-with-logits and optional geometric smoothness.
    """
    if target.shape[-1:] != logits.shape[-1:]:
        target = target.unsqueeze(-1)

    target = target.to(device=logits.device, dtype=logits.dtype)

    loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
    )

    if smooth_weight > 0:
        if field is None or edge_index is None:
            raise ValueError("field and edge_index are required for smoothness loss.")
        loss = loss + smooth_weight * neural_field_smoothness_loss(field, edge_index)

    return loss


# ---------------------------------------------------------------------
# Minimal smoke test
# ---------------------------------------------------------------------


def _example() -> None:
    """Run a minimal forward pass with synthetic data."""
    torch.manual_seed(7)

    batch_size = 2
    eeg_channels = 64
    time_points = 256
    num_vertices = 128
    num_faces = 256

    eeg = torch.randn(batch_size, eeg_channels, time_points)

    stimulus = torch.zeros(batch_size, 1, time_points)
    stimulus[:, :, [40, 96, 160, 220]] = 1.0

    vertices = torch.randn(num_vertices, 3)
    faces = torch.randint(0, num_vertices, (num_faces, 3))
    edge_index = edges_from_faces(faces)
    edge_weight = edge_length_weights(vertices, edge_index)

    model = NPIGeometricNeuralFieldEncoder(
        eeg_channels=eeg_channels,
        hidden_dim=96,
        latent_dim=64,
        graph_depth=2,
        field_steps=3,
        output_dim=1,
    )

    out = model(
        eeg=eeg,
        vertices=vertices,
        edge_index=edge_index,
        stimulus=stimulus,
        edge_weight=edge_weight,
    )

    print("latent:", tuple(out.latent.shape))
    print("field:", tuple(out.field.shape))
    print("logits:", tuple(out.logits.shape))
    print("probabilities:", tuple(out.probabilities.shape))


if __name__ == "__main__":
    _example()
