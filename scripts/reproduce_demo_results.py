#!/usr/bin/env python
"""Lightweight CAGNNF / NPI-GNFC reproducibility demo for Code Ocean.

This script is self-contained and uses synthetic EEG, cortical geometry and
text-embedding tensors. It does not require access to restricted EEG data.
The goal is to verify that the capsule can execute the core computational
workflow: scalp EEG -> perturbation/network representation -> source-space
neural-field representation -> semantic embedding reconstruction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 2026) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class DemoCAGNNF(nn.Module):
    """A compact demo network mirroring the CAGNNF computation path.

    This is not intended to reproduce the full manuscript-scale training.
    It verifies the executable dependency chain and creates interpretable
    demo outputs for peer-review capsule testing.
    """

    def __init__(
        self,
        channels: int = 8,
        time_steps: int = 32,
        n_regions: int = 8,
        n_vertices: int = 10,
        n_eigenmodes: int = 4,
        latent_dim: int = 8,
        text_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.time_steps = time_steps
        self.n_regions = n_regions
        self.n_vertices = n_vertices
        self.n_eigenmodes = n_eigenmodes
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, latent_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.region_projection = nn.Linear(latent_dim, n_regions)
        self.npi_projection = nn.Linear(n_regions, n_regions * n_regions)
        self.field_projection = nn.Linear(n_regions, n_eigenmodes)
        self.semantic_decoder = nn.Sequential(
            nn.Linear(n_vertices, 32),
            nn.GELU(),
            nn.Linear(32, text_embedding_dim),
        )

    def forward(self, eeg: torch.Tensor, eigenmodes: torch.Tensor) -> Dict[str, torch.Tensor]:
        # eeg: [B, C, T]
        h = self.temporal_encoder(eeg).transpose(1, 2)  # [B, T, latent]
        regional_state = self.region_projection(h)  # [B, T, R]

        # NPI-like directional propagation operator.
        adjacency = self.npi_projection(regional_state).reshape(
            eeg.shape[0], eeg.shape[-1], self.n_regions, self.n_regions
        )
        adjacency = torch.softmax(adjacency, dim=-1)
        propagated = torch.einsum("btrr,btr->btr", adjacency, regional_state)

        # GNFC-like projection into cortical eigenmodes and source-space field.
        coeff = self.field_projection(propagated)  # [B, T, K]
        source_field = torch.einsum("btk,vk->btv", coeff, eigenmodes)  # [B, T, V]

        pooled_field = source_field.mean(dim=1)  # [B, V]
        decoded_embedding = self.semantic_decoder(pooled_field)  # [B, E]
        return {
            "regional_state": regional_state,
            "adjacency": adjacency,
            "source_field": source_field,
            "decoded_embedding": decoded_embedding,
        }


def make_synthetic_batch(
    n_samples: int = 16,
    channels: int = 8,
    time_steps: int = 32,
    text_embedding_dim: int = 16,
) -> Dict[str, torch.Tensor]:
    eeg = torch.randn(n_samples, channels, time_steps)
    # Create a structured target embedding so the demo loss is meaningful.
    summary = torch.cat([eeg.mean(dim=-1), eeg.std(dim=-1)], dim=1)
    projection = torch.randn(summary.shape[1], text_embedding_dim) / math.sqrt(summary.shape[1])
    text_embeddings = torch.tanh(summary @ projection)
    return {"eeg": eeg, "text_embeddings": text_embeddings}


def make_synthetic_geometry(n_vertices: int = 10, n_eigenmodes: int = 4) -> torch.Tensor:
    x = torch.linspace(0, 1, n_vertices)
    modes: List[torch.Tensor] = []
    for k in range(1, n_eigenmodes + 1):
        modes.append(torch.sin(math.pi * k * x))
    eigenmodes = torch.stack(modes, dim=1)
    eigenmodes = F.normalize(eigenmodes, dim=0)
    return eigenmodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = make_synthetic_batch()
    eigenmodes = make_synthetic_geometry()
    model = DemoCAGNNF()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses: List[float] = []
    for step in range(args.steps):
        outputs = model(batch["eeg"], eigenmodes)
        loss = F.mse_loss(outputs["decoded_embedding"], batch["text_embeddings"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(f"step={step:03d} demo_mse={losses[-1]:.6f}")

    model.eval()
    with torch.no_grad():
        outputs = model(batch["eeg"], eigenmodes)
        final_mse = F.mse_loss(outputs["decoded_embedding"], batch["text_embeddings"]).item()
        cosine = F.cosine_similarity(outputs["decoded_embedding"], batch["text_embeddings"], dim=-1).mean().item()

    metrics = {
        "demo_name": "CAGNNF synthetic Code Ocean smoke test",
        "steps": args.steps,
        "seed": args.seed,
        "initial_mse": losses[0],
        "final_mse": final_mse,
        "mean_cosine_similarity": cosine,
        "decoded_embedding_shape": list(outputs["decoded_embedding"].shape),
        "source_field_shape": list(outputs["source_field"].shape),
        "adjacency_shape": list(outputs["adjacency"].shape),
    }

    with (out_dir / "demo_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (out_dir / "demo_loss_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "demo_mse"])
        for step, value in enumerate(losses):
            writer.writerow([step, value])

    np.save(out_dir / "demo_source_field.npy", outputs["source_field"].detach().cpu().numpy())
    np.save(out_dir / "demo_decoded_embeddings.npy", outputs["decoded_embedding"].detach().cpu().numpy())

    print("decoded embedding shape:", tuple(outputs["decoded_embedding"].shape))
    print("source field shape:", tuple(outputs["source_field"].shape))
    print("metrics written to:", str(out_dir / "demo_metrics.json"))


if __name__ == "__main__":
    main()
