from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from npi_gnfc.config import Config
from npi_gnfc.data import SyntheticEEGTextDataset
from npi_gnfc.geometry import synthetic_geometry
from npi_gnfc.train import build_model, make_loss_weights
from npi_gnfc.utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.model.eeg_channels = 8
    cfg.model.time_steps = 32
    cfg.model.n_regions = 8
    cfg.model.n_eigenmodes = 4
    cfg.model.latent_dim = 8
    cfg.model.text_embedding_dim = 16
    cfg.model.hidden_dim = 16
    cfg.model.diffusion_steps = 5
    cfg.data.batch_size = 4

    set_seed(cfg.seed)
    device = get_device(args.device)
    dataset = SyntheticEEGTextDataset(
        n_samples=16,
        channels=cfg.model.eeg_channels,
        time_steps=cfg.model.time_steps,
        embedding_dim=cfg.model.text_embedding_dim,
    )
    loader = DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=True)
    geometry = synthetic_geometry(n_vertices=10, n_eigenmodes=cfg.model.n_eigenmodes, device=device)
    model = build_model(cfg, geometry, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    weights = make_loss_weights(cfg)

    model.train()
    step = 0
    while step < args.steps:
        for batch in loader:
            eeg = batch["eeg"].to(device)
            y = batch["text_embeddings"].to(device)
            out = model(eeg, y, loss_weights=weights)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            opt.step()
            print(
                f"step={step:03d} loss={out.loss.item():.4f} "
                f"npi={out.losses['npi'].item():.4f} "
                f"field={out.losses['field'].item():.4f} "
                f"kl={out.losses['kl'].item():.4f} "
                f"meta={out.losses['meta'].item():.4f} "
                f"diff={out.losses['diff'].item():.4f}"
            )
            step += 1
            if step >= args.steps:
                break

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        decoded = model.decode_embeddings(batch["eeg"].to(device), steps=5)
        print("decoded embedding shape:", tuple(decoded.decoded_embeddings.shape))
        print("field shape:", tuple(decoded.encoder.field.shape))


if __name__ == "__main__":
    main()
