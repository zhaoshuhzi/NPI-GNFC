from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .geometry import GeometryData
from .model import NPIGNFCModel
from .utils import ensure_dir, get_device, set_seed


def make_loss_weights(cfg: Config) -> Dict[str, float]:
    return {
        "npi": cfg.loss.lambda_npi,
        "field": cfg.loss.lambda_field,
        "kl": cfg.loss.lambda_kl,
        "meta": cfg.loss.lambda_meta,
        "diff": cfg.loss.lambda_diff,
    }


def build_model(cfg: Config, geometry: GeometryData, device: torch.device) -> NPIGNFCModel:
    model = NPIGNFCModel(
        eeg_channels=cfg.model.eeg_channels,
        n_regions=cfg.model.n_regions,
        eigenmodes=geometry.eigenmodes[:, : cfg.model.n_eigenmodes].to(device),
        eigenvalues=geometry.eigenvalues[: cfg.model.n_eigenmodes].to(device),
        latent_dim=cfg.model.latent_dim,
        text_embedding_dim=cfg.model.text_embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        diffusion_steps=cfg.model.diffusion_steps,
        dropout=cfg.model.dropout,
    ).to(device)
    return model


def train_model(
    cfg: Config,
    model: NPIGNFCModel,
    loader: DataLoader,
    output_dir: str | Path,
    device: torch.device | None = None,
) -> None:
    device = device or get_device()
    output_dir = ensure_dir(output_dir)
    set_seed(cfg.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_weights = make_loss_weights(cfg)

    model.train()
    global_step = 0
    for epoch in range(cfg.train.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{cfg.train.epochs}")
        running = 0.0
        for batch in pbar:
            eeg = batch["eeg"].to(device)
            y = batch["text_embeddings"].to(device)
            out = model(eeg, y, loss_weights=loss_weights)
            if out.loss is None:
                raise RuntimeError("Training requires target embeddings.")
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            if cfg.train.grad_clip is not None and cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            global_step += 1
            running += float(out.loss.detach().cpu())
            pbar.set_postfix(
                loss=f"{float(out.loss.detach().cpu()):.4f}",
                npi=f"{float(out.losses['npi'].detach().cpu()):.4f}",
                diff=f"{float(out.losses['diff'].detach().cpu()):.4f}",
            )
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": cfg,
        }
        torch.save(ckpt, output_dir / "last.pt")
