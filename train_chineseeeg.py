from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from npi_gnfc.config import load_config
from npi_gnfc.data import EEGTextNPZDataset
from npi_gnfc.geometry import load_geometry_npz
from npi_gnfc.train import build_model, train_model
from npi_gnfc.utils import get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NPI+GNFC on an EEG+text embedding NPZ dataset.")
    parser.add_argument("--data", required=True, help="NPZ with eeg [N,C,T] and text_embeddings [N,E].")
    parser.add_argument("--geometry", required=True, help="NPZ with eigenmodes [V,K] and eigenvalues [K].")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="runs/npi_gnfc")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(args.device)

    dataset = EEGTextNPZDataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )
    geometry = load_geometry_npz(args.geometry, device=device)
    model = build_model(cfg, geometry, device)
    train_model(cfg, model, loader, output_dir=args.output, device=device)


if __name__ == "__main__":
    main()
