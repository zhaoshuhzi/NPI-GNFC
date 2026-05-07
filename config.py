from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    eeg_channels: int = 128
    time_steps: int = 256
    n_regions: int = 128
    n_eigenmodes: int = 32
    latent_dim: int = 128
    text_embedding_dim: int = 768
    hidden_dim: int = 256
    diffusion_steps: int = 100
    dropout: float = 0.1


@dataclass
class LossConfig:
    lambda_npi: float = 1.0
    lambda_field: float = 0.1
    lambda_kl: float = 0.01
    lambda_meta: float = 1.0
    lambda_diff: float = 1.0
    lambda_recon: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0


@dataclass
class DataConfig:
    batch_size: int = 8
    num_workers: int = 0


@dataclass
class Config:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


def _update_dataclass(obj: Any, values: Dict[str, Any]) -> Any:
    for key, value in values.items():
        if hasattr(obj, key):
            current = getattr(obj, key)
            if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
                _update_dataclass(current, value)
            else:
                setattr(obj, key, value)
    return obj


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _update_dataclass(cfg, raw)
