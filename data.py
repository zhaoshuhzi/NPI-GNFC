from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class EEGTextNPZDataset(Dataset):
    """NPZ dataset for EEG-to-embedding training."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        data = np.load(self.path, allow_pickle=True)
        if "eeg" not in data or "text_embeddings" not in data:
            raise KeyError("NPZ must contain 'eeg' [N,C,T] and 'text_embeddings' [N,E].")
        self.eeg = torch.as_tensor(data["eeg"], dtype=torch.float32)
        self.text_embeddings = torch.as_tensor(data["text_embeddings"], dtype=torch.float32)
        self.subject_ids = torch.as_tensor(data["subject_ids"], dtype=torch.long) if "subject_ids" in data else None
        self.texts = data["texts"] if "texts" in data else None
        if self.eeg.shape[0] != self.text_embeddings.shape[0]:
            raise ValueError("eeg and text_embeddings must have the same first dimension.")

    def __len__(self) -> int:
        return int(self.eeg.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "eeg": self.eeg[idx],
            "text_embeddings": self.text_embeddings[idx],
        }
        if self.subject_ids is not None:
            item["subject_ids"] = self.subject_ids[idx]
        if self.texts is not None:
            item["texts"] = str(self.texts[idx])
        return item


class SyntheticEEGTextDataset(Dataset):
    """Synthetic dataset for smoke tests and CI."""

    def __init__(self, n_samples: int = 64, channels: int = 128, time_steps: int = 256, embedding_dim: int = 768):
        g = torch.Generator().manual_seed(123)
        self.eeg = torch.randn(n_samples, channels, time_steps, generator=g)
        # Create target embeddings partly correlated with EEG mean to make the loss learnable.
        summary = self.eeg.mean(dim=-1)
        proj = torch.randn(channels, embedding_dim, generator=g) / channels**0.5
        self.text_embeddings = summary @ proj + 0.05 * torch.randn(n_samples, embedding_dim, generator=g)
        self.subject_ids = torch.arange(n_samples) % 8

    def __len__(self) -> int:
        return int(self.eeg.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {
            "eeg": self.eeg[idx],
            "text_embeddings": self.text_embeddings[idx],
            "subject_ids": self.subject_ids[idx],
        }
