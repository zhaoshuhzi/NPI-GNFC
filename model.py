from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .diffusion import ConditionalDiffusionDecoder
from .encoder import EncoderOutput, GeometricNeuralFieldEncoder
from .losses import cosine_embedding_loss
from .maml import SemanticProjectionHead


@dataclass
class NPIGNFCOutput:
    loss: Optional[Tensor]
    losses: Dict[str, Tensor]
    encoder: EncoderOutput
    semantic_prediction: Tensor
    decoded_embeddings: Optional[Tensor] = None


class NPIGNFCModel(nn.Module):
    """NPI+GNFC model.

    Pipeline:
        EEG dynamics -> Geometric Neural Field Encoder -> neural-field latent z
        z -> Meta-Diffusion Semantic Decoder -> semantic embeddings/text layer
    """

    def __init__(
        self,
        eeg_channels: int,
        n_regions: int,
        eigenmodes: Tensor,
        eigenvalues: Tensor,
        latent_dim: int = 128,
        text_embedding_dim: int = 768,
        hidden_dim: int = 256,
        diffusion_steps: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GeometricNeuralFieldEncoder(
            eeg_channels=eeg_channels,
            n_regions=n_regions,
            eigenmodes=eigenmodes,
            eigenvalues=eigenvalues,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.semantic_head = SemanticProjectionHead(latent_dim, text_embedding_dim, hidden_dim)
        self.diffusion_decoder = ConditionalDiffusionDecoder(
            latent_dim=latent_dim,
            text_embedding_dim=text_embedding_dim,
            hidden_dim=hidden_dim,
            diffusion_steps=diffusion_steps,
            dropout=dropout,
        )

    def forward(
        self,
        eeg: Tensor,
        target_embeddings: Optional[Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> NPIGNFCOutput:
        enc = self.encoder(eeg)
        semantic_pred = self.semantic_head(enc.z)
        losses: Dict[str, Tensor] = dict(enc.losses)

        if target_embeddings is not None:
            losses["meta"] = F.mse_loss(semantic_pred, target_embeddings) + cosine_embedding_loss(
                semantic_pred, target_embeddings
            )
            diff_out = self.diffusion_decoder.training_loss(enc.z, target_embeddings)
            losses["diff"] = diff_out.loss
        else:
            losses["meta"] = torch.zeros((), device=eeg.device)
            losses["diff"] = torch.zeros((), device=eeg.device)

        if loss_weights is not None:
            total = torch.zeros((), device=eeg.device)
            for name, value in losses.items():
                total = total + float(loss_weights.get(name, 1.0)) * value
        elif target_embeddings is not None:
            total = sum(losses.values())
        else:
            total = None
        return NPIGNFCOutput(loss=total, losses=losses, encoder=enc, semantic_prediction=semantic_pred)

    @torch.no_grad()
    def decode_embeddings(self, eeg: Tensor, steps: int | None = None) -> NPIGNFCOutput:
        enc = self.encoder(eeg)
        decoded = self.diffusion_decoder.sample(enc.z, steps=steps)
        semantic_pred = self.semantic_head(enc.z)
        return NPIGNFCOutput(
            loss=None,
            losses=dict(enc.losses),
            encoder=enc,
            semantic_prediction=semantic_pred,
            decoded_embeddings=decoded,
        )
