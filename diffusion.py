from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: Tensor) -> Tensor:
        device = timesteps.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / max(1, half - 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DenoiserMLP(nn.Module):
    def __init__(self, emb_dim: int, cond_dim: int, hidden_dim: int = 256, time_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim + cond_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, noisy_emb: Tensor, timesteps: Tensor, cond: Tensor) -> Tensor:
        t_emb = self.time_emb(timesteps)
        return self.net(torch.cat([noisy_emb, cond, t_emb], dim=-1))


@dataclass
class DiffusionOutput:
    loss: Tensor
    predicted_noise: Tensor
    timesteps: Tensor
    noisy_embeddings: Tensor


class ConditionalDiffusionDecoder(nn.Module):
    """Conditional DDPM-style decoder from neural-field latent z to text embeddings."""

    def __init__(
        self,
        latent_dim: int,
        text_embedding_dim: int,
        hidden_dim: int = 256,
        diffusion_steps: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        self.diffusion_steps = diffusion_steps
        self.cond_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.denoiser = DenoiserMLP(
            emb_dim=text_embedding_dim,
            cond_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        alpha_bar = self.alpha_bars[t].view(-1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

    def training_loss(self, z: Tensor, target_embeddings: Tensor) -> DiffusionOutput:
        bsz = target_embeddings.shape[0]
        t = torch.randint(0, self.diffusion_steps, (bsz,), device=target_embeddings.device)
        noise = torch.randn_like(target_embeddings)
        noisy = self.q_sample(target_embeddings, t, noise)
        cond = self.cond_proj(z)
        pred_noise = self.denoiser(noisy, t, cond)
        loss = F.mse_loss(pred_noise, noise)
        return DiffusionOutput(loss=loss, predicted_noise=pred_noise, timesteps=t, noisy_embeddings=noisy)

    @torch.no_grad()
    def sample(self, z: Tensor, steps: int | None = None) -> Tensor:
        """Generate semantic embeddings conditioned on neural-field latent z."""
        steps = steps or self.diffusion_steps
        if steps > self.diffusion_steps:
            raise ValueError("steps cannot exceed diffusion_steps.")
        x = torch.randn(z.shape[0], self.text_embedding_dim, device=z.device)
        cond = self.cond_proj(z)
        for t_int in reversed(range(steps)):
            t = torch.full((z.shape[0],), t_int, device=z.device, dtype=torch.long)
            beta_t = self.betas[t].view(-1, 1)
            alpha_t = self.alphas[t].view(-1, 1)
            alpha_bar_t = self.alpha_bars[t].view(-1, 1)
            pred_noise = self.denoiser(x, t, cond)
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise)
            if t_int > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean
        return x
