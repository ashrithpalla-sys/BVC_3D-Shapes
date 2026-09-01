"""Variational point-cloud autoencoder with a sampleable Gaussian latent space."""

import torch
from torch import nn

from .pointnet import PointDecoder, PointNetEncoder


class PointCloudVAE(nn.Module):
    def __init__(self, num_points: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = PointNetEncoder(latent_dim * 2, hidden_dim)
        self.decoder = PointDecoder(latent_dim, num_points, hidden_dim)

    def encode(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(points).chunk(2, dim=-1)

    @staticmethod
    def reparameterize(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def sample(self, count: int, device: torch.device) -> torch.Tensor:
        return self.decode(torch.randn(count, self.latent_dim, device=device))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, log_variance = self.encode(points)
        latent = self.reparameterize(mean, log_variance)
        return {"points": self.decode(latent), "latent": latent,
                "mean": mean, "log_variance": log_variance}

