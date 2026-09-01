"""Deterministic point-cloud autoencoder baseline."""

import torch
from torch import nn

from .pointnet import PointDecoder, PointNetEncoder


class PointCloudAutoencoder(nn.Module):
    def __init__(self, num_points: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim, hidden_dim)
        self.decoder = PointDecoder(latent_dim, num_points, hidden_dim)

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        return self.encoder(points)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(points)
        return {"points": self.decode(latent), "latent": latent}

