"""Small PointNet-inspired encoders and folding-free point decoders."""

from __future__ import annotations

import torch
from torch import nn


class PointNetEncoder(nn.Module):
    """Create a permutation-invariant shape descriptor using shared MLPs and max pooling."""

    def __init__(self, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, hidden_dim, 1), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 1), nn.ReLU(),
        )
        self.project = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.features(points.transpose(1, 2))
        return self.project(features.max(dim=2).values)


class PointDecoder(nn.Module):
    def __init__(self, latent_dim: int, num_points: int, hidden_dim: int = 128):
        super().__init__()
        self.num_points = num_points
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_points * 3), nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent).reshape(-1, self.num_points, 3)

