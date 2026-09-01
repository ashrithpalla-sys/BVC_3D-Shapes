"""Dataset creation, validation, splitting, and PyTorch access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import PARAMETER_NAMES, generate_stool, sample_parameters


def generate_dataset(path: str | Path, num_shapes: int, points_per_shape: int, seed: int) -> Path:
    if num_shapes < 3 or points_per_shape < 32:
        raise ValueError("Dataset requires at least 3 shapes and 32 points per shape")
    rng = np.random.default_rng(seed)
    clouds, labels, parameters = [], [], []
    for _ in range(num_shapes):
        params = sample_parameters(rng)
        cloud, part_labels = generate_stool(params, points_per_shape, rng)
        clouds.append(cloud)
        labels.append(part_labels)
        parameters.append(params.array())
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, points=np.stack(clouds), labels=np.stack(labels),
                        parameters=np.stack(parameters), parameter_names=PARAMETER_NAMES, seed=seed)
    return destination


def validate_archive(path: str | Path) -> dict[str, float | int]:
    with np.load(path) as archive:
        points = archive["points"]
        if points.ndim != 3 or points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("Point archive must be finite with shape (S, N, 3)")
        return {"shapes": len(points), "points_per_shape": points.shape[1],
                "coordinate_min": float(points.min()), "coordinate_max": float(points.max())}


class StoolPointCloudDataset(Dataset):
    def __init__(self, path: str | Path, split: str, seed: int = 0,
                 train_fraction: float = 0.8, val_fraction: float = 0.1, augment: bool = False):
        archive = np.load(path)
        self.points = archive["points"].astype(np.float32)
        self.parameters = archive["parameters"].astype(np.float32)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.points))
        train_end = int(len(indices) * train_fraction)
        val_end = train_end + int(len(indices) * val_fraction)
        ranges = {"train": indices[:train_end], "val": indices[train_end:val_end],
                  "test": indices[val_end:]}
        if split not in ranges:
            raise ValueError(f"Unknown split {split!r}")
        self.indices = ranges[split]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        points = torch.from_numpy(self.points[index].copy())
        if self.augment:
            angle = torch.rand(()) * 2 * torch.pi
            rotation = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                                     [torch.sin(angle), torch.cos(angle), 0], [0, 0, 1]])
            points = points @ rotation.T + torch.randn_like(points) * 0.003
        return {"points": points, "parameters": torch.from_numpy(self.parameters[index]),
                "index": torch.tensor(index)}

