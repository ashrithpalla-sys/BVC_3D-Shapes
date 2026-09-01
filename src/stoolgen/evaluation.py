"""Quantitative evaluation and latent-space probes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import choose_device
from .data import StoolPointCloudDataset
from .losses import chamfer_distance
from .training import load_checkpoint
from .visualization import contact_sheet


def _latent(model, points: torch.Tensor) -> torch.Tensor:
    encoded = model.encode(points)
    return encoded[0] if isinstance(encoded, tuple) else encoded


def _linear_probe(latents: np.ndarray, parameters: np.ndarray) -> float:
    """Mean in-sample R²; a descriptive probe, not a held-out predictive claim."""
    design = np.column_stack((latents, np.ones(len(latents))))
    prediction = design @ np.linalg.lstsq(design, parameters, rcond=None)[0]
    residual = ((parameters - prediction) ** 2).sum(axis=0)
    total = ((parameters - parameters.mean(axis=0)) ** 2).sum(axis=0) + 1e-8
    return float(np.mean(1 - residual / total))


def evaluate(config: dict, kind: str) -> dict[str, float]:
    device = choose_device(config.get("device", "auto"))
    checkpoint_path = Path(config["training"]["output_dir"]) / kind / "best.pt"
    model, _ = load_checkpoint(checkpoint_path, device)
    data_cfg = config["data"]
    dataset = StoolPointCloudDataset(data_cfg["path"], "test", config["seed"],
                                     data_cfg["train_fraction"], data_cfg["val_fraction"])
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"])
    distances, latents, parameters, reconstructions, inputs = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            output = model(points)
            for predicted, target in zip(output["points"], points):
                distances.append(float(chamfer_distance(predicted[None], target[None])))
            latents.append(_latent(model, points).cpu().numpy())
            parameters.append(batch["parameters"].numpy())
            reconstructions.append(output["points"].cpu().numpy())
            inputs.append(points.cpu().numpy())
    latent_array, parameter_array = np.concatenate(latents), np.concatenate(parameters)
    metrics = {"test_chamfer_mean": float(np.mean(distances)),
               "test_chamfer_std": float(np.std(distances)),
               "latent_parameter_probe_r2": _linear_probe(latent_array, parameter_array)}
    if kind == "vae":
        with torch.no_grad():
            samples = model.sample(config["evaluation"]["num_samples"], device).cpu().numpy()
        extents = np.ptp(samples, axis=1)
        metrics["generated_noncollapsed_fraction"] = float(np.mean(np.all(extents > 0.08, axis=1)))
        contact_sheet(samples[:12], "figures/generated_samples.png")
    results_dir = Path(config["training"]["output_dir"]) / kind
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    input_array, reconstruction_array = np.concatenate(inputs), np.concatenate(reconstructions)
    paired = np.stack([item for pair in zip(input_array[:4], reconstruction_array[:4]) for item in pair])
    contact_sheet(paired, "figures/reconstructions.png",
                  [label for i in range(4) for label in (f"Input {i + 1}", f"Reconstruction {i + 1}")])
    return metrics


def interpolate(config: dict, first: int = 0, second: int = 1) -> np.ndarray:
    device = choose_device(config.get("device", "auto"))
    model, _ = load_checkpoint(Path(config["training"]["output_dir"]) / "vae" / "best.pt", device)
    data_cfg = config["data"]
    dataset = StoolPointCloudDataset(data_cfg["path"], "test", config["seed"],
                                     data_cfg["train_fraction"], data_cfg["val_fraction"])
    pair = torch.stack((dataset[first]["points"], dataset[second]["points"])).to(device)
    with torch.no_grad():
        endpoints = _latent(model, pair)
        values = torch.linspace(0, 1, config["evaluation"]["interpolation_steps"], device=device)
        latent = torch.stack([(1 - value) * endpoints[0] + value * endpoints[1] for value in values])
        sequence = model.decode(latent).cpu().numpy()
    contact_sheet(sequence, "figures/interpolation.png",
                  [f"t={value:.2f}" for value in np.linspace(0, 1, len(sequence))], columns=len(sequence))
    return sequence

