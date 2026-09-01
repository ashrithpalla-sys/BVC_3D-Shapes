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
    """Held-out ridge-probe R² measuring whether procedural factors are linearly readable."""
    split = max(2, int(len(latents) * 0.7))
    train_x, test_x = latents[:split], latents[split:]
    train_y, test_y = parameters[:split], parameters[split:]
    mean, scale = train_x.mean(0), train_x.std(0) + 1e-6
    train_design = np.column_stack(((train_x - mean) / scale, np.ones(len(train_x))))
    test_design = np.column_stack(((test_x - mean) / scale, np.ones(len(test_x))))
    ridge = np.eye(train_design.shape[1]) * 1e-2
    ridge[-1, -1] = 0  # Do not regularize the intercept.
    weights = np.linalg.solve(train_design.T @ train_design + ridge,
                              train_design.T @ train_y)
    prediction = test_design @ weights
    residual = ((test_y - prediction) ** 2).sum(axis=0)
    total = ((test_y - train_y.mean(axis=0)) ** 2).sum(axis=0) + 1e-8
    return float(np.mean(1 - residual / total))


def _cloud_distance(first: np.ndarray, second: np.ndarray, device: torch.device) -> float:
    """Evaluate one pair without keeping a large all-pairs matrix in memory."""
    a = torch.from_numpy(first).to(device)[None]
    b = torch.from_numpy(second).to(device)[None]
    return float(chamfer_distance(a, b).cpu())


def _sample_quality(samples: np.ndarray, reference: np.ndarray,
                    device: torch.device) -> dict[str, float]:
    """Measure diversity, training-set proximity, and coarse stool structure.

    Structural validity deliberately checks only representation-level evidence:
    a wide upper seat region, points reaching the floor, and a nonempty lower body.
    It does not claim mesh watertightness or mechanical stability.
    """
    subset = samples[: min(24, len(samples))]
    references = reference[: min(64, len(reference))]
    pairwise = []
    for i in range(len(subset)):
        for j in range(i + 1, min(i + 5, len(subset))):
            pairwise.append(_cloud_distance(subset[i], subset[j], device))
    nearest = []
    for sample in subset:
        nearest.append(min(_cloud_distance(sample, target, device) for target in references))
    valid = []
    for sample in samples:
        top = sample[sample[:, 2] > 0.55]
        lower = sample[sample[:, 2] < 0.25]
        top_extent = np.ptp(top[:, :2], axis=0) if len(top) else np.zeros(2)
        valid.append(
            len(top) >= len(sample) * 0.12
            and len(lower) >= len(sample) * 0.25
            and np.all(top_extent > 0.35)
            and sample[:, 2].min() < -0.65
        )
    return {
        "generated_pairwise_chamfer_mean": float(np.mean(pairwise)),
        "generated_nearest_training_chamfer_mean": float(np.mean(nearest)),
        "generated_stool_structure_fraction": float(np.mean(valid)),
    }


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
    # Use the full procedural archive for the descriptive probe so the quick
    # configuration still has substantially more samples than latent dimensions.
    with np.load(data_cfg["path"]) as archive:
        probe_points = torch.from_numpy(archive["points"].astype(np.float32))
        parameter_array = archive["parameters"].astype(np.float32)
    probe_latents = []
    with torch.no_grad():
        for start in range(0, len(probe_points), config["training"]["batch_size"]):
            probe_latents.append(_latent(model, probe_points[start:start + config["training"]["batch_size"]].to(device)).cpu().numpy())
    latent_array = np.concatenate(probe_latents)
    metrics = {"test_chamfer_mean": float(np.mean(distances)),
               "test_chamfer_std": float(np.std(distances)),
               "latent_parameter_probe_r2": _linear_probe(latent_array, parameter_array)}
    if kind == "vae":
        with torch.no_grad():
            samples = model.sample(config["evaluation"]["num_samples"], device).cpu().numpy()
        extents = np.ptp(samples, axis=1)
        metrics["generated_noncollapsed_fraction"] = float(np.mean(np.all(extents > 0.08, axis=1)))
        with np.load(data_cfg["path"]) as archive:
            reference = archive["points"].astype(np.float32)
        metrics.update(_sample_quality(samples, reference, device))
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


def latent_traversal(config: dict) -> np.ndarray:
    """Traverse the most variable encoded direction around an observed test shape."""
    device = choose_device(config.get("device", "auto"))
    model, _ = load_checkpoint(Path(config["training"]["output_dir"]) / "vae" / "best.pt", device)
    data_cfg = config["data"]
    dataset = StoolPointCloudDataset(data_cfg["path"], "test", config["seed"],
                                     data_cfg["train_fraction"], data_cfg["val_fraction"])
    batch = torch.stack([dataset[i]["points"] for i in range(len(dataset))]).to(device)
    with torch.no_grad():
        encoded = _latent(model, batch)
        dimension = int(encoded.var(0).argmax())
        center = encoded.mean(0)
        scale = encoded[:, dimension].std().clamp_min(0.1)
        steps = torch.linspace(-2, 2, config["evaluation"]["interpolation_steps"], device=device)
        traversal = center.repeat(len(steps), 1)
        traversal[:, dimension] += steps * scale
        sequence = model.decode(traversal).cpu().numpy()
    contact_sheet(sequence, "figures/latent_traversal.png",
                  [f"{value:+.1f} sigma" for value in steps.cpu().numpy()], columns=len(sequence))
    return sequence
