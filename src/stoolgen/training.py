"""Shared train/evaluation loop for the deterministic and variational models."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import choose_device, seed_everything
from .data import StoolPointCloudDataset
from .losses import chamfer_distance, kl_divergence
from .models import PointCloudAutoencoder, PointCloudVAE


def build_model(kind: str, config: dict, num_points: int) -> torch.nn.Module:
    kwargs = dict(num_points=num_points, latent_dim=config["model"]["latent_dim"],
                  hidden_dim=config["model"]["hidden_dim"])
    if kind == "ae":
        return PointCloudAutoencoder(**kwargs)
    if kind == "vae":
        return PointCloudVAE(**kwargs)
    raise ValueError(f"Unknown model kind {kind!r}")


def _epoch(model, loader, optimizer, device, beta: float, chunk: int) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "chamfer": 0.0, "kl": 0.0}
    for batch in loader:
        points = batch["points"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(points)
            reconstruction = chamfer_distance(output["points"], points, chunk)
            kl = kl_divergence(output["mean"], output["log_variance"]) if "mean" in output else points.new_zeros(())
            loss = reconstruction + beta * kl
            if training:
                loss.backward()
                optimizer.step()
        for key, value in (("loss", loss), ("chamfer", reconstruction), ("kl", kl)):
            totals[key] += float(value.detach()) * len(points)
    return {key: value / len(loader.dataset) for key, value in totals.items()}


def train(config: dict, kind: str) -> Path:
    seed_everything(config["seed"])
    data_cfg, train_cfg = config["data"], config["training"]
    train_set = StoolPointCloudDataset(data_cfg["path"], "train", config["seed"],
                                       data_cfg["train_fraction"], data_cfg["val_fraction"], True)
    val_set = StoolPointCloudDataset(data_cfg["path"], "val", config["seed"],
                                     data_cfg["train_fraction"], data_cfg["val_fraction"])
    loaders = [DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=(i == 0))
               for i, dataset in enumerate((train_set, val_set))]
    device = choose_device(config.get("device", "auto"))
    model = build_model(kind, config, train_set.points.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    output_dir = Path(train_cfg["output_dir"]) / kind
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = _epoch(model, loaders[0], optimizer, device, train_cfg["beta"], train_cfg["chamfer_chunk"])
        with torch.no_grad():
            val_metrics = _epoch(model, loaders[1], None, device, train_cfg["beta"], train_cfg["chamfer_chunk"])
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(f"epoch {epoch:03d} train={train_metrics['loss']:.5f} val={val_metrics['loss']:.5f}")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            torch.save({"model": model.state_dict(), "kind": kind, "config": config,
                        "epoch": epoch, "val": val_metrics}, output_dir / "best.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output_dir / "best.pt"


def load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    data = np_load_shape(checkpoint["config"]["data"]["path"])
    model = build_model(checkpoint["kind"], checkpoint["config"], data).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def np_load_shape(path: str | Path) -> int:
    import numpy as np
    with np.load(path) as archive:
        return int(archive["points"].shape[1])

