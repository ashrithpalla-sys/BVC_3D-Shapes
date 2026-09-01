"""Single entry point for the complete reproducible research workflow."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .config import load_config
from .data import generate_dataset, validate_archive
from .evaluation import evaluate, interpolate
from .training import train
from .visualization import contact_sheet


def main() -> None:
    parser = argparse.ArgumentParser(prog="stoolgen")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "train", "evaluate", "demo"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", default="configs/quick.yaml")
        if name in ("train", "evaluate"):
            child.add_argument("--model", choices=("ae", "vae"), default="vae")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "generate":
        cfg = config["data"]
        path = generate_dataset(cfg["path"], cfg["num_shapes"], cfg["points_per_shape"], config["seed"])
        with np.load(path) as archive:
            contact_sheet(archive["points"][:12], "figures/training_samples.png")
        print(json.dumps(validate_archive(path), indent=2))
    elif args.command == "train":
        print(train(config, args.model))
    elif args.command == "evaluate":
        print(json.dumps(evaluate(config, args.model), indent=2))
    else:
        generate = config["data"]
        generate_dataset(generate["path"], generate["num_shapes"], generate["points_per_shape"], config["seed"])
        train(config, "vae")
        print(json.dumps(evaluate(config, "vae"), indent=2))
        interpolate(config)

