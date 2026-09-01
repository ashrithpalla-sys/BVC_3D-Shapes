"""Run the planned latent-size study and save one machine-readable result table.

This intentionally reuses the same dataset and split across conditions. Example:
    PYTHONPATH=src python3 scripts/run_latent_ablation.py --config configs/quick.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from stoolgen.config import load_config
from stoolgen.data import generate_dataset
from stoolgen.evaluation import evaluate
from stoolgen.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--dimensions", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--output", default="outputs/latent_ablation.json")
    args = parser.parse_args()
    base = load_config(args.config)
    data = base["data"]
    generate_dataset(data["path"], data["num_shapes"], data["points_per_shape"], base["seed"])
    results = []
    for dimension in args.dimensions:
        config = copy.deepcopy(base)
        config["model"]["latent_dim"] = dimension
        # Separate run directories prevent one condition from overwriting another.
        config["training"]["output_dir"] = f"runs/latent_{dimension}"
        train(config, "vae")
        results.append({"latent_dim": dimension, **evaluate(config, "vae")})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()

