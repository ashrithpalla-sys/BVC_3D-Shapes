"""Compare KL weights across seeds while holding architecture and data size fixed."""

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
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.0002, 0.001])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--output", default="outputs/beta_ablation.json")
    args = parser.parse_args()
    base = load_config(args.config)
    records = []
    for seed in args.seeds:
        data_path = f"data/stools_ablation_{seed}.npz"
        data = base["data"]
        generate_dataset(data_path, data["num_shapes"], data["points_per_shape"], seed)
        for beta in args.betas:
            config = copy.deepcopy(base)
            config["seed"] = seed
            config["data"]["path"] = data_path
            config["training"]["beta"] = beta
            label = str(beta).replace(".", "p")
            config["training"]["output_dir"] = f"runs/beta_{label}_seed_{seed}"
            train(config, "vae")
            records.append({"seed": seed, "beta": beta, **evaluate(config, "vae")})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
