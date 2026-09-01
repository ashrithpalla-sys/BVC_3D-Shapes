# Learning Procedural 3D Stools as Point Clouds

A complete Brown Visual Computing starter project that creates a controlled dataset of
procedural stools, learns an unordered point-cloud representation, and generates novel
geometry with a compact variational autoencoder (VAE).

![Procedural training examples](figures/training_samples.png)

## Why this project is distinct

This repository does **not** reproduce the reference SDF implementation. It uses a new
object niche (multi-part stools), explicit surface point sampling, a shared PointNet-style
encoder, and a direct point-set decoder. One network generalizes across every shape; there
is no learned embedding table with one code per training object. Saved procedural parameters
are used only for analysis, never as model inputs.

## Research questions

1. Can a point-cloud autoencoder reconstruct a procedurally varied, multi-part object family?
2. What reconstruction cost comes from regularizing the representation into a sampleable VAE?
3. Do latent interpolations remain smooth, and are procedural factors linearly readable?

## Pipeline

```text
interpretable parameters → procedural parts → normalized point cloud
                                              ↓
                            PointNet encoder → latent code
                                              ↓
                               point decoder → reconstruction / new sample
```

Each stool varies in seat width, depth, thickness, height, leg radius and spread, seat
style, and stretcher presence. The dataset includes points, part labels, parameters,
parameter names, and its seed.

## Quick start

Python 3.10+ is required. PyTorch runs on CPU, Apple Silicon (MPS), or CUDA.

```bash
python3 -m pip install -e ".[dev]"
python3 -m stoolgen generate --config configs/quick.yaml
python3 -m stoolgen train --config configs/quick.yaml --model ae
python3 -m stoolgen train --config configs/quick.yaml --model vae
python3 -m stoolgen evaluate --config configs/quick.yaml --model vae
python3 -m pytest
```

Use `python3 -m stoolgen demo` for a smoke test. The reported submission experiment uses
`configs/submission.yaml`; `configs/full.yaml` is a larger optional extension. Checkpoints
and histories go under `runs/`, and generated datasets go under `data/`. Both are excluded
from Git because every artifact can be recreated from its config and seed.

## Final results

The final experiment uses 1,000 shapes, 384 points per shape, a 24-dimensional latent code,
and 40 epochs. It trained on Apple MPS in 15.6 seconds for the AE and 13.2 seconds for the VAE.

| Model | Test Chamfer ↓ | Latent factor probe R² ↑ |
|---|---:|---:|
| Autoencoder | 0.0230 ± 0.0067 | 0.650 |
| VAE | **0.0223 ± 0.0059** | 0.643 |

All 64 sampled VAE clouds passed the non-collapse check, and 42.2% passed a conservative
coarse stool-structure test requiring a wide upper seat region, a lower supporting region,
and points reaching floor height. Samples have mean pairwise Chamfer 0.0687 and mean nearest-
training Chamfer 0.0673, providing evidence of diversity without exact training copies.
See [the report](docs/WRITEUP.md) and [what I learned](docs/WHAT_I_LEARNED.md) for interpretation.

| Reconstructions | VAE samples |
|---|---|
| ![Input and reconstruction pairs](figures/reconstructions.png) | ![Generated shapes](figures/generated_samples.png) |

### Latent-space behavior

![Interpolation between encoded stools](figures/interpolation.png)

![Traversal through the highest-variance latent coordinate](figures/latent_traversal.png)

## Repository map

```text
configs/                 quick and full reproducible experiments
src/stoolgen/geometry.py procedural part-based geometry
src/stoolgen/data.py     archive creation, validation, splits, augmentation
src/stoolgen/models/     PointNet encoder, AE, VAE, point decoder
src/stoolgen/training.py shared training and checkpointing
src/stoolgen/evaluation.py metrics, interpolation, latent traversal
src/stoolgen/visualization.py dependency-light point rendering
tests/                   geometry, data, model, loss, and rendering tests
docs/                    method details, writeup, and research log
figures/                 committed representative results
```

## Reproducibility and responsible interpretation

Seeds, data sizes, model dimensions, optimizer settings, and evaluation counts live in YAML.
The best validation checkpoint is selected automatically. The held-out linear ridge probe is
descriptive, not evidence of causal disentanglement. Generated validity is a smoke metric,
not mesh-level proof.

The KL-weight experiment is not hypothetical: `scripts/run_beta_ablation.py` evaluates
β ∈ {0, 0.0002, 0.001} over seeds 7, 17, and 29. Its raw and summarized results are committed
under `outputs/`. The experiment shows that stronger regularization improves prior-sample
proximity and diversity while increasing reconstruction error.

## License

MIT. The implementation is original and designed around the project framing above.
