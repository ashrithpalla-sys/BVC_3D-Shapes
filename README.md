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

Use `python3 -m stoolgen demo` for a single smoke-to-results command. For a serious run,
replace `configs/quick.yaml` with `configs/full.yaml`. Checkpoints and histories go under
`runs/`; dataset archives go under `data/`. Both are excluded from Git.

## Baseline results

The committed CPU experiment uses 160 shapes, 256 points, eight epochs, and seed 7. It
verifies the pipeline rather than claiming state-of-the-art quality.

| Model | Test Chamfer ↓ | Latent factor probe R² ↑ |
|---|---:|---:|
| Autoencoder | 0.0632 ± 0.0175 | 0.345 |
| VAE | 0.0784 ± 0.0237 | 0.360 |

All 24 sampled VAE clouds passed the basic non-collapse extent check. The AE reconstructs
better, while the VAE trades fidelity for a regularized, sampleable latent space. See
[the full writeup](docs/WRITEUP.md) for interpretation and limitations.

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

## License

MIT. The implementation is original and designed around the project framing above.
