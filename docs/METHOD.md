# Method

## Geometry and dataset

A stool is assembled from independently sampled but constrained parameters. Seats are box
or elliptical-cylinder surfaces. Four cylindrical legs attach below the seat, and a subset
of shapes contains two cylindrical stretchers. Fixed part sampling ratios allocate 35% of
points to the seat, 55% to legs, and 10% to supports. Unused support samples return to the
seat. Every cloud is centered vertically and normalized by total object height.

Direct surface sampling keeps installation lightweight and every dataset build deterministic.
Its limitation is that point density is controlled by part ratios, not exact surface area.

## Representation

An input is an unordered `N × 3` set. A shared multilayer perceptron, implemented as 1×1
convolutions, maps points into features. Symmetric max pooling creates a permutation-invariant
global descriptor. The decoder maps that descriptor into `N × 3` coordinates. Chamfer
distance makes training insensitive to point order.

The autoencoder maps directly to `z`. The VAE predicts `mu` and `log(var)`, samples through
the reparameterization trick, and adds beta-weighted KL divergence:

```text
loss = symmetric squared Chamfer(reconstruction, input) + beta * KL(q(z|x) || N(0,I))
```

## Evaluation

- **Reconstruction:** per-shape Chamfer distance on the held-out test split.
- **Generation smoke check:** fraction of samples with nontrivial extent on all axes.
- **Coarse structural validity:** presence of a wide upper region, lower supports, and floor-
  reaching points. This is deliberately named as a heuristic rather than mesh validity.
- **Diversity and novelty:** mean pairwise generated Chamfer and nearest-training Chamfer.
- **Latent probe:** ridge regression fitted on 70% of encoded examples and scored on 30%.
- **Qualitative:** training examples, reconstructions, prior samples, interpolation, traversal.

The probe is a readability diagnostic, not evidence of disentanglement. Procedural metadata
is returned for analysis but never consumed by either model. Evaluation never updates weights.

## Final reproducibility settings

The submission config contains every final setting: 1,000 shapes, 384 points, seed 7, 40
epochs, and a 12-epoch KL warmup. Each training run records device, Python and PyTorch versions,
parameter count, elapsed time, and best validation loss. The β experiment holds architecture
and data size fixed within each seed and repeats β ∈ {0, 0.0002, 0.001} over three seeds.
