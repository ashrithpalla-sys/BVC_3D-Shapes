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
- **Latent probe:** ridge regression fitted on 70% of encoded examples and scored on 30%.
- **Qualitative:** training examples, reconstructions, prior samples, interpolation, traversal.

The probe is a readability diagnostic, not evidence of disentanglement. Procedural metadata
is returned for analysis but never consumed by either model. Evaluation never updates weights.

