# Research Log

## Experiment 1 — End-to-end smoke run

- **Hypothesis:** A small PointNet autoencoder can learn the coarse stool family on CPU.
- **Configuration:** 160 shapes, 256 points, latent 16, hidden 96, 8 epochs, seed 7.
- **Result:** AE test Chamfer 0.0632; loss decreased throughout the run.
- **Observation:** Broad structure reconstructs, while thin legs remain less crisp.

## Experiment 2 — Variational representation

- **Hypothesis:** KL regularization worsens reconstruction but enables prior samples.
- **Result:** VAE Chamfer 0.0784; 24/24 samples passed the extent smoke check.
- **Observation:** The expected fidelity/sampleability tradeoff is visible.

## Experiment 3 — Latent diagnostics

- **Method:** Held-out ridge probe, endpoint interpolation, and variance traversal.
- **Result:** Probe R² is 0.345 (AE) and 0.360 (VAE); transitions remain non-collapsed.
- **Correction:** An initial underdetermined in-sample probe misleadingly reported 1.0. The
  final probe uses the full archive with a regularized 70/30 held-out split.

