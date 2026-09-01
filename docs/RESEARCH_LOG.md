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

## Experiment 4 — Submission-scale model

- **Change:** 1,000 shapes, 384 points, hidden width 192, latent 24, and 40 epochs.
- **Reason:** The original eight-epoch samples were non-collapsed but visually too coarse.
- **Result:** AE Chamfer 0.0230; VAE Chamfer 0.0223; probe R² approximately 0.65.
- **Generation:** 64/64 non-collapsed, 27/64 passed the conservative structural rule.
- **Observation:** Seats and legs became recognizable, but some supports remain incomplete.

## Experiment 5 — KL-weight ablation

- **Question:** How does β affect reconstruction, diversity, and prior-sample plausibility?
- **Method:** β = 0, 0.0002, and 0.001 across seeds 7, 17, and 29.
- **Result:** Test Chamfer rises from 0.0463 to 0.0527 as β increases. Pairwise sample
  Chamfer rises from 0.0236 to 0.0500, while nearest-training Chamfer falls from 0.1525 to
  0.0565.
- **Interpretation:** Stronger regularization sacrifices some reconstruction fidelity but
  produces a more useful sampling distribution. The middle β is a reasonable compromise.
