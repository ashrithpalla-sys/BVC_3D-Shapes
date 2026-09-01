# Learning a Generative Representation of Procedural 3D Stools

## Abstract

This project studies whether a compact neural model can learn the distribution of a
procedurally generated family of multi-part 3D stools. I built an original parameterized
surface sampler, represented each object as a normalized point cloud, and compared a
PointNet-style deterministic autoencoder with a variational autoencoder. A small reproducible
CPU experiment demonstrates working reconstruction, sampling, interpolation, and latent
analysis. The deterministic model achieves lower Chamfer error, while VAE regularization
produces a sampleable space at a measurable reconstruction cost.

## Motivation

Procedural programs expose meaningful controls but require the original rules to create a new
object. Generative models instead attempt to learn the distribution from examples. Stools are
a useful test category: they share clear semantic structure, while continuous proportions and
discrete seat/support choices create visible variation and multi-part topology.

## Experimental setup

The smoke experiment uses 160 stools with 256 surface points each, an 80/10/10 split, a
16-dimensional latent code, hidden width 96, batch size 16, and eight Adam epochs. Both models
use the same encoder/decoder capacity and symmetric squared Chamfer objective. The VAE uses
`beta = 0.0005`. All committed results use seed 7.

## Results

Random training examples show the intended distribution includes tall and short stools,
different seat footprints, square and round seats, varying legs, and optional supports.

![Random procedural training shapes](../figures/training_samples.png)

The deterministic baseline obtains test Chamfer **0.0632 ± 0.0175**, compared with
**0.0784 ± 0.0237** for the VAE. This expected gap reflects the cost of constraining latent
codes toward a common Gaussian prior. Training curves show both models improve rapidly in the
small run without numerical instability.

![Autoencoder loss](../figures/ae_loss.png)

![VAE loss](../figures/vae_loss.png)

Input/reconstruction pairs retain the broad seat-and-leg arrangement, although thin structures
remain challenging at only 256 output points.

![Held-out input and reconstruction pairs](../figures/reconstructions.png)

Random prior samples are spatially non-collapsed: all 24 tested outputs have nontrivial extent
along every axis. This check is intentionally weak; visual inspection shows that an eight-epoch
model learns a coarse family rather than crisp final geometry.

![Random VAE samples](../figures/generated_samples.png)

Linear interpolation between two encoded test objects changes coordinates continuously and
does not exhibit abrupt cloud collapse. Traversing the most variable coordinate provides a
second controlled view of model sensitivity.

![Latent interpolation](../figures/interpolation.png)

![Latent traversal](../figures/latent_traversal.png)

The held-out latent-to-parameter ridge probe reaches R² **0.345** for the AE and **0.360** for
the VAE, suggesting that some procedural information remains linearly accessible. This does
not establish independent or causal latent factors.

## Limitations

The committed experiment is deliberately small and should be treated as an end-to-end proof,
not a converged comparison. Chamfer distance can reward average-looking point sets and does not
ensure part connectivity. Fixed part sampling is not uniform over surface area. The validity
metric detects collapse but cannot verify four legs, stable contact, or watertightness. Finally,
only one seed is reported.

## Next experiments

The strongest follow-up is a multi-seed latent-dimension study using `configs/full.yaml`, with
confidence intervals and structural validity checks derived from part segmentation. A
conditional VAE could receive known procedural factors and test controllable generation. An
atlas- or folding-based decoder may preserve thin legs better than the fully connected decoder.

## Conclusion

The project establishes an original and reproducible procedural-to-learned 3D pipeline. It
also demonstrates a central tradeoff: the deterministic representation reconstructs held-out
examples more accurately, whereas the VAE enables prior sampling and smooth latent experiments.
The gap between coarse validity and structural correctness is the most useful next direction.
