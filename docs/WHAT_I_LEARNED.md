# What I Learned

Before this project, I understood autoencoders mostly as models for fixed-size vectors and
images. The most important thing I learned was that a point cloud changes the problem in a
fundamental way: the rows have no meaningful order. A network cannot treat point 20 as if it
always represents the same physical location. Using the same small network on every point and
then max-pooling the features gave me a concrete understanding of permutation invariance.

I also learned why an ordinary coordinate-by-coordinate loss is wrong for unordered geometry.
Two point clouds can describe the same surface while listing their points in completely
different orders. Chamfer distance solves that mismatch by finding each point's nearest
neighbor in the other cloud. Implementing it made the computational cost of 3D learning much
more tangible: doubling the number of points makes the pairwise distance work grow roughly
quadratically.

Building the procedural generator was as important as training the model. I had to decide
which variations made a stool still look like a member of the same family, how to keep legs
attached to the seat, how many samples to allocate to thin parts, and how to normalize shapes
without destroying proportion information. I initially lost a few points because integer
division did not allocate the leg budget evenly across four legs. A dataset test exposed that
bug, and I changed the generator to fill the remainder deterministically.

The autoencoder/VAE comparison taught me that reconstruction and generation are different
goals. An autoencoder can place encoded examples wherever it wants in latent space. A VAE pays
an extra KL penalty so that random normal samples land in regions the decoder understands.
When I tried the first small run, the samples were nonempty but only coarsely stool-like. More
data, a larger network, longer training, and a gradual KL warmup made the parts substantially
clearer. The final generated examples show recognizable seats and supporting legs, although
thin structures remain the hardest feature.

I made one evaluation mistake that changed how I think about experimental claims. My first
latent probe used fewer test examples than latent dimensions and reported R² = 1.0. That was
not meaningful; the regression was underdetermined. I replaced it with a regularized ridge
probe trained on 70% of all encoded examples and evaluated on the remaining 30%. The final R²
of about 0.64–0.65 is less dramatic but defensible. Recording this correction in the research
log was more useful than silently presenting the original number.

Finally, the three-seed β experiment showed me why one metric is not enough. Stronger KL
regularization made random samples closer to the training distribution and more diverse, but
it increased reconstruction error. Even the structural validity check has limits: it can
detect a wide top and lower supports, but it cannot prove that a stool is connected, stable,
or watertight. If I continued, I would add a part-aware decoder or convert outputs to meshes so
that I could measure connectivity and physical support rather than only point-set similarity.

The project left me with a much clearer view of the full research loop: define a controlled
data distribution, choose a representation that respects its symmetries, build a baseline,
test a hypothesis, find where the metric is misleading, and revise both the experiment and
the claim.
