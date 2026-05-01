# Stein Variational Gradient Descent Sampling

## How It Works

A set of particles is placed in the domain and evolves via a kernelised gradient flow toward an error-weighted target density. In each step, particles are driven by two forces: a driving term pushing them toward high-error regions, and a repulsive term (via the Stein operator with an RBF kernel) preventing particle collapse. No generator network is needed — the particles themselves form the empirical sampling distribution. The approximator is trained on the converged particle locations.

## Pros

- No generator network to train — particles directly represent the sampling distribution.
- Kernelised repulsion ensures diverse, well-spread samples.
- Deterministic update avoids the high variance of REINFORCE.
- Converges to the target distribution under appropriate kernel and step size conditions.

## Cons

- Computational cost scales quadratically with particle count due to kernel matrix.
- Particle count must be decided upfront and cannot adapt online.
- Kernel bandwidth is a sensitive hyperparameter.
- Convergence can be slow for high-dimensional or sharply peaked target densities.

## Reference

Liu, Q., & Wang, D. (2016). Stein variational gradient descent: A general purpose Bayesian inference algorithm. *Advances in Neural Information Processing Systems (NeurIPS)*, 29.
