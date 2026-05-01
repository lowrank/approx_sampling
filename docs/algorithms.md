# Algorithms

All algorithms share a common interface: given a `DownstreamTask`, a sampling `budget`,
and an evaluation grid, they return an `AlgorithmResult`.

## Empirical / Fixed-Grid

- [Uniform](algorithms/uniform.md) — equispaced grid.
- [Chebyshev](algorithms/chebyshev.md) — Chebyshev nodes mapped to \([0,1]\).
- [QMC Sobol'](algorithms/qmc_sobol.md) — scrambled Sobol' (power-of-two only).
- [QMC Halton](algorithms/qmc_halton.md) — scrambled Halton (any \(N\)).

## Adaptive

- [Adaptive Residual](algorithms/adaptive_residual.md) — greedy error-based refinement.
- [Iterative Uniform](algorithms/iter_uniform.md) — uniform base + error refinement.
- [Iterative Chebyshev](algorithms/iter_chebyshev.md) — Chebyshev base + error refinement.

## Generative (REINFORCE)

- [Adversarial](algorithms/adversarial.md) — min-max via score-function estimator.
- [Importance Sampling](algorithms/importance_sampling.md) — IS-weighted MSE + REINFORCE.

## Generative (Reparameterisation)

- [Normalizing Flow](algorithms/normalizing_flow.md) — monotonic CDF, inverse-CDF sampling.
- [MDN](algorithms/mdn.md) — Gaussian mixture, Gumbel-Softmax.

## Generative (Score-Based)

- [Diffusion](algorithms/diffusion.md) — denoising score matching + Langevin dynamics.

## Uncertainty-Based

- [GP-UCB](algorithms/gp_ucb.md) — Gaussian Process Upper Confidence Bound.

## Policy / Meta

- [Policy Gradient](algorithms/policy.md) — error-histogram → sampling distribution.
- [Neural Process](algorithms/neural_process.md) — context-set → sampling distribution.
