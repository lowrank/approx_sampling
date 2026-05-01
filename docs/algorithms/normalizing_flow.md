# Normalizing Flow Sampling

## How It Works

A piecewise-linear monotonic cumulative distribution function (CDF) is constructed via cumulative sums of softplus-activated height parameters. Sampling proceeds by drawing uniform random variates, then applying the inverse CDF transform. Because the forward mapping is a differentiable bijection, the reparameterisation trick yields low-variance gradient estimates for training the flow parameters. The flow is optimised to shape the sampling density toward high-error regions.

## Pros

- Exact likelihood evaluation via change-of-variables formula.
- Reparameterisation gradient has low variance compared to REINFORCE.
- Piecewise-linear CDF is simple, fast to invert, and provably monotonic.
- Inherently produces valid probability distributions without normalisation concerns.

## Cons

- Piecewise-linear CDF has limited expressiveness — cannot model multi-modal densities well.
- Requires uniform knot placement or heuristic knot adaptation.
- Inverse-CDF sampling is sequential and cannot be parallelised trivially.
- Limited to 1D domains in this formulation; multivariate extension requires coupling layers.

## Reference

Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*.
