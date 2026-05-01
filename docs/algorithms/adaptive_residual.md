# Adaptive Residual Sampling

## How It Works

Begins with a coarse uniform grid over the domain. An approximator (e.g. neural network) is trained on this initial set. After training, the residual error is evaluated at a dense set of candidate points. The top-k points with the largest absolute error are added to the training set, and the approximator is retrained. This greedy refinement loop repeats until a budget or error threshold is met.

## Pros

- Quasi-optimal convergence for functions with localised features such as shocks, kinks, or boundary layers.
- Simple to implement — requires only error evaluation and retraining, no auxiliary models.
- Monotonically decreasing error with each refinement step.
- Effective across diverse function classes (polynomial, trigonometric, discontinuous).

## Cons

- Computationally expensive per iteration due to full retraining.
- Requires dense candidate grid; misses features between grid points.
- Greedy selection can lead to redundant collocation in clusters.
- No mechanism to forget or down-weight stale points from earlier iterations.

## Reference

Binev, P., Cohen, A., Dahmen, W., DeVore, R., Petrova, G., & Wojtaszczyk, P. (2011). Convergence rates for greedy algorithms in reduced basis methods. *SIAM Journal on Mathematical Analysis*, 43(3), 1457–1472.
