# Iterative Uniform + Adaptive Refinement

## How It Works

A two-phase meta-strategy. In the first phase, 40% of the total sample budget is drawn uniformly at random across the domain and the approximator is trained. In the second phase, the remaining 60% of points are added incrementally via error-guided refinement: after each retraining, points with the highest residual error are selected and appended to the training set. This balances initial exploration with targeted exploitation.

## Pros

- Robust initial coverage via uniform sampling guards against missing high-error regions.
- Adaptive second phase concentrates samples where the function is hardest to learn.
- Simple hyperparameter (40/60 split) with intuitive behaviour.
- No auxiliary generative or surrogate model required.

## Cons

- Uniform phase wastes samples in flat, easy-to-approximate regions.
- Fixed 40/60 split may be suboptimal for functions with extreme localisation.
- Full retraining at each refinement step is costly.
- Does not adapt the ratio online based on approximation progress.

## Reference

Binev, P., Cohen, A., Dahmen, W., DeVore, R., Petrova, G., & Wojtaszczyk, P. (2011). Convergence rates for greedy algorithms in reduced basis methods. *SIAM Journal on Mathematical Analysis*, 43(3), 1457–1472.
