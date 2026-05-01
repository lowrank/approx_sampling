# Iterative Chebyshev + Adaptive Refinement

## How It Works

A two-phase meta-strategy similar to iterative uniform refinement, but replaces the initial 40% uniform phase with Chebyshev-node sampling. Chebyshev nodes cluster near domain boundaries, providing excellent interpolation stability for smooth functions. After training on these nodes, the remaining 60% of the budget is allocated via error-guided refinement, selecting points with the largest residual and retraining.

## Pros

- Chebyshev initialisation suppresses Runge phenomenon and improves boundary accuracy.
- Combines spectral convergence benefits of Chebyshev nodes with adaptive local refinement.
- Better initial approximation than uniform for smooth target functions.
- No auxiliary generative or surrogate model needed.

## Cons

- Chebyshev clustering at boundaries wastes samples for functions whose difficulty lies in the interior.
- Fixed 40/60 split cannot adapt to function characteristics online.
- Full retraining at each iteration is computationally expensive.
- Poor for functions with interior discontinuities far from boundaries.

## Reference

Binev, P., Cohen, A., Dahmen, W., DeVore, R., Petrova, G., & Wojtaszczyk, P. (2011). Convergence rates for greedy algorithms in reduced basis methods. *SIAM Journal on Mathematical Analysis*, 43(3), 1457–1472.
