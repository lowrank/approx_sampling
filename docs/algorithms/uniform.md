# Uniform Sampling

Equispaced points on \([0,1]\):  \(x_i = \frac{i-1}{N-1}\),  \(i=1,\dots,N\).

## How It Works

No adaptivity — the grid is fixed regardless of the target function. Training uses
standard minibatch SGD on the MSE loss evaluated at all \(N\) points for many epochs.

## Pros

* Simplest possible strategy, zero overhead.
* Optimal when the error density \(|u_\theta - f|^2\) is approximately uniform.
* Works well for smooth, slowly-varying functions.

## Cons

* Wastes budget in flat/simple regions.
* May miss narrow features (Gaussian bumps, boundary layers) if \(N\) is too small
  to place a point within the feature.
* Cannot adapt to discovered structure.

## Reference

Davis & Rabinowitz, *Methods of Numerical Integration*, Dover, 2007.
