"""
Discarded: SVGD and Ensemble methods
=====================================

## SVGD (Stein Variational Gradient Descent)

**Why discarded**: Consistently worst performer.  On spike_bg_demo at
budget 192: L² = 0.75 (vs. best method 0.0018, uniform 0.05).

**Diagnosis**: SVGD requires the gradient of log p*(x) where p* ∝ |f_θ − f|².
This gradient is ∇f_θ(x) / (f_θ(x) − f(x)), which is unstable when the
approximation error is small or the network is poorly trained.  The
particle-based approach works well for Bayesian inference but struggles
for this deterministic error-minimisation setting where the target
density changes every round.

## Ensemble (Deep Ensemble)

**Why discarded**: Second-worst performer.  On spike_bg_demo at budget
192: L² = 0.19 (vs. best 0.0018, uniform 0.05).

**Diagnosis**: With 5 ensemble members each seeing only 192/batch_size ≈ 3
batches of data, the predictive variance is poorly estimated.  The variance
at untrained regions is dominated by random initialisation noise rather than
true epistemic uncertainty.  More members and more data per member would
help, but the computational cost (5× training) makes it impractical.
"""
