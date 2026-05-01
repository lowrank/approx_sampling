"""
Discarded: Energy-Based Adaptive Sampling
==========================================

**Why discarded**

The energy-based algorithm samples proportionally to a local difficulty metric
combining squared approximation error and derivative mismatch:

    difficulty(x) = |f(x) - f_θ(x)|² + α·|f'(x) - f_θ'(x)|²

While theoretically motivated by the Deep Ritz energy functional, the method
consistently underperformed in practice on pure function approximation tasks.

**Empirical record (budget = 300, seed = 42)**

| Function             | L² error (energy-based) | Best competitor        | Best L²   |
|----------------------|-------------------------|------------------------|-----------|
| smooth_sinusoid      | 0.095                   | qmc_sobol              | 0.000079  |
| oscillatory_high_freq| 0.782                   | importance_sampling    | 0.035     |
| local_gaussians      | 0.234                   | qmc_halton             | 0.001     |

**Diagnosis**

1. **Derivative mismatch noise**: The gradient of an untrained/partially-trained
   network is essentially random, so the `α·|f' - f_θ'|²` term injects noise
   that dominates the error signal, especially in early rounds.
2. **Bias toward smooth regions**: Smooth functions have small derivatives
   everywhere, so the difficulty metric collapses to just the pointwise error
   — which is also small for smooth functions. The method then fails to
   concentrate samples effectively.
3. **Sensitivity to α**: The derivative weight `α_deriv` is difficult to tune
   per function. Small α makes the method equivalent to adaptive residual
   (but with histogram discretisation overhead); large α overweights derivative
   mismatch.
4. **Piecewise-constant discretisation**: Using a fixed 32-bin histogram limits
   the spatial resolution of the sampling distribution. For functions with
   very narrow features (e.g., Gaussian bumps), the best sampling distribution
   is not representable as a coarse histogram.

**Potential fixes (not pursued)**

- Use α = 0 (pure error-based) — this reduces to a less efficient version of
  adaptive residual refinement.
- Use a learned α schedule or per-bin α values.
- Replace histogram with a continuous generative model (which adversarial
  and importance-sampling methods already do).
- Pre-train the model uniformly for several rounds before switching to
  energy-based sampling (warm-start).

**Conclusion**

For pure function approximation in L², the adaptive residual method provides
the error-guided sampling benefit without the noise from derivative terms, and
the adversarial/importance-sampling methods provide a principled, trainable
alternative. The energy-based method may be more suitable for PDE settings
(Deep Ritz, PINNs) where the energy functional itself provides the training
signal, rather than for standalone function approximation.
"""
