# Approx Sampling

Comparison of **15+ sampling strategies** for neural network function approximation
in \(L^2[0,1]\) under a fixed evaluation budget.

## Quick Start

```bash
# Run a quick test (2 functions, 2 budgets)
python main.py --quick

# Run with generated function library
python main.py --generated --n-func-per-class 50

# Use a different model architecture
python main.py --model mmnn

# Use your own model factory
python main.py --model-factory my_pkg.my_mod:make_model

# Run a specific task
python main.py --task function_approximation --func smooth_sinusoid --budgets 64,128,256,512
```

## Key Idea

Given a target function \(f : [0,1] \to \mathbb{R}\) known only through pointwise queries,
and a total budget of \(N\) function evaluations,
**where should we sample** so that training a small neural network \(u_\theta\) on those
\(N\) points minimises

\[
\mathcal{L}(\theta) = \int_0^1 \bigl(u_\theta(x) - f(x)\bigr)^2\,dx \; ?
\]

The answer depends on the function's structure — smooth regions need few points,
oscillatory or sharply-varying regions need many.
The **sampling strategy** determines how the finite budget is allocated across \([0,1]\).

## Project Structure

| Directory | Purpose |
|---|---|
| `algorithms/` | 15+ sampling strategies (empirical, adaptive, generative, particle, uncertainty, policy) |
| `tasks/` | Downstream problem definitions (function approx, PINN, Deep Ritz) |
| `data/` | Test-function library: 13 handcrafted + 600 generated \(L^2\)-normalised functions |
| `models/` | Approximator architectures (MLP, MMNN) — user-supplied |
| `experiments/` | Multi-budget evaluation pipeline, error-vs-samples curves |
| `latex/` | Full LaTeX paper with theory, references, and guarantees |
| `docs/` | This documentation site |

## Algorithms at a Glance

| Family | Algorithms |
|---|---|
| **Fixed-grid** | Uniform, Chebyshev, QMC Sobol', QMC Halton |
| **Adaptive** | Adaptive Residual, Iterative Uniform, Iterative Chebyshev |
| **Generative (REINFORCE)** | Adversarial, Importance Sampling |
| **Generative (reparam)** | Normalizing Flow, MDN |
| **Generative (score)** | Diffusion (score matching + Langevin) |
| **Particle** | SVGD |
| **Uncertainty** | Deep Ensemble, GP-UCB |
| **Policy** | Policy Gradient, Neural Process |
