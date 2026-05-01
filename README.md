# Approx Sampling

Comparison of 17 sampling strategies for neural network function approximation in \(L^2[0,1]\).

## Setup

```bash
git clone https://github.com/lowrank/approx_sampling.git
cd approx_sampling
pip install -r requirements.txt
```

## Quick Test

```bash
python main.py --quick
```

## Full Experiment

```bash
# Train set — 420 generated functions (140 per class × 3)
python scripts/run_full_experiment.py --generated --split train --output-dir results_train

# Test set — 180 generated functions (60 per class × 3)
python scripts/run_full_experiment.py --generated --split test  --output-dir results_test

# Parallel — use all CPU cores
python scripts/run_full_experiment.py --generated --split train --workers 48

# Hardcoded 13 functions only
python scripts/run_full_experiment.py
```

Runs 17 algorithms at budgets **64, 96, 128, 160, 192**. Sobol' only at power-of-two budgets. Device auto-detects CUDA.

## Generate Reports

```bash
# LaTeX paper
PYTHONPATH=. python latex/generate_latex.py
cd latex && pdflatex main.tex && pdflatex main.tex

# mkdocs website
python docs/generate_charts.py
mkdocs build
mkdocs serve   # http://localhost:8000
```

## Custom Model

```bash
python scripts/run_full_experiment.py --model mmnn
python scripts/run_full_experiment.py --model-factory mymod:make_my_net
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--budgets` | 64,96,128,160,192 | Budget sequence (step 32) |
| `--generated` | off | 600 generated L²-normalised functions |
| `--split` | all | train (70%), test (30%), or all |
| `--n-func-per-class` | 200 | Functions per class |
| `--workers` | 1 | Parallel workers (use 48 for full CPU) |
| `--model` | mlp | mlp or mmnn |
| `--model-factory` | — | Custom `pkg.mod:func` |
| `--device` | auto | auto, cpu, or cuda |

## Algorithms

| Family | Methods |
|---|---|
| Fixed-grid | Uniform, Chebyshev, QMC Sobol', QMC Halton |
| Adaptive | Adaptive Residual, Iterative Uniform, Iterative Chebyshev |
| Generative (REINFORCE) | Adversarial, Importance Sampling |
| Generative (reparam) | Normalizing Flow, MDN |
| Generative (score) | Diffusion (score matching + Langevin) |
| Particle | SVGD |
| Uncertainty | Deep Ensemble, GP-UCB |
| Policy / Meta | Policy Gradient, Neural Process |

## Website

[lowrank.github.io/approx_sampling](https://lowrank.github.io/approx_sampling) — interactive charts with toggleable methods.
