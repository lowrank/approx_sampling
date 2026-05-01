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

# Or the 13 hardcoded functions (no train/test split)
python scripts/run_full_experiment.py
```

This runs 17 algorithms at budgets 64, 96, 128, 160, 192.  
Sobol' only runs at power-of-two budgets. Results saved as `curves.json`.

## Generate Reports

```bash
# LaTeX paper
PYTHONPATH=. python latex/generate_latex.py
cd latex && pdflatex main.tex && pdflatex main.tex

# mkdocs website (also deployed to gh-pages)
PYTHONPATH=. python docs/generate_charts.py
mkdocs build
mkdocs serve   # preview at http://localhost:8000
```

## Custom Model

```bash
python scripts/run_full_experiment.py --model mmnn
python scripts/run_full_experiment.py --model-factory mymod:make_my_net
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--budgets` | 64,96,128,160,192 | Budget sequence |
| `--generated` | off | Use 600 generated functions |
| `--split` | all | train (70%), test (30%), or all |
| `--n-func-per-class` | 200 | Functions per class |
| `--model` | mlp | mlp or mmnn |
| `--model-factory` | — | Custom `pkg.mod:func` |
| `--device` | cpu | cpu or cuda |
