#!/usr/bin/env python3
"""
Full experiment runner.

For every function in the (hardcoded or generated) library, runs all 17
sampling algorithms at increasing budgets starting from 64.

QMC Sobol' is only evaluated at power-of-two budgets.  Results saved as
``results/curves.json``.

Usage::

    python scripts/run_full_experiment.py                         # 13 hardcoded funcs
    python scripts/run_full_experiment.py --generated             # 600 generated funcs
    python scripts/run_full_experiment.py --generated --split train
    python scripts/run_full_experiment.py --model mmnn            # MMNN arch
    python scripts/run_full_experiment.py --device cuda           # GPU
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Callable, List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import run_experiment

# ---------------------------------------------------------------------------
# Built-in model factories
# ---------------------------------------------------------------------------

_BUILTIN: dict[str, Callable[[], nn.Module]] = {}

_BUILTIN["mlp"] = lambda: __import__(
    "models.approximator", fromlist=["MLP"]
).MLP(hidden_dims=[32, 32, 32, 32], activation=nn.Tanh())

_BUILTIN["mmnn"] = lambda: __import__(
    "models.approximator", fromlist=["MMNN"]
).MMNN(input_size=1, rank=5, hidden_size=200, seed=42)


def _resolve_factory(model: str, mf: str | None) -> Callable[[], nn.Module]:
    if mf is not None:
        parts = mf.split(":")
        if len(parts) != 2:
            raise ValueError(f"Expected 'pkg.module:func', got '{mf}'")
        mod = importlib.import_module(parts[0])
        factory = getattr(mod, parts[1])
        if not callable(factory):
            raise TypeError(f"{parts[1]} is not callable")
        return factory
    if model not in _BUILTIN:
        raise ValueError(f"Unknown --model '{model}'. Available: mlp, mmnn")
    return _BUILTIN[model]


# ---------------------------------------------------------------------------
# Budget schedule
# ---------------------------------------------------------------------------


def make_budgets(start: int = 64, max_budget: int = 196, step: int = 32) -> List[int]:
    budgets: List[int] = []
    b = start
    while b <= max_budget:
        budgets.append(b)
        b += step
    return budgets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full experiment: all functions × 17 algorithms × progressive budgets."
    )
    p.add_argument("--func", type=str, default=None,
                   help="Comma-separated function names (default: all).")
    p.add_argument("--max-budget", type=int, default=196,
                   help="Largest sampling budget (default: 196, step 32 → 64,96,128,160,192).")
    p.add_argument("--model", type=str, default="mlp",
                   help="Built-in model (mlp, mmnn).")
    p.add_argument("--model-factory", type=str, default=None,
                   help="Custom 'pkg.module:function'.")
    p.add_argument("--device", type=str, default="auto",
                   choices=["cpu", "cuda", "auto"],
                   help="PyTorch device (default: auto-detect CUDA).")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generated", action="store_true",
                   help="Use the generated 600-function library.")
    p.add_argument("--n-func-per-class", type=int, default=200,
                   help="Functions per class when --generated.")
    p.add_argument("--split", type=str, default="all",
                   choices=["all", "train", "test"],
                   help="Use train (70%%), test (30%%), or all functions.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    funcs: Optional[List[str]] = None
    if args.func is not None:
        funcs = [s.strip() for s in args.func.split(",") if s.strip()]

    budgets = make_budgets(max_budget=args.max_budget, step=32)
    factory = _resolve_factory(args.model, args.model_factory)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    n_funcs = len(funcs) if funcs else (
        args.n_func_per_class * 3 if args.generated else 13
    )

    print(f"{'='*70}")
    print(f"Full experiment")
    print(f"  Functions:     {n_funcs}")
    print(f"  Budgets:       {len(budgets)} ({budgets[0]}…{budgets[-1]}, step 32)")
    print(f"  Algorithms:    17")
    print(f"  Device:        {device}")
    print(f"  Output:        {args.output_dir}")
    print(f"{'='*70}\n")

    run_experiment(
        function_names=funcs,
        budgets=budgets,
        model_factory=factory,
        generated=args.generated,
        n_func_per_class=args.n_func_per_class,
        split=args.split,
        device=device,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    curves_path = os.path.join(args.output_dir, "curves.json")
    print(f"\nResults → {os.path.abspath(curves_path)}")

    # Generate chart data for the website
    import json
    if os.path.exists(curves_path):
        with open(curves_path) as f:
            data = json.load(f)
        sampled: dict = {}
        for cls in ("smooth", "oscillatory", "sharp"):
            fns = sorted([k for k in data if k.startswith(cls)])
            for fn in fns[:3]:
                if fn in data:
                    sampled[fn] = data[fn]
        for fn in data:
            if not any(fn.startswith(p) for p in ("smooth_", "oscillatory_", "sharp_")):
                sampled[fn] = data[fn]
        chart_data_path = "docs/figures/charts_data.json"
        os.makedirs(os.path.dirname(chart_data_path), exist_ok=True)
        with open(chart_data_path, "w") as f:
            json.dump(sampled, f)
        print(f"Chart data → {os.path.abspath(chart_data_path)}")


if __name__ == "__main__":
    main()
