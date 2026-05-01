#!/usr/bin/env python3
"""
Main entry point.

Usage::

    python main.py                                     # default budgets
    python main.py --budgets 64,128,256,512,1024       # custom budget sequence
    python main.py --func smooth_sinusoid
    python main.py --model mmnn
    python main.py --model-factory mypkg.mymod:factory
    python main.py --generated --n-func-per-class 50
    python main.py --quick
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Callable

import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
).MMNN(input_size=1, rank=10, hidden_size=1000, seed=42)


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
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sampling strategy comparison — error-vs-samples curves."
    )
    p.add_argument("--func", type=str, default=None,
                   help="Comma-separated function names.")
    p.add_argument("--budgets", type=str, default="64,128,256,512,1024",
                   help="Comma-separated sampling budgets (default: 64,128,256,512,1024).")
    p.add_argument("--model", type=str, default="mlp",
                   help="Built-in model (mlp, mmnn).")
    p.add_argument("--model-factory", type=str, default=None,
                   help="Custom 'pkg.module:function'.")
    p.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"],
                   help="PyTorch device (default: auto-detect CUDA).")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="Quick test: 2 functions, budgets=64,128,256.")
    p.add_argument("--generated", action="store_true",
                   help="Use generated library.")
    p.add_argument("--n-func-per-class", type=int, default=50,
                   help="Functions per class when --generated.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    funcs = None
    if args.func is not None:
        funcs = [s.strip() for s in args.func.split(",") if s.strip()]

    budgets = [int(x) for x in args.budgets.split(",")]
    factory = _resolve_factory(args.model, args.model_factory)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    if args.quick:
        if funcs is None:
            funcs = ["smooth_sinusoid", "oscillatory_high_freq"]
        budgets = [64, 128, 256]
        print(f"[quick] {len(funcs)} funcs, budgets={budgets}, model={args.model}")

    run_experiment(
        function_names=funcs,
        budgets=budgets,
        model_factory=factory,
        generated=args.generated,
        n_func_per_class=args.n_func_per_class,
        device=device,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(f"\nResults → {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
