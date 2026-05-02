"""
Experiment runner — multi-budget evaluation.

For each (function, algorithm) pair the pipeline runs at a sequence of
increasing sampling budgets, producing L^2-error-vs-#samples curves.
Results are saved as JSON and per-function convergence plots are
generated.

The QMC Sobol' algorithm is only evaluated at power-of-two budgets
(others skip those entries for Sobol').
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from algorithms.adversarial import AdversarialSampling
from algorithms.adaptive_residual import AdaptiveResidualSampling
from algorithms.iterative_refinement import IterativeRefinementSampling
from algorithms.base import AlgorithmResult
from algorithms.chebyshev import ChebyshevSampling
from algorithms.diffusion import DiffusionSampling, ScoreNetwork1D
from algorithms.importance_sampling import ImportanceSampling
from algorithms.normalizing_flow import MonotonicFlow1D, NormalizingFlowSampling
from algorithms.mdn import MDNSampler, MDNSampling
from algorithms.policy_sampler import PolicyNetwork, PolicySampling
from algorithms.neural_process import NeuralProcessSampler, NeuralProcessSampling
from algorithms.qmc import QMCSampling
from algorithms.uniform import UniformSampling
from data.function_library import FUNCTION_LIBRARY
from models.approximator import PiecewiseConstantGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ---------------------------------------------------------------------------
# Algorithm builder
# ---------------------------------------------------------------------------


def _build_algorithms(
    budget: int,
    model_factory: Callable[[], nn.Module],
    device: str,
    skip_power_of_two: bool = False,
    lr: float = 1.5e-3,
) -> List[Tuple[str, Any]]:
    """Return list of (name, algorithm_instance) for a given *budget*.

    If *skip_power_of_two* is True, algorithms that only make sense at
    power-of-two budgets (currently ``qmc_sobol``) are omitted.
    """
    algs: List[Tuple[str, Any]] = []
    TARGET = 40000  # target gradient steps for all methods
    batch = 32
    n_rounds_est = max(1, budget // batch)
    batches_per_epoch = max(1, budget // batch)

    def _epochs(target: int = TARGET) -> int:
        return max(50, target // batches_per_epoch)

    # --- Empirical ---
    algs.append(("uniform", UniformSampling(
        budget=budget, model=model_factory(),
        epochs=_epochs(), batch_size=batch, lr=lr, device=device,
    )))
    algs.append(("chebyshev", ChebyshevSampling(
        budget=budget, model=model_factory(),
        epochs=_epochs(), batch_size=batch, lr=lr, device=device,
    )))
    if not skip_power_of_two:
        algs.append(("qmc_sobol", QMCSampling(
            budget=budget, model=model_factory(),
            sequence="sobol", epochs=_epochs(), batch_size=batch, lr=lr, device=device,
        )))
    algs.append(("qmc_halton", QMCSampling(
        budget=budget, model=model_factory(),
        sequence="halton", epochs=_epochs(), batch_size=batch, lr=lr, device=device,
    )))
    algs.append(("adaptive_residual", AdaptiveResidualSampling(
        budget=budget, model=model_factory(),
        n_initial=min(budget, max(20, budget // 4)),
        n_add=min(50, max(10, budget // 8)),
        candidate_size=2000, epochs_per_round=_epochs(TARGET//4), final_epochs=_epochs(TARGET//2),
        batch_size=batch, lr=lr, device=device,
    )))

    # --- Learnable: REINFORCE-based ---
    algs.append(("adversarial", AdversarialSampling(
        budget=budget, model=model_factory(),
        generator=PiecewiseConstantGenerator(32),
        batch_size=batch, total_theta_steps=8000, n_phi_steps_per_outer=20,
        lr_theta=1e-3, lr_phi=1e-2,
        entropy_weight=0.05, baseline_momentum=0.9, device=device,
    )))
    algs.append(("importance_sampling", ImportanceSampling(
        budget=budget, model=model_factory(),
        proposal=PiecewiseConstantGenerator(32),
        batch_size=batch, total_theta_steps=8000, n_phi_steps_per_outer=20,
        lr_theta=1e-3, lr_phi=1e-2,
        entropy_weight=0.05, baseline_momentum=0.9, device=device,
    )))

    # --- Learnable: reparameterisation-based ---
    algs.append(("normalizing_flow", NormalizingFlowSampling(
        budget=budget, model=model_factory(),
        flow=MonotonicFlow1D(64),
        batch_size=batch, total_theta_steps=8000, n_flow_steps_per_outer=30,
        lr_theta=1e-3, lr_flow=1e-3,
        entropy_weight=0.02, device=device,
    )))
    algs.append(("mdn", MDNSampling(
        budget=budget, model=model_factory(),
        mdn=MDNSampler(n_components=8),
        batch_size=batch, total_theta_steps=8000, n_mdn_steps_per_outer=30,
        lr_theta=1e-3, lr_mdn=1e-3,
        entropy_weight=0.02, device=device,
    )))

    # --- Policy ---
    algs.append(("policy", PolicySampling(
        budget=budget, model=model_factory(),
        policy=PolicyNetwork(n_bins=32, hidden_dim=64),
        n_bins=32, batch_size=batch, total_theta_steps=8000,
        n_policy_steps_per_outer=10,
        lr_theta=1e-3, lr_policy=1e-3, device=device,
    )))

    # --- Neural Process (amortised, conditional on observed data) ---
    algs.append(("neural_process", NeuralProcessSampling(
        budget=budget, model=model_factory(),
        np_sampler=NeuralProcessSampler(n_bins=32, dim_h=128, dim_r=128),
        batch_size=batch, total_theta_steps=8000, n_np_steps_per_outer=20,
        lr_theta=1e-3, lr_np=1e-3,
        entropy_weight=0.05, initial_random=20, device=device,
    )))

    # --- Diffusion / score-matching ---
    algs.append(("diffusion", DiffusionSampling(
        budget=budget, model=model_factory(),
        score_net=ScoreNetwork1D(hidden_dim=64, n_layers=4),
        batch_size=batch, total_theta_steps=8000, n_score_steps_per_outer=50,
        sigma_min=0.01, sigma_max=1.0, n_sigma_levels=10,
        langevin_steps=10, langevin_step_size=2e-5,
        lr_theta=1e-3, lr_score=1e-3, device=device,
    )))

    return algs


# ---------------------------------------------------------------------------
# Main experiment entry point
# ---------------------------------------------------------------------------


def run_experiment(
    function_names: Optional[List[str]] = None,
    budgets: Optional[List[int]] = None,
    model_factory: Optional[Callable[[], nn.Module]] = None,
    generated: bool = False,
    n_func_per_class: int = 50,
    split: str = "all",
    device: str = "cpu",
    output_dir: str = "results",
    seed: int = 42,
    lr: float = 1.5e-3,
) -> Dict[str, Any]:
    """Run multi-budget evaluation.

    Parameters
    ----------
    function_names : list of str, optional
        Functions to evaluate (None → all hardcoded or generated).
    budgets : list of int
        Increasing sampling budgets, e.g. ``[32, 64, 128, 256, 512, 1024]``.
    model_factory : callable ``() -> nn.Module``
        User-supplied model factory.
    generated : bool
        Use generated library.
    n_func_per_class : int
        Functions per class when ``generated``.
    split : str
        ``"train"`` (70%), ``"test"`` (30%), or ``"all"``.
    device : str
    output_dir : str
    seed : int

    Returns
    -------
    dict
        ``curves[func_name][alg_name] = {"budgets": [...], "errors": [...]}``
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if budgets is None:
        budgets = [32, 64, 128, 256, 512, 1024]

    if model_factory is None:
        from models.approximator import MLP
        def model_factory() -> nn.Module:
            return MLP(hidden_dims=[32, 32, 32, 32], activation=torch.nn.Tanh())

    os.makedirs(output_dir, exist_ok=True)

    # Resolve function library
    if function_names is None:
        if generated:
            from data.function_generator import generate_function_library
            _lib = generate_function_library(n_per_class=n_func_per_class, seed=seed, split=split)
            function_names = sorted(_lib.keys())
        else:
            _lib = FUNCTION_LIBRARY
            function_names = sorted(FUNCTION_LIBRARY.keys())
    else:
        _lib = dict(FUNCTION_LIBRARY)
        if generated:
            from data.function_generator import generate_function_library
            _gen = generate_function_library(n_per_class=n_func_per_class, seed=seed, split="all")
            _lib.update(_gen)

    # Data structure: curves[func][alg] = {"budgets": [...], "errors": [...]}
    curves: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: {"budgets": [], "errors": []})
    )

    n_funcs = len(function_names)

    for fi, func_name in enumerate(function_names):
        func = _lib[func_name]
        eval_grid = np.linspace(0.0, 1.0, func.dense_resolution)
        from tasks.downstream import FunctionApproximation
        task = FunctionApproximation(f_target=func.evaluate, label=func_name)

        print(f"\n{'='*70}")
        print(f"[{fi+1}/{n_funcs}] {func_name}  [{device}]")
        print(f"{'='*70}")

        for budget in budgets:
            is_pow2 = _is_power_of_two(budget)
            algs = _build_algorithms(
                budget=budget,
                model_factory=model_factory,
                device=device,
                skip_power_of_two=(not is_pow2),
                lr=lr,
            )

            print(f"  budget={budget} ({len(algs)} algorithms)")

            for alg_name, alg in algs:
                t0 = time.time()
                try:
                    result: AlgorithmResult = alg.run(
                        task=task,
                        eval_grid=eval_grid,
                        function_name=func_name,
                    )
                    elapsed = time.time() - t0
                    err = float(result.final_l2_error)
                    curves[func_name][alg_name]["budgets"].append(budget)
                    curves[func_name][alg_name]["errors"].append(err)
                    print(
                        f"    {alg_name:<22s} L²={err:.6f} "
                        f"({elapsed:.1f}s)"
                    )
                except Exception as e:
                    print(f"    {alg_name:<22s} FAILED: {e}")

    # Serialise
    serialisable = {}
    for fn, algs in curves.items():
        serialisable[fn] = {}
        for an, d in algs.items():
            serialisable[fn][an] = {
                "budgets": d["budgets"],
                "errors": d["errors"],
            }

    # Merge with existing curves.json if present
    curves_path = os.path.join(output_dir, "curves.json")
    if os.path.exists(curves_path):
        with open(curves_path, "r") as f:
            existing = json.load(f)
        serialisable.update(existing)

    with open(curves_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    # Print summary at the largest budget
    if budgets:
        last_budget = budgets[-1]
        print(f"\n{'='*70}")
        print(f"  Final L² errors at budget = {last_budget}")
        print(f"{'='*70}")
        alg_names_all = sorted(set(
            an for fn in serialisable for an in serialisable[fn]
        ))
        header = f"{'Function':<30s}" + "".join(f"{a:>18s}" for a in alg_names_all)
        print(header)
        print("-" * len(header))
        for fn in function_names:
            row = f"{fn:<30s}"
            best = min(
                (v for v in [serialisable[fn].get(a, {}).get("errors", [None]) for a in alg_names_all] if v),
                key=lambda x: x[-1] if x else float("inf"),
                default=[float("inf")],
            )[-1] if serialisable.get(fn) else float("inf")
            for an in alg_names_all:
                rec = serialisable.get(fn, {}).get(an, {})
                errs = rec.get("errors", [])
                if errs and errs[-1] == errs[-1]:  # not NaN
                    val = errs[-1]
                    star = " *" if abs(val - best) < 1e-10 else ""
                    row += f"{val:>16.6f}{star}"
                else:
                    row += f"{'—':>18s}"
            print(row)

    return serialisable
