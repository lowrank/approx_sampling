"""
Iterative-refinement meta-sampler.

A meta-algorithm that wraps any base sampling strategy in an iterative
refinement loop:

1. Allocate a fraction *init_frac* of the budget to the base sampler.
2. Train the approximator on those initial points.
3. Evaluate the pointwise error on a dense candidate grid (seeing the
   full target function — allowed in our setting).
4. Spend the remaining budget on the *k* points with largest error.
5. Retrain on the union of initial and refined points.

This combines the coverage benefits of a structured initial grid with
the adaptivity of residual-based refinement.  The base sampler can be
any fixed-grid method (uniform, Chebyshev, QMC) or even a learnable
method.

The setting
-----------
All methods have unrestricted access to evaluate the target function
*f* at any *x* ∈ [0,1].  The *budget N* controls only how many distinct
points are used to train the approximator *u_θ*.  Using *f* to compute
the pointwise error for refinement does **not** consume training
budget — this is the standard "dense ground truth available" protocol
used in experimental design and active learning benchmarks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


class IterativeRefinementSampling(BaseSamplingAlgorithm):
    """Iterative refinement meta-algorithm.

    Parameters
    ----------
    budget : int
        Total training points.
    model : MLP
        Approximator.
    base_sampler_name : str
        Identifier of the base sampling strategy (for record-keeping).
    base_sampler_factory : callable ``(int) -> BaseSamplingAlgorithm``
        Given a sub-budget, returns a configured base algorithm.
    init_frac : float
        Fraction of budget spent on the initial coarse sample.
    candidate_size : int
        Dense grid size for error evaluation.
    final_epochs : int
        Training epochs after the refinement step.
    lr : float
        Learning rate.
    batch_size : int
        Minibatch size.
    device : str
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        base_sampler_name: str = "uniform",
        base_sampler_factory: Callable[..., BaseSamplingAlgorithm] | None = None,
        init_frac: float = 0.4,
        candidate_size: int = 2000,
        final_epochs: int = 1000,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        super().__init__(f"iter_{base_sampler_name}", budget, model, device)
        self.base_name = base_sampler_name
        self.base_factory = base_sampler_factory
        self.init_frac = init_frac
        self.candidate_size = candidate_size
        self.final_epochs = final_epochs
        self.lr = lr
        self.batch_size = batch_size

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        candidate = np.linspace(0.0, 1.0, self.candidate_size)

        n_init = max(10, int(self.budget * self.init_frac))
        n_refine = self.budget - n_init

        l2_history: list[float] = []

        # ---- Phase 1: base sampler on initial budget ----
        if self.base_factory is not None:
            base_alg = self.base_factory(n_init, self.model)
            base_result = base_alg.run(task, eval_grid, function_name)
            x_init = base_result.sampling_points
            l2_history.extend(base_result.l2_error_history)
        else:
            # Default: uniform
            x_init = np.linspace(0.0, 1.0, n_init)

        y_init = task.source_values(x_init)

        # ---- Phase 2: train on initial points (if not already from base) ----
        if self.base_factory is None:
            self.model.train()
            x_t = torch.from_numpy(x_init.astype(np.float32)).to(self.device)
            y_t = torch.from_numpy(y_init.astype(np.float32)).to(self.device)
            dataset = TensorDataset(x_t, y_t)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            for _ in range(500):
                for bx, _ in loader:
                    opt.zero_grad()
                    loss = torch.mean(task.pointwise_loss(self.model, bx))
                    loss.backward()
                    opt.step()
            err = task.compute_l2_error(self.model, eval_grid)
            l2_history.append(err)

        # ---- Phase 3: evaluate error on candidate grid ----
        self.model.eval()
        with torch.no_grad():
            t_cand = torch.from_numpy(candidate.astype(np.float32)).to(self.device)
            y_pred = self.model(t_cand).squeeze(-1).cpu().numpy()
        self.model.train()

        y_cand = task.reference(candidate)
        abs_err = np.abs(y_cand - y_pred)

        # Exclude points already in the training set
        mask = np.ones_like(candidate, dtype=bool)
        tol = 1e-4
        for x0 in x_init:
            mask &= np.abs(candidate - x0) > tol

        # ---- Phase 4: select refinement points ----
        if mask.any() and n_refine > 0:
            err_masked = np.where(mask, abs_err, -np.inf)
            worst_idx = np.argpartition(err_masked, -n_refine)[-n_refine:]
            x_refine = candidate[worst_idx]
            y_refine = task.source_values(x_refine)

            x_all = np.concatenate([x_init, x_refine])
            y_all = np.concatenate([y_init, y_refine])
        else:
            x_all = x_init
            y_all = y_init

        # ---- Phase 5: final training on all points ----
        self.model.train()
        x_t = torch.from_numpy(x_all.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y_all.astype(np.float32)).to(self.device)
        dataset = TensorDataset(x_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.final_epochs):
            for bx, _ in loader:
                opt.zero_grad()
                loss = torch.mean(task.pointwise_loss(self.model, bx))
                loss.backward()
                opt.step()

        err = task.compute_l2_error(self.model, eval_grid)
        l2_history.append(err)

        return AlgorithmResult(
            algorithm_name=self.algorithm_name,
            function_name=function_name,
            final_l2_error=l2_history[-1],
            l2_error_history=l2_history,
            sampling_points=x_all,
            trained_model=self.model,
            extra_info={"n_init": n_init, "n_refine": n_refine},
        )
