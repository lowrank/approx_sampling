"""
Quasi-Monte Carlo sampling algorithms.

Provides Sobol and Halton low-discrepancy sequences.  These offer
better space-filling than i.i.d. uniform and often yield O(1/N) (up to
log factors) integration error vs O(1/sqrt(N)).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from scipy.stats import qmc
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


class QMCSampling(BaseSamplingAlgorithm):
    """Quasi-Monte Carlo sampling using low-discrepancy sequences.

    Parameters
    ----------
    budget : int
        Number of function evaluations.
    model : MLP
        Trainable network.
    sequence : {"sobol", "halton"}
        Which low-discrepancy sequence to use.
    epochs : int
        Number of full passes.
    batch_size : int
        Minibatch size.
    lr : float
        Learning rate.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        sequence: Literal["sobol", "halton"] = "sobol",
        epochs: int = 2000,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        name = f"qmc_{sequence}"
        super().__init__(name, budget, model, device)
        self.sequence = sequence
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    @staticmethod
    def generate_sequence(
        n: int, seq_type: str = "sobol"
    ) -> np.ndarray:
        """Generate *n* points in [0, 1] using *seq_type*."""
        # Round to next power of 2 for Sobol' balance properties
        n_gen = n
        if seq_type == "sobol":
            p = 1
            while p < n:
                p <<= 1
            n_gen = p
            sampler = qmc.Sobol(d=1, scramble=True, seed=42)
        elif seq_type == "halton":
            sampler = qmc.Halton(d=1, scramble=True, seed=42)
        else:
            raise ValueError(f"Unknown sequence: {seq_type}")
        points = sampler.random(n_gen)
        return points.flatten()[:n]

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        n = self.budget
        x_train = self.generate_sequence(n, self.sequence)
        x_train = np.clip(x_train, 0.0, 1.0)  # ensure bounds
        y_train = task.source_values(x_train)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        x_t = torch.from_numpy(x_train.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y_train.astype(np.float32)).to(self.device)
        dataset = TensorDataset(x_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        l2_history = []

        self.model.train()
        for ep in range(self.epochs):
            for bx, _ in loader:
                opt.zero_grad()
                loss = torch.mean(task.pointwise_loss(self.model, bx))
                loss.backward()
                opt.step()
                scheduler.step()

            if ep % 100 == 0 or ep == self.epochs - 1:
                err = task.compute_l2_error(self.model, eval_grid)
                l2_history.append(err)

        return AlgorithmResult(
            algorithm_name=self.algorithm_name,
            function_name=function_name,
            final_l2_error=l2_history[-1],
            l2_error_history=l2_history,
            sampling_points=x_train,
            trained_model=self.model,
        )
