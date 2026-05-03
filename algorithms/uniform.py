"""
Uniform sampling algorithm.

Draws equispaced points on [0, 1] and trains the model with standard
minibatch SGD on the MSE loss.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


class UniformSampling(BaseSamplingAlgorithm):
    """Equispaced sampling on [0, 1].

    Parameters
    ----------
    budget : int
        Number of function evaluations (distinct points).
    model : MLP
        Trainable network.
    epochs : int
        Number of full passes over the sampled points.
    batch_size : int
        Minibatch size.
    lr : float
        Learning rate for Adam.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        epochs: int = 2000,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        super().__init__("uniform", budget, model, device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        n = self.budget
        x_train = np.linspace(0.0, 1.0, n)
        y_train = task.source_values(x_train)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs
        )

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
