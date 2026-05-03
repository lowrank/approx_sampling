"""
Chebyshev-node sampling algorithm.

Maps Chebyshev nodes of the second kind from [-1, 1] to [0, 1].
The clustering near boundaries helps control Runge phenomenon and
often yields superior L^2 approximation for smooth functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


class ChebyshevSampling(BaseSamplingAlgorithm):
    """Chebyshev-node sampling mapped to [0, 1].

    Uses Chebyshev nodes of the second kind:
    ``x_i = cos((2i-1)/(2N) * pi)`` → map to ``[0, 1]``.

    Parameters
    ----------
    budget : int
        Number of function evaluations.
    model : MLP
        Trainable network.
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
        epochs: int = 2000,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        super().__init__("chebyshev", budget, model, device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    @staticmethod
    def chebyshev_nodes(n: int) -> np.ndarray:
        """Generate *n* Chebyshev nodes mapped to [0, 1]."""
        i = np.arange(1, n + 1)
        # Chebyshev nodes of the second kind on [-1, 1]
        x_cheb = np.cos((2 * i - 1) / (2 * n) * np.pi)
        # Map to [0, 1]
        return 0.5 * (x_cheb + 1.0)

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        n = self.budget
        x_train = self.chebyshev_nodes(n)
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
