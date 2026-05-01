"""
Adaptive residual-based refinement algorithm.

Starts with a small uniform grid, trains the model, then iteratively
adds points where the pointwise approximation error is largest.  This
is an empirical (non-learnable) strategy that mimics greedy reduced-basis
approaches.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


class AdaptiveResidualSampling(BaseSamplingAlgorithm):
    """Adaptive sampling driven by pointwise residual.

    At each refinement round the current model is evaluated on a dense
    candidate set; the *n_add* points with highest absolute error are
    appended to the training set and the model is retrained (warm-started).

    Parameters
    ----------
    budget : int
        Total number of distinct function evaluations permitted.
    model : MLP
        Trainable network.
    n_initial : int
        Number of points in the initial uniform grid.
    n_add : int
        Number of points added per refinement round.
    candidate_size : int
        Number of candidate points for error evaluation.
    epochs_per_round : int
        Training epochs in each refinement round.
    final_epochs : int
        Extra epochs after the last refinement.
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
        n_initial: int = 50,
        n_add: int = 50,
        candidate_size: int = 2000,
        epochs_per_round: int = 300,
        final_epochs: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        super().__init__("adaptive_residual", budget, model, device)
        self.n_initial = n_initial
        self.n_add = n_add
        self.candidate_size = candidate_size
        self.epochs_per_round = epochs_per_round
        self.final_epochs = final_epochs
        self.batch_size = batch_size
        self.lr = lr

    @staticmethod
    def _train_epochs(
        model: MLP,
        x_np: np.ndarray,
        y_np: np.ndarray,
        epochs: int,
        batch_size: int,
        lr: float,
        device: str,
        task: Any,
    ) -> None:
        """Train *model* on fixed dataset for *epochs* passes."""
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs
        )

        x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
        y_t = torch.from_numpy(y_np.astype(np.float32)).to(device)
        dataset = TensorDataset(x_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(epochs):
            for bx, _ in loader:
                opt.zero_grad()
                loss = torch.mean(task.pointwise_loss(model, bx))
                loss.backward()
                opt.step()
            scheduler.step()

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        candidate_grid = np.linspace(0.0, 1.0, self.candidate_size)

        # Initial points
        x_train = np.linspace(0.0, 1.0, self.n_initial)
        y_train = task.source_values(x_train)

        l2_history: list[float] = []
        n_used = self.n_initial
        current_budget = self.budget

        while n_used < current_budget:
            # Train on current set
            self._train_epochs(
                self.model, x_train, y_train,
                self.epochs_per_round, self.batch_size, self.lr, self.device,
                task,
            )

            # Evaluate L^2 error
            err = task.compute_l2_error(self.model, eval_grid)
            l2_history.append(err)

            # Compute pointwise error on candidate grid
            self.model.eval()
            with torch.no_grad():
                t_cand = torch.from_numpy(
                    candidate_grid.astype(np.float32)
                ).to(self.device)
                y_pred = self.model(t_cand).squeeze(-1).cpu().numpy()
            self.model.train()
            y_cand = task.reference(candidate_grid)
            abs_err = np.abs(y_cand - y_pred)

            # Exclude already-sampled neighbourhoods (simple mask)
            # to avoid exact duplicates while keeping diversity
            mask = np.ones_like(candidate_grid, dtype=bool)
            tol = 1e-4
            for x0 in x_train:
                mask &= np.abs(candidate_grid - x0) > tol
            if not mask.any():
                break  # no new points to add

            # Select n_add worst points
            err_masked = np.where(mask, abs_err, -np.inf)
            n_to_add = min(self.n_add, current_budget - n_used)
            worst_idx = np.argpartition(err_masked, -n_to_add)[-n_to_add:]
            new_x = candidate_grid[worst_idx]
            new_y = task.source_values(new_x)

            x_train = np.concatenate([x_train, new_x])
            y_train = np.concatenate([y_train, new_y])
            n_used += n_to_add

        # Final training
        self._train_epochs(
            self.model, x_train, y_train,
            self.final_epochs, self.batch_size, self.lr, self.device,
            task,
        )
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
