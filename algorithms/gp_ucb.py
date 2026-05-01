"""
GP-UCB active sampling over the error surface.

Fits a Gaussian Process to the pointwise error and maximises the
Upper Confidence Bound acquisition function to select new samples.

Reference
---------
Srinivas et al. (2010) — "Gaussian Process Optimization in the Bandit
Setting: No Regret and Experimental Design" (GP-UCB)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


class GPUCBSampling(BaseSamplingAlgorithm):
    """GP-UCB active sampling over the error surface.

    Parameters
    ----------
    budget : int
        Total function evaluations.
    model : MLP
        Approximator f_θ.
    batch_size : int
        New points per round.
    epochs_per_round : int
        Training epochs for f_θ per round.
    lr : float
        LR for f_θ.
    beta : float
        UCB exploration coefficient.
    device : str
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        batch_size: int = 64,
        epochs_per_round: int = 500,
        lr: float = 1e-3,
        beta: float = 2.0,
        device: str = "cpu",
    ) -> None:
        super().__init__("gp_ucb", budget, model, device)
        self.batch_size = batch_size
        self.epochs_per_round = epochs_per_round
        self.lr = lr
        self.beta = beta

    def _fit_gp(self, x: np.ndarray, y: np.ndarray) -> tuple:
        from scipy.linalg import cho_factor, cho_solve

        x = x.reshape(-1, 1)
        y = y.ravel()
        length_scale = 0.1
        noise = 1e-4
        K = np.exp(-0.5 * np.sum((x[:, None] - x[None, :]) ** 2, axis=-1) / length_scale**2)
        K += noise * np.eye(len(x))
        L = cho_factor(K)
        alpha = cho_solve(L, y)
        return alpha, x, length_scale

    def _predict_gp(
        self, x_star: np.ndarray, alpha: np.ndarray, x_train: np.ndarray, length_scale: float
    ) -> tuple[np.ndarray, np.ndarray]:
        K_star = np.exp(
            -0.5 * np.sum((x_star[:, None] - x_train[None, :]) ** 2, axis=-1) / length_scale**2
        )
        mu = K_star @ alpha
        v = np.ones(len(x_star)) - np.sum(K_star**2, axis=-1)
        v = np.maximum(v, 0.0)
        return mu, np.sqrt(v + 1e-10)

    def run(
        self,
        task: Callable[[np.ndarray], np.ndarray],
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        f_target = task.reference
        y_eval = f_target(eval_grid)
        n_rounds = max(1, self.budget // self.batch_size)

        l2_history: list[float] = []
        all_points: list[np.ndarray] = []
        all_values: list[np.ndarray] = []
        all_errors: list[np.ndarray] = []

        candidate = np.linspace(0.0, 1.0, 500)

        for rnd in range(n_rounds):
            if rnd < 2 or len(all_points) < 10:
                x_new = np.random.uniform(0, 1, self.batch_size)
            else:
                err_x = np.concatenate(all_points)
                err_y = np.concatenate(all_errors)
                try:
                    alpha, x_tr, ls = self._fit_gp(err_x, err_y)
                    mu, sigma = self._predict_gp(candidate, alpha, x_tr, ls)
                    ucb = mu + self.beta * sigma
                    top_idx = np.argpartition(ucb, -self.batch_size)[-self.batch_size:]
                    x_new = candidate[top_idx]
                except Exception:
                    x_new = np.random.uniform(0, 1, self.batch_size)

            y_new = f_target(x_new)
            all_points.append(x_new)
            all_values.append(y_new)

            all_x = np.concatenate(all_points)
            all_y = np.concatenate(all_values)
            x_t = torch.from_numpy(all_x.astype(np.float32)).to(self.device)
            y_t = torch.from_numpy(all_y.astype(np.float32)).to(self.device)
            n_buf = all_x.shape[0]

            self.model.train()
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            for _ in range(self.epochs_per_round):
                idx = torch.randperm(n_buf, device=self.device)[
                    :min(self.batch_size, n_buf)
                ]
                bx, by = x_t[idx], y_t[idx]
                opt.zero_grad()
                loss = torch.mean((self.model(bx).squeeze(-1) - by) ** 2)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                t_cand = torch.from_numpy(candidate.astype(np.float32)).to(self.device)
                pred = self.model(t_cand).squeeze(-1).cpu().numpy()
            err_vals = (pred - f_target(candidate)) ** 2
            all_errors.append(err_vals)
            self.model.train()

            err = task.compute_l2_error(self.model, eval_grid)
            l2_history.append(err)

        pts = np.unique(np.concatenate(all_points))

        return AlgorithmResult(
            algorithm_name=self.algorithm_name,
            function_name=function_name,
            final_l2_error=l2_history[-1] if l2_history else float("inf"),
            l2_error_history=l2_history,
            sampling_points=pts,
            extra_info={"beta": self.beta},
            trained_model=self.model,
        )
