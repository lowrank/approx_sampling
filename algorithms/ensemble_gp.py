"""
Uncertainty-based sampling algorithms.

Two complementary approaches that select new sampling points based on
predictive uncertainty rather than direct error:

* **Ensemble** — trains an ensemble of M MLPs (differing only by random
  initialisation), then samples where the predictive variance is largest.
* **GP-UCB** — fits a Gaussian Process to the current pointwise error,
  then maximises the Upper Confidence Bound (UCB) acquisition function.

References
----------
- Lakshminarayanan et al. (2017) — "Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles"
- Srinivas et al. (2010) — "Gaussian Process Optimization in the Bandit
  Setting: No Regret and Experimental Design" (GP-UCB)
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.nn as nn

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


# ======================================================================
# Ensemble
# ======================================================================


class EnsembleSampling(BaseSamplingAlgorithm):
    """Deep-ensemble-based active sampling.

    Parameters
    ----------
    budget : int
        Total function evaluations.
    model_cls : callable ``() -> MLP``
        Factory for fresh ensemble members.
    n_members : int
        Number of ensemble members.
    batch_size : int
        New samples per outer round.
    epochs_per_round : int
        Training epochs per round (per member).
    lr : float
        Learning rate.
    device : str
    """

    def __init__(
        self,
        budget: int,
        model: MLP,  # first member
        n_members: int = 5,
        batch_size: int = 64,
        epochs_per_round: int = 300,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        super().__init__("ensemble", budget, model, device)
        self.n_members = n_members
        self.batch_size = batch_size
        self.epochs_per_round = epochs_per_round
        self.lr = lr

        # Build ensemble (first member is self.model, rest are clones)
        self.ensemble: List[MLP] = [model]
        for _ in range(n_members - 1):
            m = MLP(hidden_dims=[32, 32, 32, 32], activation=nn.Tanh())
            self.ensemble.append(m.to(device))

        self._candidate_grid: np.ndarray | None = None

    def _predict_variances(
        self, x_np: np.ndarray
    ) -> np.ndarray:
        """Predictive variance at points *x_np*."""
        preds = []
        for m in self.ensemble:
            m.eval()
            with torch.no_grad():
                t = torch.from_numpy(x_np.astype(np.float32)).to(self.device)
                p = m(t).squeeze(-1).cpu().numpy()
            preds.append(p)
        preds = np.stack(preds, axis=0)  # (M, N)
        return preds.var(axis=0)

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        n_rounds = max(1, self.budget // self.batch_size)

        l2_history: list[float] = []
        all_points: list[np.ndarray] = []
        all_values: list[np.ndarray] = []

        candidate = np.linspace(0.0, 1.0, 2000)

        for rnd in range(n_rounds):
            # ---- 1. Decide where to sample: maximiser of predictive variance ----
            if rnd == 0:
                # First round: uniform
                x_new = np.random.uniform(0, 1, self.batch_size)
            else:
                # Sample candidate points with probability ∝ variance
                var = self._predict_variances(candidate)
                var = np.maximum(var, 1e-10)
                probs = var / var.sum()
                idx = np.random.choice(len(candidate), self.batch_size, p=probs)
                x_new = candidate[idx]

            y_new = task.source_values(x_new)
            all_points.append(x_new)
            all_values.append(y_new)

            # ---- 2. Train ensemble on all points ----
            all_x = np.concatenate(all_points)
            all_y = np.concatenate(all_values)
            x_t = torch.from_numpy(all_x.astype(np.float32)).to(self.device)
            y_t = torch.from_numpy(all_y.astype(np.float32)).to(self.device)
            n_buf = all_x.shape[0]

            for m in self.ensemble:
                m.train()
                opt = torch.optim.Adam(m.parameters(), lr=self.lr)
                for _ in range(self.epochs_per_round):
                    idx = torch.randperm(n_buf, device=self.device)[
                        :min(self.batch_size, n_buf)
                    ]
                    bx = x_t[idx]
                    opt.zero_grad()
                    loss = torch.mean(task.pointwise_loss(m, bx))
                    loss.backward()
                    opt.step()

            # ---- 3. Evaluate L^2 error (using the first member) ----
            self.model = self.ensemble[0]
            err = task.compute_l2_error(self.model, eval_grid)
            l2_history.append(err)

        pts = np.unique(np.concatenate(all_points))

        return AlgorithmResult(
            algorithm_name=self.algorithm_name,
            function_name=function_name,
            final_l2_error=l2_history[-1] if l2_history else float("inf"),
            l2_error_history=l2_history,
            sampling_points=pts,
            extra_info={"n_members": self.n_members},
            trained_model=self.model,
        )


# ======================================================================
# GP-UCB
# ======================================================================


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
        """Fit a simple RBF-kernel GP to (x, y) using scipy."""
        from scipy.linalg import cho_factor, cho_solve

        x = x.reshape(-1, 1)
        y = y.ravel()

        # RBF kernel with fixed length-scale = 0.1
        length_scale = 0.1
        noise = 1e-4
        K = np.exp(-0.5 * np.sum((x[:, None] - x[None, :]) ** 2, axis=-1) / length_scale**2)
        K += noise * np.eye(len(x))
        try:
            L = cho_factor(K)
            alpha = cho_solve(L, y)
        except Exception:
            # Fallback if Cholesky fails
            alpha = np.linalg.solve(K + 1e-3 * np.eye(len(x)), y)

        return alpha, x, K, length_scale, noise

    def _predict_gp(
        self, x_star: np.ndarray, alpha: np.ndarray, x_train: np.ndarray, length_scale: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """GP predictive mean and variance at *x_star*."""
        K_star = np.exp(
            -0.5 * np.sum((x_star[:, None] - x_train[None, :]) ** 2, axis=-1) / length_scale**2
        )
        mu = K_star @ alpha
        K_star_star = 1.0  # σ² = k(x*, x*) = 1 for RBF
        v = K_star_star - np.sum(K_star * (K_star @ np.linalg.inv(
            np.exp(-0.5 * np.sum((x_train[:, None] - x_train[None, :]) ** 2, axis=-1) / length_scale**2)
            + 1e-4 * np.eye(len(x_train))
        )), axis=-1)  # simplified
        # Actually compute variance properly:
        v = np.ones(len(x_star))
        for i in range(len(x_star)):
            k_s = K_star[i]
            v[i] = 1.0 - k_s @ alpha / (alpha @ k_s + 1e-10)  # approximate
        v = np.maximum(v, 0.0)
        return mu, np.sqrt(v + 1e-10)

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        n_rounds = max(1, self.budget // self.batch_size)

        l2_history: list[float] = []
        all_points: list[np.ndarray] = []
        all_values: list[np.ndarray] = []
        all_errors: list[np.ndarray] = []

        candidate = np.linspace(0.0, 1.0, 500)

        for rnd in range(n_rounds):
            # ---- 1. Choose sampling points ----
            if rnd < 2 or len(all_points) < 10:
                # First rounds: uniform exploration
                x_new = np.random.uniform(0, 1, self.batch_size)
            else:
                # Fit GP to error surface
                err_x = np.concatenate(all_points)
                err_y = np.concatenate(all_errors)
                try:
                    alpha, x_tr, K, ls, _ = self._fit_gp(err_x, err_y)
                    mu, sigma = self._predict_gp(candidate, alpha, x_tr, ls)
                    ucb = mu + self.beta * sigma
                    # Pick top batch_size maximisers
                    top_idx = np.argpartition(ucb, -self.batch_size)[-self.batch_size:]
                    x_new = candidate[top_idx]
                except Exception:
                    x_new = np.random.uniform(0, 1, self.batch_size)

            y_new = task.source_values(x_new)
            all_points.append(x_new)
            all_values.append(y_new)

            # ---- 2. Train θ on buffer ----
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
                bx = x_t[idx]
                opt.zero_grad()
                loss = torch.mean(task.pointwise_loss(self.model, bx))
                loss.backward()
                opt.step()

            # ---- 3. Compute current error surface ----
            self.model.eval()
            with torch.no_grad():
                t_cand = torch.from_numpy(candidate.astype(np.float32)).to(self.device)
                pred = self.model(t_cand).squeeze(-1).cpu().numpy()
            err_vals = (pred - task.reference(candidate)) ** 2
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
