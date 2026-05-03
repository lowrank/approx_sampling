"""
Normalizing Flow sampling via monotonic cumulative-sum transform.

A piecewise-linear monotonic CDF G_φ is learned via reparameterisation
gradients (no REINFORCE).  Sampling is ``x = G_φ(z)`` with ``z ~ U[0,1]``
and the exact density is ``p(x) = dG_φ^{-1}/dx`` (computed analytically
from the piecewise slopes).

This replaces the score-function estimator used in adversarial / IS
with lower-variance reparameterisation gradients.

Reference
---------
Tabak & Vanden-Eijnden (2010) — "Density estimation by dual ascent of
the log-likelihood".  The piecewise-linear inverse-CDF flow is a simple
instance of a normalising flow.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


# ---------------------------------------------------------------------------
# Monotonic flow module
# ---------------------------------------------------------------------------


class MonotonicFlow1D(nn.Module):
    """Piecewise-linear monotonic CDF on [0, 1].

    The flow is built from *n_segments* learnable positive heights.
    The cumulative sum produces a monotonic CDF normalised to [0, 1].
    Sampling is exact inverse-CDF; density is piecewise constant.
    """

    def __init__(self, n_segments: int = 64) -> None:
        super().__init__()
        self.n_segments = n_segments
        self.raw = nn.Parameter(torch.randn(n_segments))
        self._x_grid: torch.Tensor  # registered in buffer below
        self.register_buffer(
            "_x_grid",
            torch.linspace(0.0, 1.0, n_segments + 1),
        )

    def _heights(self) -> torch.Tensor:
        """Positive, normalised heights (shape ``(n_segments,)``)."""
        h = F.softplus(self.raw) + 1e-6
        return h / h.sum()

    def _cdf(self) -> torch.Tensor:
        """CDF values at bin edges (length ``n_segments + 1``)."""
        c = torch.cumsum(self._heights(), dim=0)
        return torch.cat([torch.zeros(1, device=c.device), c])

    def sample(self, n: int) -> torch.Tensor:
        """Draw *n* samples by inverse CDF (differentiable)."""
        device = self.raw.device
        z = torch.rand(n, device=device)
        h = self._heights()
        c = self._cdf()
        # Find which segment each z falls into
        idx = torch.searchsorted(c[1:-1], z)
        idx = torch.clamp(idx, 0, self.n_segments - 1)
        # Left edge and slope of chosen segment
        x_lo = self._x_grid[idx]
        c_lo = c[idx]
        c_hi = c[idx + 1]
        slope = h[idx]  # ΔCDF / Δx, Δx = 1/n_segments
        # Inverse CDF: x = x_lo + (z - c_lo) / slope / n_segments
        dx = (z - c_lo) / (slope + 1e-10) / self.n_segments
        return x_lo + dx

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-density ``log p(x)``."""
        h = self._heights()
        idx = torch.clamp(
            (x * self.n_segments).long(), 0, self.n_segments - 1
        )
        density = h[idx] * self.n_segments  # p(x) = h_i * n_segments
        return torch.log(density + 1e-10)


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


class NormalizingFlowSampling(BaseSamplingAlgorithm):
    """Reparameterisation-based generative sampling via monotonic flow.

    Parameters
    ----------
    budget : int
        Max distinct function evaluations.
    model : MLP
        Approximator f_θ.
    flow : MonotonicFlow1D
        Normalising flow G_φ.
    batch_size : int
        New samples per outer iteration.
    total_theta_steps : int
        Total gradient updates for f_θ.
    n_flow_steps_per_outer : int
        Flow gradient updates per outer iteration.
    lr_theta : float
        LR for f_θ.
    lr_flow : float
        LR for G_φ.
    entropy_weight : float
        Entropy regulariser weight (prevents collapse).
    device : str
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        flow: MonotonicFlow1D,
        batch_size: int = 64,
        total_theta_steps: int = 8000,
        n_flow_steps_per_outer: int = 30,
        lr_theta: float = 1e-3,
        lr_flow: float = 1e-3,
        entropy_weight: float = 0.02,
        device: str = "cpu",
    ) -> None:
        super().__init__("normalizing_flow", budget, model, device)
        self.flow = flow.to(device)
        self.batch_size = batch_size
        self.total_theta_steps = total_theta_steps
        self.n_flow_steps_per_outer = n_flow_steps_per_outer
        self.lr_theta = lr_theta
        self.lr_flow = lr_flow
        self.entropy_weight = entropy_weight

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        n_outer = max(1, self.budget // self.batch_size)
        theta_steps_per_outer = max(1, self.total_theta_steps // n_outer)

        opt_theta = torch.optim.Adam(self.model.parameters(), lr=self.lr_theta)
        opt_flow = torch.optim.Adam(self.flow.parameters(), lr=self.lr_flow)
        sched_theta = torch.optim.lr_scheduler.CosineAnnealingLR(opt_theta, T_max=self.total_theta_steps)

        l2_history: list[float] = []
        all_points: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        self.model.train()
        self.flow.train()
        log_freq = max(1, n_outer // 10)

        for outer in range(n_outer):
            # ---- 1. Sample from flow (differentiable) ----
            z_new = self.flow.sample(self.batch_size)
            y_new = torch.from_numpy(
                task.source_values(z_new.detach().cpu().numpy()).astype(np.float32)
            ).to(self.device)
            all_points.append(z_new.detach().cpu())
            all_values.append(y_new.cpu())

            # ---- 2. Train θ on replay buffer ----
            buf_x = torch.cat([b.to(self.device) for b in all_points])
            buf_y = torch.cat([b.to(self.device) for b in all_values])
            n_buf = buf_x.shape[0]
            w_buf = self._is_weights(buf_x.cpu().numpy()).to(self.device)

            for _ in range(theta_steps_per_outer):
                idx = torch.randperm(n_buf, device=self.device)[:self.batch_size]
                bx = buf_x[idx]
                opt_theta.zero_grad()
                loss = torch.mean(w_buf[idx] * task.pointwise_loss(self.model, bx))
                loss.backward()
                opt_theta.step()
                sched_theta.step()

            # ---- 3. Update flow via reparameterisation gradient ----
            # Maximise: E_{z~U}[ (f_θ(G_φ(z)) - f(G_φ(z)))² ]
            for _ in range(self.n_flow_steps_per_outer):
                opt_flow.zero_grad()
                z = torch.rand(self.batch_size, device=self.device)
                x = self.flow.sample(z.shape[0])  # uses its own .sample
                x.requires_grad_(True)
                sq_err = task.pointwise_loss(self.model, x)
                obj = torch.mean(sq_err)  # maximise error (so we minimise negative)
                # Entropy bonus
                log_p = self.flow.log_prob(x)
                entropy = -torch.mean(log_p)
                loss_flow = -obj - self.entropy_weight * entropy
                loss_flow.backward()
                opt_flow.step()

            if outer % log_freq == 0 or outer == n_outer - 1:
                err = task.compute_l2_error(self.model, eval_grid)
                l2_history.append(err)

        pts = (
            np.unique(torch.cat(all_points).cpu().numpy())
            if all_points
            else np.array([])
        )

        return AlgorithmResult(
            algorithm_name=self.algorithm_name,
            function_name=function_name,
            final_l2_error=l2_history[-1] if l2_history else float("inf"),
            l2_error_history=l2_history,
            sampling_points=pts,
            extra_info={},
            trained_model=self.model,
        )
