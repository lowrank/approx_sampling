"""
Density-network sampling.

A small MLP with exponential output models a positive density
p_φ(x) ∝ exp(NN_φ(x)).  Sampling uses discretised inverse-CDF on a fine
grid of N_grid points (exact for the discretisation).  The normalisation
constant Z_φ = ∫exp(NN_φ) is computed by trapezoidal integration.

Training
--------
At each outer iteration:
  1. Sample a batch from p_φ (via inverse CDF).
  2. Train f_θ on the replay buffer, importance-weighted by 1/p_φ(x_i)
     so that the empirical loss remains an unbiased estimator of the
     true L² loss.
  3. Update p_φ via reparameterised gradient to maximise the expected
     squared error (same adversarial objective), plus an entropy
     regulariser to prevent collapse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


class DensityNetwork(nn.Module):
    """1D density: p(x) ∝ exp(NN(x)), normalised via trapezoidal rule.

    Parameters
    ----------
    hidden_dim : int
    n_layers : int
    n_grid : int
        Discretisation points for CDF construction.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 3, n_grid: int = 2000) -> None:
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.n_grid = n_grid

        # Fixed integration grid
        self.register_buffer("_grid", torch.linspace(0.0, 1.0, n_grid))

    def _unnorm_log_density(self, x: torch.Tensor) -> torch.Tensor:
        """Raw log-density NN(x) (not normalised)."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x).squeeze(-1)

    def _log_Z(self) -> torch.Tensor:
        """Log normalisation constant via trapezoidal rule."""
        log_u = self._unnorm_log_density(self._grid)
        Z = torch.trapz(torch.exp(log_u), self._grid)
        return torch.log(Z + 1e-10)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-density log p(x)."""
        return self._unnorm_log_density(x) - self._log_Z()

    def sample(self, n: int) -> torch.Tensor:
        """Draw *n* samples via inverse CDF."""
        with torch.no_grad():
            log_u = self._unnorm_log_density(self._grid)
            p = torch.exp(log_u - torch.logsumexp(log_u, dim=0))  # softmax
            # Build CDF on grid points
            cdf = torch.cumsum(p, dim=0)
            cdf = torch.cat([torch.zeros(1, device=cdf.device), cdf])
            cdf = cdf / cdf[-1]
            # Inverse CDF
            z = torch.rand(n, device=cdf.device)
            idx = torch.searchsorted(cdf, z) - 1
            idx = idx.clamp(0, self.n_grid - 1)
            # Linear interpolation within bin
            x_lo = self._grid[idx]
            x_hi = self._grid[(idx + 1).clamp(0, self.n_grid - 1)]
            c_lo = cdf[idx]
            c_hi = cdf[idx + 1]
            slope = (x_hi - x_lo) / (c_hi - c_lo + 1e-10)
            return x_lo + slope * (z - c_lo)


class DensityNetworkSampling(BaseSamplingAlgorithm):
    """Density-network-based importance sampling.

    Parameters
    ----------
    budget : int
        Total function evaluations.
    model : MLP
        Approximator f_θ.
    density_net : DensityNetwork
        Learnable density p_φ.
    batch_size : int
        New samples per outer iteration.
    total_theta_steps : int
        Total gradient updates for f_θ.
    n_density_steps : int
        Density-network gradient updates per outer iteration.
    lr_theta : float
        LR for f_θ.
    lr_density : float
        LR for p_φ.
    entropy_weight : float
    device : str
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        density_net: DensityNetwork,
        batch_size: int = 32,
        total_theta_steps: int = 8000,
        n_density_steps: int = 30,
        lr_theta: float = 1.5e-3,
        lr_density: float = 1e-3,
        entropy_weight: float = 0.02,
        device: str = "cpu",
    ) -> None:
        super().__init__("density_network", budget, model, device)
        self.density_net = density_net.to(device)
        self.batch_size = batch_size
        self.total_theta_steps = total_theta_steps
        self.n_density_steps = n_density_steps
        self.lr_theta = lr_theta
        self.lr_density = lr_density
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
        opt_phi = torch.optim.Adam(self.density_net.parameters(), lr=self.lr_density)
        WARMUP, DECAY = 10, 0.999
        step_counter = 0

        l2_history: list[float] = []
        all_points: list[np.ndarray] = []
        all_values: list[np.ndarray] = []

        self.model.train()
        self.density_net.train()

        for outer in range(n_outer):
            # ---- 1. Sample from density (hard, no grad) ----
            with torch.no_grad():
                z_new = self.density_net.sample(self.batch_size)
            y_new = np.array(task.source_values(z_new.cpu().numpy()), dtype=np.float32)
            all_points.append(z_new.cpu().numpy())
            all_values.append(y_new)

            # ---- 2. Train θ on buffer with density IS weights ----
            buf_x_np = np.concatenate(all_points)
            buf_y_np = np.concatenate(all_values)
            buf_x = torch.from_numpy(buf_x_np.astype(np.float32)).to(self.device)
            buf_y = torch.from_numpy(buf_y_np.astype(np.float32)).to(self.device)

            # Density IS weights: w = 1/p(x)
            with torch.no_grad():
                log_p = self.density_net.log_prob(buf_x)
                w_density = torch.exp(-log_p)  # 1/p(x)
                w_density = w_density / w_density.mean()
            # Also add empirical-density IS weights
            w_emp = self._is_weights(buf_x_np).to(self.device)
            w_total = w_density * w_emp

            n_buf = buf_x.shape[0]
            for _ in range(theta_steps_per_outer):
                idx = torch.randperm(n_buf, device=self.device)[:self.batch_size]
                bx, by, bw = buf_x[idx], buf_y[idx], w_total[idx]
                opt_theta.zero_grad()
                pred = self.model(bx).squeeze(-1)
                loss = torch.mean(bw * (pred - by) ** 2)
                loss.backward()
                opt_theta.step()
                step_counter += 1
                if step_counter >= WARMUP:
                    for g in opt_theta.param_groups:
                        g["lr"] = self.lr_theta * (DECAY ** (step_counter - WARMUP))

            # ---- 3. Train density via reparameterised gradient ----
            for _ in range(self.n_density_steps):
                opt_phi.zero_grad()
                # Reparameterised sample (differentiable)
                z = self.density_net.sample(self.batch_size)
                pred = self.model(z).squeeze(-1)
                y_val = torch.from_numpy(
                    task.source_values(z.detach().cpu().numpy()).astype(np.float32)
                ).to(self.device)
                sq_err = (pred - y_val) ** 2
                # Maximise expected error
                obj = torch.mean(sq_err)
                # Entropy bonus
                log_p = self.density_net.log_prob(z)
                entropy = -torch.mean(log_p)
                loss_phi = -obj - self.entropy_weight * entropy
                loss_phi.backward()
                opt_phi.step()

            # ---- Log ----
            err = task.compute_l2_error(self.model, eval_grid)
            l2_history.append(err)

        pts = np.unique(np.concatenate(all_points)) if all_points else np.array([])

        return AlgorithmResult(
            algorithm_name=self.algorithm_name,
            function_name=function_name,
            final_l2_error=l2_history[-1] if l2_history else float("inf"),
            l2_error_history=l2_history,
            sampling_points=pts,
            extra_info={},
            trained_model=self.model,
        )
