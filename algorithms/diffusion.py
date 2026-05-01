"""
Diffusion-based generative sampling via denoising score matching.

Trains a score network s_φ(x, σ) to approximate the score of the
error-weighted density

    p*(x) ∝ |f_θ(x) − f(x)|²

using weighted denoising score matching (DSM).  New samples are drawn
via annealed Langevin dynamics (gradient ascent on the log-density
plus injected noise), which converges to the learned distribution as
the noise level decreases.

This replaces the REINFORCE policy-gradient estimator used in the
adversarial approach with score matching, which is typically more
stable and avoids high-variance gradient estimates.

Reference
---------
Song & Ermon (2019) — "Generative Modeling by Estimating Gradients of
the Data Distribution" (NCSN).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


# ---------------------------------------------------------------------------
# Score network for 1D data on [0, 1]
# ---------------------------------------------------------------------------


class ScoreNetwork1D(nn.Module):
    """1D score network with Gaussian Fourier random-feature time embedding.

    Parameters
    ----------
    hidden_dim : int
        Width of hidden layers.
    n_layers : int
        Number of hidden layers.
    n_time_features : int
        Dimension of the time embedding.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        n_time_features: int = 32,
    ) -> None:
        super().__init__()
        self.n_time_features = n_time_features
        # Fixed random Fourier features for time conditioning
        self.register_buffer(
            "time_W",
            torch.randn(1, n_time_features) * 2.0 * math.pi,
        )

        layers: list[nn.Module] = []
        in_dim = 1 + 2 * n_time_features  # x + sin/cos embedding
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def _time_embedding(self, log_sigma: torch.Tensor) -> torch.Tensor:
        """Gaussian Fourier feature embedding of log_sigma."""
        if log_sigma.dim() == 1:
            log_sigma = log_sigma.unsqueeze(-1)
        proj = log_sigma @ self.time_W  # (N, n_features)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        """Predict the score ∇_x log p_σ(x).

        Parameters
        ----------
        x : torch.Tensor, shape ``(N,)`` or ``(N, 1)``
            Spatial coordinate in [0, 1].
        log_sigma : torch.Tensor, shape ``(N,)`` or ``(N, 1)``
            Log noise level.

        Returns
        -------
        torch.Tensor, shape ``(N, 1)``
            Predicted score.
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if log_sigma.dim() == 1:
            log_sigma = log_sigma.unsqueeze(-1)
        t_emb = self._time_embedding(log_sigma)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


# ---------------------------------------------------------------------------
# Diffusion sampling algorithm
# ---------------------------------------------------------------------------


class DiffusionSampling(BaseSamplingAlgorithm):
    """Diffusion / score-matching generative sampling.

    Parameters
    ----------
    budget : int
        Max distinct function evaluations.
    model : MLP
        Approximator network f_θ.
    score_net : ScoreNetwork1D
        Score network s_φ.
    batch_size : int
        New samples per outer iteration.
    total_theta_steps : int
        Total gradient updates for f_θ.
    n_score_steps_per_outer : int
        Score-network gradient updates per outer iteration.
    sigma_min : float
        Smallest noise level.
    sigma_max : float
        Largest noise level.
    n_sigma_levels : int
        Number of noise levels (geometric progression).
    langevin_steps : int
        Langevin steps per noise level during sampling.
    langevin_step_size : float
        Base step-size multiplier for Langevin dynamics.
    lr_theta : float
        Learning rate for f_θ.
    lr_score : float
        Learning rate for s_φ.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        score_net: ScoreNetwork1D,
        batch_size: int = 64,
        total_theta_steps: int = 8000,
        n_score_steps_per_outer: int = 50,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        n_sigma_levels: int = 10,
        langevin_steps: int = 10,
        langevin_step_size: float = 2e-5,
        lr_theta: float = 1e-3,
        lr_score: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        super().__init__("diffusion", budget, model, device)
        self.score_net = score_net.to(device)
        self.batch_size = batch_size
        self.total_theta_steps = total_theta_steps
        self.n_score_steps_per_outer = n_score_steps_per_outer
        self.lr_theta = lr_theta
        self.lr_score = lr_score

        # Noise schedule (geometric)
        self.sigma_levels: list[float] = list(
            np.exp(
                np.linspace(
                    np.log(sigma_max), np.log(sigma_min), n_sigma_levels
                )
            )
        )
        self.langevin_steps = langevin_steps
        self.langevin_step_size = langevin_step_size

    # ------------------------------------------------------------------
    # Sampling via annealed Langevin dynamics
    # ------------------------------------------------------------------

    def _sample_langevin(self, n_samples: int) -> torch.Tensor:
        """Draw *n_samples* from the score model via annealed Langevin.

        Returns
        -------
        torch.Tensor, shape ``(n_samples,)``, values in [0, 1].
        """
        self.score_net.eval()
        device = self.device

        # Start from uniform on [0, 1]
        x = torch.rand(n_samples, device=device)

        for sigma in reversed(self.sigma_levels):
            log_sigma = torch.full(
                (n_samples,), math.log(sigma), device=device
            )
            alpha = self.langevin_step_size * (sigma / self.sigma_levels[-1]) ** 2
            for _ in range(self.langevin_steps):
                x.requires_grad_(False)
                score = self.score_net(x, log_sigma).squeeze(-1)
                noise = torch.randn(n_samples, device=device)
                x = x + alpha * score + math.sqrt(2.0 * alpha) * noise
                # Reflect at boundaries (mirror)
                x = torch.where(x < 0.0, -x, x)
                x = torch.where(x > 1.0, 2.0 - x, x)
                x = x.clamp(0.0, 1.0)

        self.score_net.train()
        return x.detach()

    # ------------------------------------------------------------------
    # Score-matching training
    # ------------------------------------------------------------------

    def _train_score_net(
        self,
        x_data: torch.Tensor,
        weights: torch.Tensor,
        opt_score: torch.optim.Optimizer,
    ) -> float:
        """One step of weighted denoising score matching.

        Parameters
        ----------
        x_data : torch.Tensor, shape ``(N,)``
            Points sampled from the current distribution.
        weights : torch.Tensor, shape ``(N,)``
            Per-point error weights ``|f_θ(x_i) - f(x_i)|²``.
        opt_score : torch.optim.Optimizer
            Optimiser for the score network.

        Returns
        -------
        float
            Score-matching loss value.
        """
        N = x_data.shape[0]

        # Sample random noise level per point
        idx = torch.randint(0, len(self.sigma_levels), (N,), device=self.device)
        sigma = torch.tensor(
            [self.sigma_levels[i] for i in idx.tolist()],
            device=self.device,
            dtype=torch.float32,
        )
        log_sigma = torch.log(sigma)

        # Perturb data
        noise = torch.randn(N, device=self.device)
        x_noisy = x_data + sigma * noise
        # Reflect to stay in [0, 1] (consistent with Langevin)
        x_noisy = torch.where(x_noisy < 0.0, -x_noisy, x_noisy)
        x_noisy = torch.where(x_noisy > 1.0, 2.0 - x_noisy, x_noisy)

        # Target score: ∇_x log p(x_noisy | x) = -(x_noisy - x) / σ²
        target_score = -(x_noisy - x_data) / (sigma**2 + 1e-8)

        # Normalise weights for stable training
        w = weights / (weights.sum() + 1e-8) * N

        opt_score.zero_grad()
        pred_score = self.score_net(x_noisy, log_sigma).squeeze(-1)
        loss = torch.mean(w * (pred_score - target_score) ** 2)
        loss.backward()
        opt_score.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        n_outer = max(1, self.budget // self.batch_size)
        theta_steps_per_outer = max(1, self.total_theta_steps // n_outer)

        opt_theta = torch.optim.Adam(self.model.parameters(), lr=self.lr_theta)
        opt_score = torch.optim.Adam(self.score_net.parameters(), lr=self.lr_score)
        sched_theta = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_theta, T_max=self.total_theta_steps
        )

        l2_history: list[float] = []
        all_points: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        self.model.train()
        self.score_net.train()

        log_freq = max(1, n_outer // 10)
        step_counter = 0

        for outer in range(n_outer):
            # ---- 1. Sample new points via annealed Langevin ----
            with torch.no_grad():
                z_new = self._sample_langevin(self.batch_size)
            y_new = torch.from_numpy(
                task.source_values(z_new.cpu().numpy()).astype(np.float32)
            ).to(self.device)

            all_points.append(z_new.cpu())
            all_values.append(y_new.cpu())

            # ---- 2. Train θ on replay buffer ----
            buffer_x = torch.cat([b.to(self.device) for b in all_points])
            buffer_y = torch.cat([b.to(self.device) for b in all_values])
            n_buffer = buffer_x.shape[0]

            for _ in range(theta_steps_per_outer):
                idx = torch.randperm(n_buffer, device=self.device)[:self.batch_size]
                bx = buffer_x[idx]
                opt_theta.zero_grad()
                loss = torch.mean(task.pointwise_loss(self.model, bx))
                loss.backward()
                opt_theta.step()
                sched_theta.step()
                step_counter += 1

            # ---- 3. Compute error weights for score-model training ----
            with torch.no_grad():
                errors = task.pointwise_loss(self.model, buffer_x)
                # Normalise across buffer for stable score-matching weights
                errors = errors / (errors.mean() + 1e-8)

            # ---- 4. Train score network via weighted score matching ----
            for _ in range(self.n_score_steps_per_outer):
                self._train_score_net(buffer_x, errors, opt_score)

            # ---- Log L^2 error ----
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
            extra_info={
                "total_theta_steps": step_counter,
                "sigma_levels": self.sigma_levels,
            },
            trained_model=self.model,
        )
