"""
Stein Variational Gradient Descent (SVGD) particle sampler.

Maintains a population of *K* particles that evolve via kernelised
gradient flow toward the error-weighted target density

    p*(x) ∝ |f_θ(x) − f(x)|².

At each iteration every particle is displaced by

    Δx_i = ε Σ_j [ k(x_j, x_i) ∇ log p*(x_j) + ∇_{x_j} k(x_j, x_i) ]

where k is an RBF kernel.  No generator network is required — the
particles themselves are the sampling points.

Reference
---------
Liu & Wang (2016) — "Stein Variational Gradient Descent"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


class SVGDReplayBuffer:
    """Fixed-size replay buffer of particles + values."""

    def __init__(self) -> None:
        self.x: list[torch.Tensor] = []
        self.y: list[torch.Tensor] = []

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x.append(x.cpu())
        self.y.append(y.cpu())

    def all_x(self, device: str) -> torch.Tensor:
        return torch.cat([b.to(device) for b in self.x])

    def all_y(self, device: str) -> torch.Tensor:
        return torch.cat([b.to(device) for b in self.y])

    def __len__(self) -> int:
        return sum(b.shape[0] for b in self.x)


class SVGDSampling(BaseSamplingAlgorithm):
    """SVGD particle-based sampling.

    Parameters
    ----------
    budget : int
        Total function evaluations.
    model : MLP
        Approximator f_θ.
    n_particles : int
        Number of SVGD particles.
    kernel_width : float
        RBF kernel bandwidth (median-heuristic is used per step).
    svgd_step_size : float
        Step-size ε for particle updates.
    svgd_steps_per_round : int
        SVGD iterations per outer round.
    epochs_per_round : int
        Training epochs for θ per round.
    lr : float
        LR for f_θ.
    batch_size : int
        Minibatch size for θ training.
    device : str
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        n_particles: int = 50,
        kernel_width: float = 0.05,
        svgd_step_size: float = 0.01,
        svgd_steps_per_round: int = 50,
        epochs_per_round: int = 500,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        super().__init__("svgd", budget, model, device)
        self.n_particles = n_particles
        self.kernel_width = kernel_width
        self.svgd_step_size = svgd_step_size
        self.svgd_steps_per_round = svgd_steps_per_round
        self.epochs_per_round = epochs_per_round
        self.lr = lr
        self.batch_size = batch_size

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RBF kernel matrix K(x_i, x_j)."""
        x = x.unsqueeze(1)  # (N, 1)
        y = y.unsqueeze(0)  # (1, M)
        sq_dist = (x - y) ** 2
        # Median heuristic for bandwidth
        if hasattr(self, '_h') and self._h > 0:
            h = self._h
        else:
            with torch.no_grad():
                h = torch.median(sq_dist)
                self._h = h.item() if h > 0 else 1e-3
            h = max(h, torch.tensor(1e-4))
        return torch.exp(-sq_dist / (2 * h))

    def _score_fn(
        self, x: torch.Tensor, task: Any
    ) -> torch.Tensor:
        """∇_x log p*(x) where p* ∝ |f_θ − f|²."""
        x_grad = x.detach().clone().requires_grad_(True)
        sq_err = task.pointwise_loss(self.model, x_grad) + 1e-10
        log_p = torch.log(sq_err)
        grad = torch.autograd.grad(log_p.sum(), x_grad, create_graph=True)[0]
        return grad

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        n_rounds = max(1, self.budget // self.n_particles)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        l2_history: list[float] = []
        buffer = SVGDReplayBuffer()

        # Initialise particles uniformly
        particles = torch.rand(self.n_particles, device=self.device)
        self._h = self.kernel_width

        self.model.train()

        for rnd in range(n_rounds):
            # ---- 0. Evaluate f at current particles ----
            with torch.no_grad():
                y_vals_np = task.source_values(particles.cpu().numpy())
            buffer.add(particles.detach(), torch.from_numpy(y_vals_np.astype(np.float32)))

            # ---- 1. Train θ on buffer ----
            buf_x = buffer.all_x(self.device)
            buf_y = buffer.all_y(self.device)
            n_buf = buf_x.shape[0]
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.epochs_per_round
            )
            for _ in range(self.epochs_per_round):
                idx = torch.randperm(n_buf, device=self.device)[
                    :min(self.batch_size, n_buf)
                ]
                bx = buf_x[idx]
                opt.zero_grad()
                loss = torch.mean(task.pointwise_loss(self.model, bx))
                loss.backward()
                opt.step()
                scheduler.step()

            # ---- 2. SVGD particle update ----
            for _ in range(self.svgd_steps_per_round):
                N = particles.shape[0]
                particles.requires_grad_(True)

                score = self._score_fn(particles, task)
                K = self._rbf_kernel(particles, particles)

                # SVGD update: Δx = K @ score + div K
                # div K = Σ_j ∇_{x_j} k(x_j, x_i) (column sum of grad_kernel)
                grad_K = torch.autograd.grad(
                    K.sum(), particles, create_graph=True
                )[0]

                drift = (K @ score) / N + grad_K / N
                particles = particles + self.svgd_step_size * drift
                particles = particles.detach().clamp(0.0, 1.0)

            # ---- Log ----
            err = task.compute_l2_error(self.model, eval_grid)
            l2_history.append(err)

        pts = buffer.all_x("cpu").numpy()

        return AlgorithmResult(
            algorithm_name=self.algorithm_name,
            function_name=function_name,
            final_l2_error=l2_history[-1] if l2_history else float("inf"),
            l2_error_history=l2_history,
            sampling_points=pts,
            extra_info={"n_particles": self.n_particles},
            trained_model=self.model,
        )
