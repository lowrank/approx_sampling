"""
Adversarial generative sampling algorithm.

Trains a piecewise-constant generator G_φ adversarially against the
approximator f_θ.  The generator learns to concentrate probability mass
where the current approximation error is largest, creating a self-
improving loop (min-max game):

    min_θ  E_{x∼p_φ}[(f_θ(x) − f(x))^2]
    max_φ  E_{x∼p_φ}[(f_θ(x) − f(x))^2]

The generator uses the REINFORCE (score-function) gradient estimator
with a moving-average baseline for variance reduction.  Each distinct
sampled point counts toward the budget.  A replay buffer accumulates all
sampled points so that θ is trained on the full history, avoiding
catastrophic forgetting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP, PiecewiseConstantGenerator


class AdversarialSampling(BaseSamplingAlgorithm):
    """Adversarial min-max training with replay buffer.

    Parameters
    ----------
    budget : int
        Maximum number of distinct function evaluations.
    model : MLP
        Approximator network.
    generator : PiecewiseConstantGenerator
        Sampling distribution generator.
    batch_size : int
        Number of new samples drawn per outer iteration.
    total_theta_steps : int
        Total gradient updates for f_θ (spread across outer iterations).
    n_phi_steps_per_outer : int
        Gradient updates for G_φ per outer iteration.
    lr_theta : float
        Learning rate for f_θ.
    lr_phi : float
        Learning rate for G_φ.
    entropy_weight : float
        Weight of entropy regulariser for G_φ (prevents collapse).
    baseline_momentum : float
        EMA factor for the REINFORCE baseline.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        generator: PiecewiseConstantGenerator,
        batch_size: int = 64,
        total_theta_steps: int = 8000,
        n_phi_steps_per_outer: int = 20,
        lr_theta: float = 1e-3,
        lr_phi: float = 1e-2,
        entropy_weight: float = 0.05,
        baseline_momentum: float = 0.9,
        device: str = "cpu",
    ) -> None:
        super().__init__("adversarial", budget, model, device)
        self.generator = generator.to(device)
        self.batch_size = batch_size
        self.total_theta_steps = total_theta_steps
        self.n_phi_steps_per_outer = n_phi_steps_per_outer
        self.lr_theta = lr_theta
        self.lr_phi = lr_phi
        self.entropy_weight = entropy_weight
        self.baseline_momentum = baseline_momentum

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        n_outer = max(1, self.budget // self.batch_size)
        theta_steps_per_outer = max(1, self.total_theta_steps // n_outer)

        opt_theta = torch.optim.Adam(self.model.parameters(), lr=self.lr_theta)
        opt_phi = torch.optim.Adam(self.generator.parameters(), lr=self.lr_phi)
        WARMUP, DECAY = 10, 0.999

        baseline = 0.0
        l2_history: list[float] = []
        all_points: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        self.model.train()
        self.generator.train()

        # For logging
        log_freq = max(1, n_outer // 10)
        step_counter = 0

        for outer in range(n_outer):
            # ---- 1. Draw new batch from generator (cost: batch_size evals) ----
            with torch.no_grad():
                z_new = self.generator.sample(self.batch_size).to(self.device)
            y_new = torch.from_numpy(
                task.source_values(z_new.cpu().numpy()).astype(np.float32)
            ).to(self.device)

            all_points.append(z_new.cpu())
            all_values.append(y_new.cpu())

            # ---- 2. Train θ on replay buffer ----
            buffer_x = torch.cat([b.to(self.device) for b in all_points])
            buffer_y = torch.cat([b.to(self.device) for b in all_values])
            n_buffer = buffer_x.shape[0]
            w_buf = self._is_weights(buffer_x.cpu().numpy()).to(self.device)

            for _ in range(theta_steps_per_outer):
                # Random minibatch from full buffer
                idx = torch.randperm(n_buffer, device=self.device)[:self.batch_size]
                bx = buffer_x[idx]

                opt_theta.zero_grad()
                loss = torch.mean(w_buf[idx] * task.pointwise_loss(self.model, bx))
                loss.backward()
                opt_theta.step()
                if step_counter >= WARMUP:
                    for g in opt_theta.param_groups:
                        g["lr"] = self.lr_theta * (DECAY ** (step_counter - WARMUP))
                step_counter += 1

            # ---- 3. Evaluate losses on current batch for φ update ----
            with torch.no_grad():
                per_point_loss = task.pointwise_loss(self.model, z_new)
                per_point_loss = per_point_loss / (per_point_loss.mean() + 1e-8)

            # ---- 4. Update φ via REINFORCE ----
            for _ in range(self.n_phi_steps_per_outer):
                opt_phi.zero_grad()
                log_p = self.generator.log_prob(z_new)
                advantage = per_point_loss.detach() - baseline
                reinforce_loss = -torch.mean(advantage * log_p)
                entropy = -torch.sum(
                    self.generator.probs
                    * torch.log(self.generator.probs + 1e-8)
                )
                total_phi_loss = reinforce_loss - self.entropy_weight * entropy
                total_phi_loss.backward()
                opt_phi.step()

            # ---- Update baseline ----
            baseline = (
                self.baseline_momentum * baseline
                + (1 - self.baseline_momentum) * per_point_loss.mean().item()
            )

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
                "final_probs": self.generator.probs.detach().cpu().tolist(),
                "total_theta_steps": step_counter,
            },
            trained_model=self.model,
        )
