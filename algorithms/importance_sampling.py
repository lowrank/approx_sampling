"""
Neural importance sampling algorithm.

Trains a proposal distribution q_φ to minimise the variance of the
importance-sampling (IS) estimator for the L^2 loss.  Points are drawn
from q_φ and importance-weighted so that the gradient for the
approximator f_θ remains unbiased:

    min_θ  E_{x∼q_φ}[ (f_θ(x) − f(x))^2 / q_φ(x) ]

The proposal q_φ is trained via REINFORCE to concentrate on high-error
regions, motivated by the fact that the optimal IS proposal is
q*(x) ∝ |f_θ(x) − f(x)|.

A replay buffer stores all previously sampled points; θ is trained on
the full buffer using IS-corrected weights, while φ is updated using
only the most recent batch (to obtain unbiased REINFORCE gradients).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP, PiecewiseConstantGenerator


class ImportanceSampling(BaseSamplingAlgorithm):
    """Importance-sampling-based training with learnable proposal.

    Parameters
    ----------
    budget : int
        Maximum number of distinct function evaluations.
    model : MLP
        Approximator network.
    proposal : PiecewiseConstantGenerator
        Learnable proposal distribution q_φ.
    batch_size : int
        New samples per outer iteration.
    total_theta_steps : int
        Total gradient updates for f_θ.
    n_phi_steps_per_outer : int
        Gradient updates for q_φ per outer iteration.
    lr_theta : float
        Learning rate for f_θ.
    lr_phi : float
        Learning rate for q_φ.
    entropy_weight : float
        Entropy regularisation weight for q_φ.
    baseline_momentum : float
        EMA factor for REINFORCE baseline.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        proposal: PiecewiseConstantGenerator,
        batch_size: int = 64,
        total_theta_steps: int = 8000,
        n_phi_steps_per_outer: int = 20,
        lr_theta: float = 1e-3,
        lr_phi: float = 1e-2,
        entropy_weight: float = 0.05,
        baseline_momentum: float = 0.9,
        device: str = "cpu",
    ) -> None:
        super().__init__("importance_sampling", budget, model, device)
        self.proposal = proposal.to(device)
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
        opt_phi = torch.optim.Adam(self.proposal.parameters(), lr=self.lr_phi)
        sched_theta = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_theta, T_max=self.total_theta_steps
        )

        baseline = 0.0
        l2_history: list[float] = []
        all_points: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []
        all_log_q: list[torch.Tensor] = []

        self.model.train()
        self.proposal.train()

        log_freq = max(1, n_outer // 10)
        step_counter = 0

        for outer in range(n_outer):
            # ---- 1. Draw new batch from proposal ----
            with torch.no_grad():
                z_new, log_q_new = self.proposal.sample_with_log_prob(
                    self.batch_size
                )
                z_new = z_new.to(self.device)
                log_q_new = log_q_new.to(self.device)
            y_new = torch.from_numpy(
                task.source_values(z_new.cpu().numpy()).astype(np.float32)
            ).to(self.device)

            all_points.append(z_new.cpu())
            all_values.append(y_new.cpu())
            all_log_q.append(log_q_new.cpu())

            # ---- 2. Train θ on replay buffer with IS weights ----
            buffer_x = torch.cat([b.to(self.device) for b in all_points])
            buffer_y = torch.cat([b.to(self.device) for b in all_values])
            buffer_log_q = torch.cat([b.to(self.device) for b in all_log_q])
            n_buffer = buffer_x.shape[0]
            w_buf = self._is_weights(buffer_x.cpu().numpy()).to(self.device)

            for _ in range(theta_steps_per_outer):
                idx = torch.randperm(n_buffer, device=self.device)[:self.batch_size]
                bx, blog_q = buffer_x[idx], buffer_log_q[idx]

                # Self-normalised IS weights
                w_raw = torch.exp(-blog_q)  # 1 / q(x)
                w = w_raw / (w_raw.sum() + 1e-8) * self.batch_size

                opt_theta.zero_grad()
                is_loss = torch.mean(w_buf[idx] * w * task.pointwise_loss(self.model, bx))
                is_loss.backward()
                opt_theta.step()
                sched_theta.step()
                step_counter += 1

            # ---- 3. Evaluate losses on current batch for φ update ----
            with torch.no_grad():
                per_point_loss = task.pointwise_loss(self.model, z_new)
                per_point_loss = per_point_loss / (per_point_loss.mean() + 1e-8)

            # ---- 4. Update φ via REINFORCE ----
            for _ in range(self.n_phi_steps_per_outer):
                opt_phi.zero_grad()
                log_p = self.proposal.log_prob(z_new)
                advantage = per_point_loss.detach() - baseline
                reinforce_loss = -torch.mean(advantage * log_p)
                entropy = -torch.sum(
                    self.proposal.probs
                    * torch.log(self.proposal.probs + 1e-8)
                )
                total_phi_loss = reinforce_loss - self.entropy_weight * entropy
                total_phi_loss.backward()
                opt_phi.step()

            baseline = (
                self.baseline_momentum * baseline
                + (1 - self.baseline_momentum) * per_point_loss.mean().item()
            )

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
                "final_probs": self.proposal.probs.detach().cpu().tolist(),
                "total_theta_steps": step_counter,
            },
            trained_model=self.model,
        )
