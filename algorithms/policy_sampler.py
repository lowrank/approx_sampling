"""
Learned policy / meta-sampler.

A small policy network π_ψ observes the current error histogram (or
summary statistics of the error surface encoded by a few past samples)
and outputs a probability distribution over sampling locations.  The
policy is trained via REINFORCE to minimise the final L^2 error.

This is distinct from the adversarial/IS approaches in that the policy
observes the *current state* (error surface) and produces the *next
action* (sampling locations) — a Markov decision process formulation.

Reference
---------
- Mnih et al. (2016) — "Asynchronous Methods for Deep Reinforcement
  Learning" (A3C / REINFORCE baseline).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


class PolicyNetwork(nn.Module):
    """Policy that maps an error histogram to bin probabilities.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins.
    hidden_dim : int
        Hidden dimension.
    """

    def __init__(self, n_bins: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.net = nn.Sequential(
            nn.Linear(n_bins, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_bins),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return logits over bins."""
        return self.net(state)

    def sample(self, state: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample *n* points given state (error histogram).

        Returns
        -------
        x : torch.Tensor  shape ``(n,)``
        log_prob : torch.Tensor  shape ``(n,)``
        """
        logits = self.forward(state)  # (n_bins,)
        probs = F.softmax(logits, dim=0)
        bins = torch.multinomial(probs, n, replacement=True)
        bin_w = 1.0 / self.n_bins
        offsets = torch.rand(n, device=bins.device) * bin_w
        x = bins.float() * bin_w + offsets
        log_p = torch.log(probs[bins] + 1e-10) + np.log(self.n_bins)
        return x, log_p


class PolicySampling(BaseSamplingAlgorithm):
    """REINFORCE-trained policy for sequential sampling.

    Parameters
    ----------
    budget : int
        Total function evaluations.
    model : MLP
        Approximator f_θ.
    policy : PolicyNetwork
        Policy network π_ψ.
    n_bins : int
        Histogram bins for state representation.
    batch_size : int
        New points per outer iteration.
    total_theta_steps : int
        Total gradient updates for f_θ.
    n_policy_steps_per_outer : int
        Policy gradient updates per outer iteration.
    lr_theta : float
        LR for f_θ.
    lr_policy : float
        LR for π_ψ.
    device : str
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        policy: PolicyNetwork,
        n_bins: int = 32,
        batch_size: int = 64,
        total_theta_steps: int = 8000,
        n_policy_steps_per_outer: int = 10,
        lr_theta: float = 1e-3,
        lr_policy: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        super().__init__("policy", budget, model, device)
        self.policy = policy.to(device)
        self.n_bins = n_bins
        self.batch_size = batch_size
        self.total_theta_steps = total_theta_steps
        self.n_policy_steps_per_outer = n_policy_steps_per_outer
        self.lr_theta = lr_theta
        self.lr_policy = lr_policy

    def _compute_state(
        self, x_buf: np.ndarray, err_buf: np.ndarray
    ) -> torch.Tensor:
        """Build error histogram (state vector)."""
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        hist, _ = np.histogram(
            np.clip(x_buf, 0, 1), bins=edges, weights=err_buf
        )
        # Also add count histogram for coverage
        cnt, _ = np.histogram(np.clip(x_buf, 0, 1), bins=edges)
        n_total = max(len(x_buf), 1)
        state = hist / (n_total + 1e-10)
        state = state / (state.sum() + 1e-10)
        return torch.from_numpy(state.astype(np.float32)).to(self.device)

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        n_outer = max(1, self.budget // self.batch_size)
        theta_steps_per_outer = max(1, self.total_theta_steps // n_outer)

        opt_theta = torch.optim.Adam(self.model.parameters(), lr=self.lr_theta)
        opt_policy = torch.optim.Adam(self.policy.parameters(), lr=self.lr_policy)
        WARMUP, DECAY = 10, 0.999

        l2_history: list[float] = []
        all_points: list[np.ndarray] = []
        all_values: list[np.ndarray] = []

        self.model.train()
        self.policy.train()
        log_freq = max(1, n_outer // 10)
        step_counter = 0
        baseline_r = 0.0
        gamma = 0.9

        for outer in range(n_outer):
            # ---- 1. Compute state from past data ----
            if len(all_points) == 0:
                state = torch.ones(self.n_bins, device=self.device) / self.n_bins
            else:
                all_x = np.concatenate(all_points)
                x_t = torch.from_numpy(all_x.astype(np.float32)).to(self.device)
                with torch.no_grad():
                    err = task.pointwise_loss(self.model, x_t)
                state = self._compute_state(
                    np.concatenate(all_points), err.cpu().numpy()
                )

            # ---- 2. Sample from policy ----
            x_new, log_p = self.policy.sample(state, self.batch_size)
            y_new = torch.from_numpy(
                task.source_values(x_new.cpu().numpy()).astype(np.float32)
            ).to(self.device)
            all_points.append(x_new.cpu().numpy())
            all_values.append(y_new.cpu().numpy())

            # ---- 3. Train θ on replay buffer ----
            buf_x = torch.from_numpy(
                np.concatenate(all_points).astype(np.float32)
            ).to(self.device)
            buf_y = torch.from_numpy(
                np.concatenate(all_values).astype(np.float32)
            ).to(self.device)
            n_buf = buf_x.shape[0]
            w_buf = self._is_weights(buf_x.cpu().numpy()).to(self.device)

            for _ in range(theta_steps_per_outer):
                idx = torch.randperm(n_buf, device=self.device)[:self.batch_size]
                bx = buf_x[idx]
                opt_theta.zero_grad()
                loss = torch.mean(w_buf[idx] * task.pointwise_loss(self.model, bx))
                loss.backward()
                opt_theta.step()
                if step_counter >= WARMUP:
                    for g in opt_theta.param_groups:
                        g["lr"] = self.lr_theta * (DECAY ** (step_counter - WARMUP))
                step_counter += 1

            # ---- 4. Compute reward (negative L² error improvement) ----
            err_before = l2_history[-1] if l2_history else 10.0
            err_now = task.compute_l2_error(self.model, eval_grid)
            reward = err_before - err_now  # positive if error decreased
            l2_history.append(err_now)

            # ---- 5. Train policy via REINFORCE ----
            baseline_r = gamma * baseline_r + (1 - gamma) * reward
            advantage = reward - baseline_r

            for _ in range(self.n_policy_steps_per_outer):
                opt_policy.zero_grad()
                _, log_p_new = self.policy.sample(state.detach(), self.batch_size)
                loss_p = -advantage * log_p_new.mean()
                entropy = -(F.softmax(self.policy(state), dim=0) * F.log_softmax(
                    self.policy(state), dim=0
                )).sum()
                (loss_p - 0.01 * entropy).backward()
                opt_policy.step()

            if outer % log_freq == 0:
                # log already recorded above for reward computation
                pass

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
