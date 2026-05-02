"""
Mixture Density Network (MDN) generative sampler.

A learnable Gaussian mixture with K components learns the error-weighted
density.  Samples are drawn via Gumbel-Softmax component selection
(differentiable relaxation) followed by reparameterised Gaussian
sampling, avoiding REINFORCE variance.

The mixture parameters (μ_k, σ_k, π_k) are trained to maximise the
expected squared error:

    max  E_{x ~ q_φ} [|f_θ(x) − f(x)|²]

with an entropy regulariser to prevent collapse.

Reference
---------
Bishop (1994) — "Mixture Density Networks"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


class MDNSampler(nn.Module):
    """Gaussian mixture sampler with trainable parameters.

    Parameters
    ----------
    n_components : int
        Number of Gaussian components.
    init_temperature : float
        Initial Gumbel-Softmax temperature (annealed over training).
    """

    def __init__(self, n_components: int = 8, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.n_components = n_components
        # Means in (0,1) via sigmoid; init evenly spaced
        raw_means = torch.linspace(0.1, 0.9, n_components)
        self.means_raw = nn.Parameter(torch.logit(raw_means, eps=1e-4))
        self.log_scales = nn.Parameter(torch.full((n_components,), -2.0))
        self.logits = nn.Parameter(torch.zeros(n_components))
        self.register_buffer("temperature", torch.tensor(init_temperature))

    def anneal_temperature(self, factor: float = 0.99) -> None:
        self.temperature.data = torch.clamp(self.temperature * factor, min=0.1)

    @property
    def means(self) -> torch.Tensor:
        return torch.sigmoid(self.means_raw)

    @property
    def scales(self) -> torch.Tensor:
        return F.softplus(self.log_scales) + 1e-4

    @property
    def probs(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=0)

    def sample(self, n: int, hard: bool = False) -> torch.Tensor:
        """Draw *n* samples.  Uses Gumbel-Softmax for differentiability
        when ``hard=False``."""
        device = self.logits.device
        logits = self.logits.unsqueeze(0).expand(n, -1)
        if hard:
            cats = torch.multinomial(self.probs, n, replacement=True)
            one_hot = F.one_hot(cats, self.n_components).float()
        else:
            one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        eps = torch.randn(n, device=device)
        mu = (one_hot @ self.means).squeeze(-1) if one_hot.dim() > 1 else one_hot @ self.means
        sigma = (one_hot @ self.scales).squeeze(-1) if one_hot.dim() > 1 else one_hot @ self.scales
        x = mu + sigma * eps
        return torch.clamp(x, 0.0, 1.0)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Exact mixture log-density."""
        device = x.device
        x = x.unsqueeze(-1)
        mu = self.means.unsqueeze(0)  # (1, K)
        sigma = self.scales.unsqueeze(0)
        pi = self.probs.unsqueeze(0)
        log_component = (
            -0.5 * ((x - mu) / sigma) ** 2
            - torch.log(sigma)
            - 0.5 * np.log(2 * np.pi)
        )
        return torch.logsumexp(torch.log(pi + 1e-10) + log_component, dim=-1)


class MDNSampling(BaseSamplingAlgorithm):
    """Mixture Density Network generative sampling.

    Parameters
    ----------
    budget : int
        Max distinct function evaluations.
    model : MLP
        Approximator f_θ.
    mdn : MDNSampler
        Learnable mixture sampler.
    batch_size : int
        New samples per outer iteration.
    total_theta_steps : int
        Total gradient updates for f_θ.
    n_mdn_steps_per_outer : int
        MDN gradient updates per outer iteration.
    lr_theta : float
        LR for f_θ.
    lr_mdn : float
        LR for MDN.
    entropy_weight : float
        Entropy regulariser weight.
    device : str
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        mdn: MDNSampler,
        batch_size: int = 64,
        total_theta_steps: int = 8000,
        n_mdn_steps_per_outer: int = 30,
        lr_theta: float = 1e-3,
        lr_mdn: float = 1e-3,
        entropy_weight: float = 0.02,
        device: str = "cpu",
    ) -> None:
        super().__init__("mdn", budget, model, device)
        self.mdn = mdn.to(device)
        self.batch_size = batch_size
        self.total_theta_steps = total_theta_steps
        self.n_mdn_steps_per_outer = n_mdn_steps_per_outer
        self.lr_theta = lr_theta
        self.lr_mdn = lr_mdn
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
        opt_mdn = torch.optim.Adam(self.mdn.parameters(), lr=self.lr_mdn)
        WARMUP, DECAY = 10, 0.999

        l2_history: list[float] = []
        all_points: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        self.model.train()
        self.mdn.train()
        log_freq = max(1, n_outer // 10)
        step_counter = 0

        for outer in range(n_outer):
            # ---- 1. Hard-sample from MDN (function evals) ----
            with torch.no_grad():
                z_new = self.mdn.sample(self.batch_size, hard=True)
            y_new = torch.from_numpy(
                task.source_values(z_new.cpu().numpy()).astype(np.float32)
            ).to(self.device)
            all_points.append(z_new.cpu())
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
                if step_counter >= WARMUP:
                    for g in opt_theta.param_groups:
                        g["lr"] = self.lr_theta * (DECAY ** (step_counter - WARMUP))
                step_counter += 1

            # ---- 3. Update MDN via reparameterisation gradient ----
            for _ in range(self.n_mdn_steps_per_outer):
                opt_mdn.zero_grad()
                x = self.mdn.sample(self.batch_size, hard=False)  # Gumbel-softmax
                sq_err = task.pointwise_loss(self.model, x)
                # Entropy: sample a fresh batch for entropy estimation
                log_p = self.mdn.log_prob(x)
                entropy = -torch.mean(log_p)
                loss_mdn = -torch.mean(sq_err) - self.entropy_weight * entropy
                loss_mdn.backward()
                opt_mdn.step()

            self.mdn.anneal_temperature(0.98)

            if outer % log_freq == 0 or outer == n_outer - 1:
                err = task.compute_l2_error(self.model, eval_grid)
                l2_history.append(err)

        pts = np.unique(torch.cat(all_points).cpu().numpy()) if all_points else np.array([])

        return AlgorithmResult(
            algorithm_name=self.algorithm_name,
            function_name=function_name,
            final_l2_error=l2_history[-1] if l2_history else float("inf"),
            l2_error_history=l2_history,
            sampling_points=pts,
            extra_info={},
            trained_model=self.model,
        )
