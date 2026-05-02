"""
Neural Process adaptive sampler.

A Neural Process (NP) learns a mapping from an observed *context set*
of (x, f(x)) pairs to a sampling distribution over the domain [0,1].
The NP is trained across many functions so that, given a small set of
already-evaluated points, it predicts where the approximator f_θ will
struggle and proposes new x-locations accordingly.

Architecture
------------
* **Encoder**: Deep Sets / permutation-invariant encoding.
  Each context point (x, f(x)) is embedded via an MLP; embeddings are
  aggregated via mean pooling (optionally with cross-attention) to
  produce a global latent representation r.
* **Decoder**: A small MLP maps r to the logits of a piecewise-constant
  density q(x | r) on [0, 1].
* **Training** (amortised): For each function, simulate the sequential
  sampling process.  At each step the NP proposes a batch of new points;
  the reward is the improvement in L² error of the approximator f_θ.
  The NP is updated via REINFORCE (or reparameterisation when using
  Gumbel-Softmax relaxed sampling).

Key difference vs. all other algorithms
---------------------------------------
The NP is **amortised and conditional**: trained once across many
functions, it learns heuristics such as "oscillatory regions need
denser sampling" or "smooth plateaus are safe to skip".  During
inference on a new function it requires no additional training — only
forward passes through the encoder–decoder.

References
----------
- Garnelo et al. (2018) — "Conditional Neural Processes"  (ICML)
- Garnelo et al. (2018) — "Neural Processes"             (ICML workshop)
- Kim et al.  (2019) — "Attentive Neural Processes"      (ICLR)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


# ======================================================================
# Neural Process module
# ======================================================================


class NeuralProcessSampler(nn.Module):
    """Encoder–decoder NP that outputs a sampling distribution over [0,1].

    Parameters
    ----------
    n_bins : int
        Number of histogram bins for the output density.
    dim_x : int
        Input dimension per point (x + f(x) = 2 for 1D).
    dim_h : int
        Hidden dimension of the encoder and decoder MLPs.
    dim_r : int
        Dimension of the global context representation r.
    use_attention : bool
        If True, use cross-attention when decoding.
    """

    def __init__(
        self,
        n_bins: int = 32,
        dim_x: int = 2,
        dim_h: int = 128,
        dim_r: int = 128,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.dim_r = dim_r
        self.use_attention = use_attention

        # Per-point encoder: (x, f(x)) → embedding
        self.point_encoder = nn.Sequential(
            nn.Linear(dim_x, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, dim_r),
        )

        # Global aggregator: mean-pooled embeddings → r
        self.aggregator = nn.Sequential(
            nn.Linear(dim_r, dim_r),
            nn.ReLU(),
            nn.Linear(dim_r, dim_r),
        )

        # Decoder: r → logits over bins
        self.decoder = nn.Sequential(
            nn.Linear(dim_r, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, dim_h),
            nn.ReLU(),
            nn.Linear(dim_h, n_bins),
        )

        if use_attention:
            self.attn_Wq = nn.Linear(dim_r, dim_r)
            self.attn_Wk = nn.Linear(dim_r, dim_r)
            self.attn_Wv = nn.Linear(dim_r, dim_r)

    def encode(self, context_x: torch.Tensor) -> torch.Tensor:
        """Encode a context set into global representation r.

        Parameters
        ----------
        context_x : torch.Tensor, shape ``(N, 2)``
            Stacked (x, f(x)) pairs.  ``N`` can vary.

        Returns
        -------
        torch.Tensor, shape ``(dim_r,)``
        """
        if context_x.shape[0] == 0:
            return torch.zeros(self.dim_r, device=context_x.device)
        emb = self.point_encoder(context_x)  # (N, dim_r)
        r = emb.mean(dim=0)                   # permutation-invariant pool
        return self.aggregator(r)

    def decode(self, r: torch.Tensor) -> torch.Tensor:
        """Decode r into logits over bins.

        Returns
        -------
        torch.Tensor, shape ``(n_bins,)``
        """
        return self.decoder(r)

    def forward(self, context_x: torch.Tensor) -> torch.Tensor:
        """Full encode–decode: logits over bins."""
        r = self.encode(context_x)
        return self.decode(r)

    def sample(
        self, context_x: torch.Tensor, n: int, hard: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample *n* points given context.

        Parameters
        ----------
        context_x : (N, 2) tensor
        n : int
        hard : bool
            ``True`` → categorical (no grad); ``False`` → Gumbel-Softmax.

        Returns
        -------
        x : ``(n,)`` tensor in [0,1]
        log_prob : ``(n,)`` tensor  (log-density at samples)
        """
        logits = self.forward(context_x)  # (n_bins,)
        bin_w = 1.0 / self.n_bins

        if hard:
            probs = F.softmax(logits, dim=0)
            bins = torch.multinomial(probs, n, replacement=True)
            offsets = torch.rand(n, device=bins.device) * bin_w
            x = bins.float() * bin_w + offsets
            log_p = torch.log(probs[bins] + 1e-10) + np.log(self.n_bins)
            return x.detach(), log_p
        else:
            # Gumbel-Softmax relaxation for differentiable sampling
            one_hot = F.gumbel_softmax(
                logits.unsqueeze(0).expand(n, -1),
                tau=0.5, hard=False,
            )  # (n, n_bins)
            offsets = torch.rand(n, device=logits.device) * bin_w
            x = (one_hot @ torch.arange(
                self.n_bins, device=logits.device, dtype=torch.float32
            )) * bin_w + offsets
            x = torch.clamp(x, 0.0, 1.0)
            probs = F.softmax(logits, dim=0)
            bin_idx = torch.argmax(one_hot, dim=-1)
            log_p = torch.log(probs[bin_idx] + 1e-10) + np.log(self.n_bins)
            return x, log_p


# ======================================================================
# NP-based algorithm
# ======================================================================


class NeuralProcessSampling(BaseSamplingAlgorithm):
    """Neural-Process-driven adaptive sampling.

    The NP is trained across functions (amortised).  For a fair
    per-function comparison it is also trainable on a single function
    in the standard outer-iteration loop.

    Parameters
    ----------
    budget : int
        Total function evaluations.
    model : MLP
        Approximator f_θ.
    np_sampler : NeuralProcessSampler
        The NP module.
    batch_size : int
        New points per outer iteration.
    total_theta_steps : int
        Total gradient updates for f_θ.
    n_np_steps_per_outer : int
        NP gradient updates per outer iteration.
    lr_theta : float
        LR for f_θ.
    lr_np : float
        LR for NP.
    entropy_weight : float
        Entropy regularisation for NP's output distribution.
    initial_random : int
        Number of initial uniform-random context points before NP takes over.
    device : str
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        np_sampler: NeuralProcessSampler,
        batch_size: int = 64,
        total_theta_steps: int = 8000,
        n_np_steps_per_outer: int = 20,
        lr_theta: float = 1e-3,
        lr_np: float = 1e-3,
        entropy_weight: float = 0.05,
        initial_random: int = 20,
        device: str = "cpu",
    ) -> None:
        super().__init__("neural_process", budget, model, device)
        self.np = np_sampler.to(device)
        self.batch_size = batch_size
        self.total_theta_steps = total_theta_steps
        self.n_np_steps_per_outer = n_np_steps_per_outer
        self.lr_theta = lr_theta
        self.lr_np = lr_np
        self.entropy_weight = entropy_weight
        self.initial_random = initial_random

    def _build_context(self, x_hist: list[np.ndarray], y_hist: list[np.ndarray]) -> torch.Tensor:
        """Build (N, 2) context tensor from history."""
        if not x_hist:
            return torch.empty(0, 2, device=self.device)
        x_all = np.concatenate(x_hist)
        y_all = np.concatenate(y_hist)
        ctx = np.stack([x_all, y_all], axis=1).astype(np.float32)
        return torch.from_numpy(ctx).to(self.device)

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        n_outer = max(1, self.budget // self.batch_size)
        theta_steps_per_outer = max(1, self.total_theta_steps // n_outer)

        opt_theta = torch.optim.Adam(self.model.parameters(), lr=self.lr_theta)
        opt_np = torch.optim.Adam(self.np.parameters(), lr=self.lr_np)
        WARMUP, DECAY = 10, 0.999

        l2_history: list[float] = []
        all_points: list[np.ndarray] = []
        all_values: list[np.ndarray] = []

        self.model.train()
        self.np.train()
        log_freq = max(1, n_outer // 10)
        step_counter = 0

        for outer in range(n_outer):
            # ---- 1. Build context from history ----
            ctx = self._build_context(all_points, all_values)

            # ---- 2. Sample new points ----
            if ctx.shape[0] < self.initial_random:
                # Early rounds: uniform exploration (no NP yet)
                x_new = torch.rand(self.batch_size, device=self.device)
                y_new = torch.from_numpy(
                    task.source_values(x_new.cpu().numpy()).astype(np.float32)
                ).to(self.device)
            else:
                # NP proposes points (hard sample for function eval)
                with torch.no_grad():
                    x_new, _ = self.np.sample(ctx, self.batch_size, hard=True)
                y_new = torch.from_numpy(
                    task.source_values(x_new.cpu().numpy()).astype(np.float32)
                ).to(self.device)

            all_points.append(x_new.cpu().numpy())
            all_values.append(y_new.cpu().numpy())

            # ---- 3. Train θ on all history ----
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

            # ---- 4. Train NP via REINFORCE + reparam ----
            if ctx.shape[0] >= self.initial_random:
                for _ in range(self.n_np_steps_per_outer):
                    opt_np.zero_grad()

                    # Soft sample from NP (Gumbel-Softmax for differentiability)
                    x_soft, log_p = self.np.sample(ctx, self.batch_size, hard=False)
                    per_point_err = task.pointwise_loss(self.model, x_soft)

                    # Objective: maximise expected error (same as adversarial)
                    loss_np = -torch.mean(per_point_err)
                    # Entropy bonus
                    probs = F.softmax(self.np(ctx), dim=0)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum()
                    (loss_np - self.entropy_weight * entropy).backward()
                    opt_np.step()

            # ---- Log ----
            err = task.compute_l2_error(self.model, eval_grid)
            l2_history.append(err)

            if outer % log_freq == 0:
                pass  # already logged above

        pts = np.unique(np.concatenate(all_points)) if all_points else np.array([])

        return AlgorithmResult(
            algorithm_name=self.algorithm_name,
            function_name=function_name,
            final_l2_error=l2_history[-1] if l2_history else float("inf"),
            l2_error_history=l2_history,
            sampling_points=pts,
            extra_info={"n_context": len(pts)},
            trained_model=self.model,
        )
