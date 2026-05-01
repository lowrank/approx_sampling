"""
Energy-based adaptive sampling algorithm.

Inspired by the Deep Ritz method, this algorithm samples from a piecewise-
constant distribution whose bin probabilities are updated each round to
target regions where a local difficulty metric is large:

    difficulty(x) = |f(x) − f_θ(x)|² + α · |f'(x) − f_θ'(x)|²

This is an empirical adaptive method (no adversarial training) that
concentrates samples in regions of high approximation error and steep
gradient mismatch.

To ensure stable training, a minimum number of points is guaranteed in
each bin, with additional points distributed according to the current
difficulty estimate.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import (
    AlgorithmResult,
    BaseSamplingAlgorithm,
    _to_torch_func,
)
from models.approximator import MLP


class EnergyBasedSampling(BaseSamplingAlgorithm):
    """Energy-based adaptive sampling.

    Parameters
    ----------
    budget : int
        Total function evaluations.
    model : MLP
        Approximator network.
    n_rounds : int
        Number of adaptation rounds.
    n_bins : int
        Number of equal-width bins for the piecewise-constant density.
    alpha_deriv : float
        Weight of the derivative term in the local difficulty metric.
    candidate_size : int
        Dense grid size for evaluating local difficulty.
    epochs_per_round : int
        Training epochs per round.
    lr : float
        Learning rate for Adam.
    batch_size : int
        Minibatch size.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        n_rounds: int = 8,
        n_bins: int = 32,
        alpha_deriv: float = 0.1,
        candidate_size: int = 2000,
        epochs_per_round: int = 500,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        super().__init__("energy_based", budget, model, device)
        self.n_rounds = n_rounds
        self.n_bins = n_bins
        self.alpha_deriv = alpha_deriv
        self.candidate_size = candidate_size
        self.epochs_per_round = epochs_per_round
        self.lr = lr
        self.batch_size = batch_size

    def run(
        self,
        f_target: Callable[[np.ndarray], np.ndarray],
        f_target_torch: Callable[[torch.Tensor], torch.Tensor],
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        y_eval = f_target(eval_grid)
        n_per_round = self.budget // self.n_rounds
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        candidate = np.linspace(0.0, 1.0, self.candidate_size)

        # Uniform initial distribution
        bin_probs = np.ones(self.n_bins) / self.n_bins

        l2_history: list[float] = []
        all_points: list[np.ndarray] = []

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for rnd in range(self.n_rounds):
            # ---- Sample from current distribution ----
            # Multinomial allocation ensuring total = n_per_round
            if n_per_round >= self.n_bins:
                # Guarantee at least 1 point per bin for coverage
                min_per_bin = 1
                guaranteed = min_per_bin * self.n_bins
                leftover = n_per_round - guaranteed
                counts = np.full(self.n_bins, min_per_bin, dtype=int)
                if leftover > 0:
                    extra = np.random.multinomial(leftover, bin_probs)
                    counts = counts + extra
            else:
                # Spread as evenly as possible
                counts = np.random.multinomial(n_per_round, bin_probs)

            samples = []
            for b in range(self.n_bins):
                cnt = int(counts[b])
                if cnt > 0:
                    lo = bin_edges[b]
                    hi = bin_edges[b + 1]
                    pts = np.random.uniform(lo, hi, cnt)
                    samples.append(pts)
            x_rnd = np.concatenate(samples) if samples else np.array([])
            np.random.shuffle(x_rnd)
            all_points.append(x_rnd)

            y_rnd = f_target(x_rnd)

            # ---- Train on ALL accumulated points (replay buffer) ----
            self.model.train()
            # Concatenate all points collected so far
            all_x = np.concatenate(all_points) if all_points else x_rnd
            all_y = f_target(all_x) if len(all_points) > 1 else y_rnd
            x_t = torch.from_numpy(all_x.astype(np.float32)).to(self.device)
            y_t = torch.from_numpy(all_y.astype(np.float32)).to(self.device)
            n_buffer = len(all_x)
            dataset = TensorDataset(x_t, y_t)
            loader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, max(1, n_buffer)),
                shuffle=True,
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.epochs_per_round
            )
            for _ in range(self.epochs_per_round):
                for bx, by in loader:
                    opt.zero_grad()
                    pred = self.model(bx).squeeze(-1)
                    loss = torch.mean((pred - by) ** 2)
                    loss.backward()
                    opt.step()
                scheduler.step()

            # ---- Evaluate L^2 error ----
            err = self._compute_l2_error(eval_grid, y_eval)
            l2_history.append(err)

            # ---- Update bin probabilities from local difficulty ----
            self.model.eval()
            t_cand = torch.from_numpy(candidate.astype(np.float32)).to(self.device)
            t_cand.requires_grad_(True)
            pred = self.model(t_cand).squeeze(-1)
            y_cand = f_target(candidate)
            sq_err = (pred.detach().cpu().numpy() - y_cand) ** 2

            # Derivative mismatch
            grad_pred = torch.autograd.grad(
                pred.sum(), t_cand, create_graph=False
            )[0].detach().cpu().numpy()
            y_cand_grad = np.gradient(y_cand, candidate)
            deriv_mismatch = (grad_pred - y_cand_grad) ** 2

            difficulty = sq_err + self.alpha_deriv * deriv_mismatch
            difficulty = np.maximum(difficulty, 1e-12)

            # Aggregate into bins
            bin_idx = np.clip(
                np.searchsorted(bin_edges[1:], candidate, side="right"),
                0,
                self.n_bins - 1,
            )
            bin_sums = np.bincount(bin_idx, weights=difficulty, minlength=self.n_bins)
            bin_cnts = np.bincount(bin_idx, minlength=self.n_bins)
            avg_diff = np.divide(
                bin_sums, bin_cnts,
                out=np.ones_like(bin_sums),
                where=bin_cnts > 0,
            )

            # Smooth update via EMA
            new_probs = avg_diff / (avg_diff.sum() + 1e-12)
            bin_probs = 0.6 * bin_probs + 0.4 * new_probs
            bin_probs = bin_probs / bin_probs.sum()

            self.model.train()

        # Final evaluation
        err = self._compute_l2_error(eval_grid, y_eval)
        l2_history.append(err)

        pts = np.unique(np.concatenate(all_points)) if all_points else np.array([])

        return AlgorithmResult(
            algorithm_name=self.algorithm_name,
            function_name=function_name,
            final_l2_error=l2_history[-1],
            l2_error_history=l2_history,
            sampling_points=pts,
            extra_info={"final_bin_probs": bin_probs.tolist()},
            trained_model=self.model,
        )
