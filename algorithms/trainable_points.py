"""
Trainable-points sampling.

Treats the sampling locations {x_i} directly as optimisable parameters.
At each round, points are displaced in the direction of increasing
pointwise loss via the proxy gradient

    ∂L/∂x_i ≈ 2 (u_θ(x_i) − f(x_i)) · u_θ'(x_i),

computed by automatic differentiation through the approximator.
Points that reach the same neighbourhood are merged; merges free
budget for new exploratory points (uniform-random).

The method is end-to-end differentiable w.r.t. the sampling positions
and requires no generator network.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from algorithms.base import AlgorithmResult, BaseSamplingAlgorithm
from models.approximator import MLP


class TrainablePointsSampling(BaseSamplingAlgorithm):
    """Direct optimisation of sampling-point locations.

    Parameters
    ----------
    budget : int
        Total distinct function evaluations.
    model : MLP
        Approximator f_θ.
    batch_size : int
        Minibatch size for θ training.
    n_rounds : int
        Number of point-optimisation rounds.
    theta_steps_per_round : int
        Gradient updates for θ per round.
    point_lr : float
        Step size for point-displacement gradient ascent.
    merge_tol : float
        Points closer than this are merged.
    lr_theta : float
        Learning rate for θ.
    device : str
    """

    def __init__(
        self,
        budget: int,
        model: MLP,
        batch_size: int = 32,
        n_rounds: int = 20,
        theta_steps_per_round: int = 200,
        point_lr: float = 0.005,
        merge_tol: float = 0.01,
        lr_theta: float = 1.5e-3,
        device: str = "cpu",
    ) -> None:
        super().__init__("trainable_points", budget, model, device)
        self.batch_size = batch_size
        self.n_rounds = n_rounds
        self.theta_steps_per_round = theta_steps_per_round
        self.point_lr = point_lr
        self.merge_tol = merge_tol
        self.lr_theta = lr_theta

    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        y_eval = task.reference(eval_grid)
        N = self.budget

        # Initialise points uniformly
        x_np = np.linspace(0.0, 1.0, N)
        y_np = task.source_values(x_np)
        f_np = task.reference  # direct numpy callable for efficiency

        opt_theta = torch.optim.Adam(self.model.parameters(), lr=self.lr_theta)
        WARMUP, DECAY = 10, 0.999
        step_counter = 0

        l2_history: list[float] = []

        self.model.train()

        for rnd in range(self.n_rounds):
            # ---- 1. Train θ on current points (importance-weighted) ----
            w_np = BaseSamplingAlgorithm._is_weights(x_np)
            x_t = torch.from_numpy(x_np.astype(np.float32)).to(self.device)
            y_t = torch.from_numpy(y_np.astype(np.float32)).to(self.device)
            w_t = w_np.to(self.device)
            n_pts = len(x_np)

            for _ in range(self.theta_steps_per_round):
                idx = torch.randperm(n_pts, device=self.device)[
                    :min(self.batch_size, n_pts)
                ]
                bx, by, bw = x_t[idx], y_t[idx], w_t[idx]
                opt_theta.zero_grad()
                pred = self.model(bx).squeeze(-1)
                loss = torch.mean(bw * (pred - by) ** 2)
                loss.backward()
                opt_theta.step()
                step_counter += 1
                if step_counter >= WARMUP:
                    for g in opt_theta.param_groups:
                        g["lr"] = self.lr_theta * (DECAY ** (step_counter - WARMUP))

            # ---- 2. Move points toward higher-error regions ----
            x_var = torch.from_numpy(x_np.astype(np.float32)).to(self.device)
            x_var.requires_grad_(True)

            pred = self.model(x_var).squeeze(-1)
            # Model derivative u'(x)
            u_x = torch.autograd.grad(
                pred.sum(), x_var, create_graph=False, retain_graph=False
            )[0].detach()

            y_vals = torch.from_numpy(f_np(x_np).astype(np.float32)).to(self.device)
            error = pred.detach() - y_vals

            # Proxy gradient: d/dx [ (u - f)^2 ] ≈ 2 (u - f) u'
            grad = 2.0 * error * u_x

            with torch.no_grad():
                x_new = x_np + self.point_lr * grad.cpu().numpy()
                x_new = np.clip(x_new, 0.0, 1.0)

            # ---- 3. Merge close points, add exploratory ones ----
            x_new_sorted = np.sort(x_new)
            merged = [x_new_sorted[0]]
            for xi in x_new_sorted[1:]:
                if xi - merged[-1] > self.merge_tol:
                    merged.append(xi)
                else:
                    merged[-1] = 0.5 * (merged[-1] + xi)  # average
            merged = np.array(merged)

            if len(merged) < len(x_new):
                n_explore = len(x_new) - len(merged)
                explore = np.random.uniform(0, 1, n_explore)
                x_new = np.concatenate([merged, explore])
            else:
                x_new = merged

            x_np = np.clip(x_new, 0.0, 1.0)
            y_np = f_np(x_np)

            # ---- 4. Log ----
            err = task.compute_l2_error(self.model, eval_grid)
            l2_history.append(err)
            if rnd % max(1, self.n_rounds // 5) == 0:
                print(f"    rnd={rnd:2d}  pts={len(x_np):3d}  L2={err:.6f}")

        return AlgorithmResult(
            algorithm_name=self.algorithm_name,
            function_name=function_name,
            final_l2_error=l2_history[-1] if l2_history else float("inf"),
            l2_error_history=l2_history,
            sampling_points=x_np,
            extra_info={"final_n_points": len(x_np)},
            trained_model=self.model,
        )
