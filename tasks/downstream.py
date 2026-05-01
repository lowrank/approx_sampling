"""
Downstream tasks for neural network approximation and PDE solving.

* ``FunctionApproximation`` — fit ``u_θ(x) ≈ f(x)`` in L².
* ``PoissonPINN``         — solve ``−Δu = f`` with homogeneous Dirichlet BCs
                            via physics-informed neural networks.
* ``DeepRitz``            — minimise the Dirichlet energy
                            ``E(u) = ∫[½|∇u|² − f·u] dx``.

All tasks expose the :class:`DownstreamTask` interface defined in
:mod:`tasks.base`.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

from tasks.base import DownstreamTask, register_task


# ======================================================================
# Helper: 1D FEM solver for −u'' = f,  u(0)=u(1)=0
# ======================================================================


def solve_poisson_fem(
    f_np: Callable[[np.ndarray], np.ndarray],
    n_elements: int = 2000,
) -> Callable[[np.ndarray], np.ndarray]:
    """Solve −u'' = f on [0,1] with u(0)=u(1)=0 using linear FEM.

    Returns an interpolation callable that evaluates the solution at
    arbitrary points in [0,1].
    """
    n_nodes = n_elements + 1
    h = 1.0 / n_elements
    nodes = np.linspace(0.0, 1.0, n_nodes)

    # Assemble stiffness matrix K and load vector F
    # Local stiffness: (1/h) * [[1, -1], [-1, 1]]
    # Local mass (for load): (h/6) * [[2, 1], [1, 2]]
    main_diag = (2.0 / h) * np.ones(n_nodes)
    off_diag = (-1.0 / h) * np.ones(n_nodes - 1)
    K = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format="csr")

    # Lumped mass matrix for simple load integration
    f_vals = f_np(nodes[1:-1])  # interior nodes only
    F = h * f_vals  # lumped integration

    # Apply Dirichlet BCs
    u = np.zeros(n_nodes)
    u[1:-1] = spsolve(K[1:-1, 1:-1], F)

    # Interpolation callable
    def u_ref(x: np.ndarray) -> np.ndarray:
        return np.interp(x, nodes, u)

    return u_ref


# ======================================================================
# 1. Function Approximation
# ======================================================================


class FunctionApproximation(DownstreamTask):
    """Fit ``u_θ(x) ≈ f(x)`` in the L² sense.

    Parameters
    ----------
    f_target : callable (numpy)
        Target function ``f: [0,1] → ℝ``.
    f_target_torch : callable (torch), optional
        Torch version of *f_target*.  If ``None``, a wrapper is built.
    label : str
        Descriptive label for logging.
    """

    def __init__(
        self,
        f_target: Callable[[np.ndarray], np.ndarray],
        f_target_torch: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        label: str = "function_approx",
    ) -> None:
        super().__init__("function_approximation")
        self.f_np = f_target
        self.label = label
        if f_target_torch is None:
            def _wrap(x: torch.Tensor) -> torch.Tensor:
                return torch.from_numpy(
                    f_target(x.detach().cpu().numpy()).astype(np.float32)
                ).to(x.device)
            self.f_torch = _wrap
        else:
            self.f_torch = f_target_torch

    def pointwise_loss(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        pred = model(x).squeeze(-1)
        target = self.f_torch(x)
        return (pred - target) ** 2

    def reference(self, x: np.ndarray) -> np.ndarray:
        return self.f_np(x)

    def predict(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return model(x).squeeze(-1)


register_task("function_approximation", FunctionApproximation)


# ======================================================================
# 2. Poisson PINN  (−Δu = f,  u(0)=u(1)=0)
# ======================================================================


class PoissonPINN(DownstreamTask):
    r"""Solve ``−u'' = f`` on ``[0,1]`` with ``u(0)=u(1)=0`` via PINN.

    The network is hard-constrained to satisfy the boundary conditions:
    ``u_θ(x) = x(1−x) · NN_θ(x)``.  The residual loss is

        r(x) = −u_θ''(x) − f(x).

    The reference solution is computed via linear FEM on a fine mesh.

    Parameters
    ----------
    f_np : callable (numpy)
        Right-hand side ``f(x)``.
    n_fem_elements : int
        Number of elements for the FEM reference solution.
    label : str
    """

    def __init__(
        self,
        f_np: Callable[[np.ndarray], np.ndarray],
        n_fem_elements: int = 2000,
        label: str = "poisson_pinn",
    ) -> None:
        super().__init__("poisson_pinn")
        self.f_np = f_np
        self.label = label
        self._u_ref = solve_poisson_fem(f_np, n_fem_elements)

        def _f_torch(x: torch.Tensor) -> torch.Tensor:
            return torch.from_numpy(
                f_np(x.detach().cpu().numpy()).astype(np.float32)
            ).to(x.device)
        self.f_torch = _f_torch

    def predict(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Hard-constrained: u(x) = x(1−x)·NN(x)."""
        raw = model(x).squeeze(-1)
        return x.squeeze(-1) * (1.0 - x.squeeze(-1)) * raw

    def pointwise_loss(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x_in = x.detach().clone().requires_grad_(True)
        if x_in.dim() == 1:
            x_in = x_in.unsqueeze(-1)
        # u = x*(1-x)*NN(x)
        raw = model(x_in).squeeze(-1)
        x1d = x_in.squeeze(-1)
        u = x1d * (1.0 - x1d) * raw

        # u'' via autograd
        u_x = torch.autograd.grad(
            u.sum(), x_in, create_graph=True, retain_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x.sum(), x_in, create_graph=True, retain_graph=True
        )[0].squeeze(-1)

        # Residual: −u'' − f
        f_val = self.f_torch(x)
        residual = -u_xx - f_val
        return residual ** 2

    def reference(self, x: np.ndarray) -> np.ndarray:
        return self._u_ref(x)


register_task("poisson_pinn", PoissonPINN)


# ======================================================================
# 3. Deep Ritz  (min E(u) = ∫ [½|∇u|² − f·u] dx)
# ======================================================================


class DeepRitz(DownstreamTask):
    r"""Minimise the Dirichlet energy for ``−Δu = f`` via the Deep Ritz method.

    The energy functional is

        E(u) = ∫₀¹ [½·(u'(x))² − f(x)·u(x)] dx.

    The network is hard-constrained as ``u(x) = x(1−x)·NN(x)``.
    The pointwise energy density is used as the training loss (not its
    square — the energy itself is the loss).  Sampling points where the
    energy density is large captures regions of high gradient.

    Parameters
    ----------
    f_np : callable (numpy)
    n_fem_elements : int
    label : str
    """

    def __init__(
        self,
        f_np: Callable[[np.ndarray], np.ndarray],
        n_fem_elements: int = 2000,
        label: str = "deep_ritz",
    ) -> None:
        super().__init__("deep_ritz")
        self.f_np = f_np
        self.label = label
        self._u_ref = solve_poisson_fem(f_np, n_fem_elements)

        def _f_torch(x: torch.Tensor) -> torch.Tensor:
            return torch.from_numpy(
                f_np(x.detach().cpu().numpy()).astype(np.float32)
            ).to(x.device)
        self.f_torch = _f_torch

    def predict(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        raw = model(x).squeeze(-1)
        return x.squeeze(-1) * (1.0 - x.squeeze(-1)) * raw

    def pointwise_loss(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x_in = x.detach().clone().requires_grad_(True)
        if x_in.dim() == 1:
            x_in = x_in.unsqueeze(-1)
        raw = model(x_in).squeeze(-1)
        x1d = x_in.squeeze(-1)
        u = x1d * (1.0 - x1d) * raw

        u_x = torch.autograd.grad(
            u.sum(), x_in, create_graph=True
        )[0].squeeze(-1)

        f_val = self.f_torch(x)
        # Pointwise energy density: ½|u'|² − f·u
        energy_density = 0.5 * u_x ** 2 - f_val * u
        return energy_density  # minimise → lower energy

    def reference(self, x: np.ndarray) -> np.ndarray:
        return self._u_ref(x)


register_task("deep_ritz", DeepRitz)
