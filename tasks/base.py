"""
Abstract base class for downstream tasks.

A *Task* encapsulates everything a sampling algorithm needs to know
about the problem being solved:

* how to compute the pointwise training loss,
* what the reference (ground-truth) solution is,
* how to compute the L² error of a model against the reference.

Subclasses implement function approximation, PINN-based PDE solving,
Deep Ritz energy minimisation, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import trapezoid


class DownstreamTask(ABC):
    """Abstract interface for a downstream task.

    Parameters
    ----------
    name : str
        Human-readable task identifier.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    # ------------------------------------------------------------------
    # Methods that every task must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def pointwise_loss(
        self, model: nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        """Scalar pointwise loss at each location *x* (shape ``(N,)``).

        For function approximation this is ``(u_θ(x) - f(x))²``.
        For PINN this is the PDE-residual squared (interior only).
        """
        ...

    def boundary_loss(self, model: nn.Module) -> torch.Tensor:
        """Optional boundary / regularisation loss (scalar).

        Default is zero.  PINN subclasses override this to add
        boundary-condition penalties.
        """
        return torch.tensor(0.0, device=next(model.parameters()).device)

    @abstractmethod
    def reference(self, x: np.ndarray) -> np.ndarray:
        """Reference (ground-truth) values at numpy points *x*.

        For function approximation this is ``f(x)``.
        For PDE problems this is the true solution ``u(x)``.
        """
        ...

    @abstractmethod
    def predict(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Model prediction at *x* (scalar, shape ``(N,)``).

        For hard-encoded boundary conditions this may wrap the raw
        network output with a cut-off factor like ``x(1-x)``.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def compute_l2_error(
        self,
        model: nn.Module,
        eval_grid: np.ndarray,
    ) -> float:
        """Estimate ``||predict - reference||_L²[0,1]``."""
        y_ref = self.reference(eval_grid)
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(eval_grid.astype(np.float32)).to(
                next(model.parameters()).device
            )
            y_pred = self.predict(model, t).cpu().numpy()
        model.train()
        return float(np.sqrt(trapezoid((y_pred - y_ref) ** 2, eval_grid)))

    def source_values(self, x: np.ndarray) -> np.ndarray:
        """Return the source / RHS ``f(x)`` at numpy points *x*.

        Used by algorithms that need to store (x, f(x)) in a replay
        buffer.  Default delegates to :meth:`reference` (which is the
        same as *f* for function approximation).
        """
        return self.reference(x)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, type[DownstreamTask]] = {}


def register_task(name: str, cls: type[DownstreamTask]) -> None:
    TASK_REGISTRY[name] = cls


def get_task_class(name: str) -> type[DownstreamTask]:
    if name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise KeyError(f"Unknown task '{name}'. Available: {available}")
    return TASK_REGISTRY[name]
