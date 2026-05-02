"""
Abstract base class for sampling algorithms.

Every algorithm receives a target function and a sampling budget, and
must produce a trained approximator together with diagnostic information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.integrate import trapezoid

from models.approximator import MLP


@dataclass
class AlgorithmResult:
    """Container returned by every algorithm's ``.run()`` method.

    Attributes
    ----------
    algorithm_name : str
        Human-readable algorithm identifier.
    function_name : str
        Name of the target function.
    final_l2_error : float
        L^2[0, 1] error of the trained model on a dense reference grid.
    l2_error_history : list of float
        L^2 error recorded at each checkpoint during training.
    sampling_points : np.ndarray
        All distinct x-values where the target was evaluated.
    extra_info : dict
        Algorithm-specific diagnostics (e.g., generator probabilities).
    trained_model : MLP
        The final trained MLP.
    """

    algorithm_name: str
    function_name: str
    final_l2_error: float
    l2_error_history: List[float] = field(default_factory=list)
    sampling_points: np.ndarray = field(default_factory=lambda: np.array([]))
    extra_info: Dict[str, Any] = field(default_factory=dict)
    trained_model: Optional[MLP] = None


class BaseSamplingAlgorithm(ABC):
    """Abstract interface for a sampling strategy.

    Subclasses must implement :meth:`run`.

    Parameters
    ----------
    algorithm_name : str
        Unique identifier (used in result tables).
    budget : int
        Total number of function evaluations permitted.
    model : MLP
        Trainable network.
    device : str
        PyTorch device (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        algorithm_name: str,
        budget: int,
        model: MLP,
        device: str = "cpu",
    ) -> None:
        self.algorithm_name = algorithm_name
        self.budget = budget
        self.model = model.to(device)
        self.device = device

    @abstractmethod
    def run(
        self,
        task: Any,
        eval_grid: np.ndarray,
        function_name: str,
    ) -> AlgorithmResult:
        """Execute the sampling + training loop.

        Parameters
        ----------
        task : DownstreamTask
            Encapsulates target evaluation, pointwise loss, and L² error.
        eval_grid : np.ndarray
            Dense grid on [0, 1] for L^2 error computation.
        function_name : str
            Name of the target function (for record-keeping).

        Returns
        -------
        AlgorithmResult
        """
        ...

    def _compute_l2_error(
        self,
        eval_grid: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Compute current L^2 error using the stored model."""
        self.model.eval()
        with torch.no_grad():
            t_grid = torch.from_numpy(eval_grid.astype(np.float32)).to(
                self.device
            )
            y_pred = self.model(t_grid).squeeze(-1).cpu().numpy()
        self.model.train()
        return float(np.sqrt(trapezoid((y_true - y_pred) ** 2, eval_grid)))

    @staticmethod
    def _is_weights(x_np: np.ndarray, n_bins: int = 64) -> torch.Tensor:
        """Importance weights: inverse empirical density (histogram-based).

        Returns weights ~ 1/p(x) normalised to mean 1.  Points in sparse
        regions get high weight; points in dense regions get low weight.
        """
        bins = np.linspace(0, 1, n_bins + 1)
        counts, _ = np.histogram(x_np, bins=bins)
        density = counts / (len(x_np) * (1.0 / n_bins)) + 1e-10
        idx = np.clip(np.searchsorted(bins[1:], x_np, side="right"), 0, n_bins - 1)
        w = 1.0 / density[idx]
        w = w / w.mean()
        return torch.from_numpy(w.astype(np.float32))



