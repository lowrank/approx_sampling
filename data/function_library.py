"""
Function library for neural network approximation experiments.

Provides a diverse collection of 1D test functions on [0, 1] with
various characteristics (smooth, oscillatory, localised, singular, mixed).
Each function exposes its analytical form, a string label, the number of
evaluation points needed for dense "ground-truth" resolution, and optional
sampling hints (e.g., where the function varies rapidly).

Typical usage::

    from data.function_library import FUNCTION_LIBRARY
    f = FUNCTION_LIBRARY["oscillatory_high_freq"]
    y = f.evaluate(x)   # x in [0, 1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.integrate import trapezoid


@dataclass
class TestFunction:
    r"""A 1D test function with metadata.

    Parameters
    ----------
    name : str
        Unique identifier, e.g. ``"smooth_sinusoid"``.
    label : str
        Human-readable label for plots / LaTeX, e.g. ``r"$\sin(2\pi x)$"``.
    callable : Callable[[np.ndarray], np.ndarray]
        Vectorised function ``y = f(x)``.
    dense_resolution : int
        Number of equispaced points considered sufficient to resolve the
        function in the L^2 sense (used for reference integrals).
    local_variation : Optional[Callable[[np.ndarray], np.ndarray]]
        Optional estimate of local difficulty, e.g. ``|f''(x)|``, for
        oracle-guided sampling methods.
    """

    name: str
    label: str
    callable: Callable[[np.ndarray], np.ndarray]
    dense_resolution: int
    local_variation: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the function at points *x*."""
        return self.callable(x)

    def __repr__(self) -> str:
        return f"TestFunction({self.name})"


# ---------------------------------------------------------------------------
# Raw function definitions
# ---------------------------------------------------------------------------

def _smooth_sinusoid(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * math.pi * x)


def _smooth_poly(x: np.ndarray) -> np.ndarray:
    return 4.0 * x * (1.0 - x)


def _smooth_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(x) * np.sin(math.pi * x)


def _oscillatory_high_freq(x: np.ndarray) -> np.ndarray:
    return np.sin(8.0 * math.pi * x) + 0.5 * np.cos(16.0 * math.pi * x)


def _oscillatory_chirp(x: np.ndarray) -> np.ndarray:
    return np.sin(1.0 / (x + 0.02))


def _local_gaussians(x: np.ndarray) -> np.ndarray:
    return np.exp(-100.0 * (x - 0.3) ** 2) + np.exp(-200.0 * (x - 0.7) ** 2)


def _local_discontinuity(x: np.ndarray) -> np.ndarray:
    return np.tanh(40.0 * (x - 0.5))


def _boundary_layer(x: np.ndarray) -> np.ndarray:
    eps = 0.02
    return np.exp(-x / eps) + np.exp((x - 1.0) / eps)


def _multiscale(x: np.ndarray) -> np.ndarray:
    return (
        np.sin(2.0 * math.pi * x)
        + 0.2 * np.sin(20.0 * math.pi * x)
        + 0.1 * np.sin(50.0 * math.pi * x)
    )


def _lorentzian(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 100.0 * (x - 0.5) ** 2)


def _mixed_osc_decay(x: np.ndarray) -> np.ndarray:
    return np.sin(10.0 * math.pi * x) * np.exp(-3.0 * x)


def _cusp(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(x - 0.4)) * np.sin(6.0 * math.pi * x)


def _wide_bump(x: np.ndarray) -> np.ndarray:
    return np.exp(-20.0 * (x - 0.55) ** 2) * np.sin(12.0 * math.pi * x)


# ---------------------------------------------------------------------------
# Local-variation helpers (oracle difficulty indicators)
# ---------------------------------------------------------------------------


def _lv_sinusoid(x: np.ndarray) -> np.ndarray:
    return np.abs(2.0 * math.pi * np.cos(2.0 * math.pi * x))


def _lv_high_freq(x: np.ndarray) -> np.ndarray:
    return np.abs(
        -(8.0 * math.pi) ** 2 * np.sin(8.0 * math.pi * x)
        - 0.5 * (16.0 * math.pi) ** 2 * np.cos(16.0 * math.pi * x)
    )


def _lv_gaussians(x: np.ndarray) -> np.ndarray:
    g1 = np.exp(-100.0 * (x - 0.3) ** 2)
    g2 = np.exp(-200.0 * (x - 0.7) ** 2)
    dg1 = -200.0 * (x - 0.3) * g1
    dg2 = -400.0 * (x - 0.7) * g2
    return np.abs(dg1 + dg2)


def _lv_boundary(x: np.ndarray) -> np.ndarray:
    eps = 0.02
    return np.abs(-np.exp(-x / eps) / eps + np.exp((x - 1.0) / eps) / eps)


def _lv_multiscale(x: np.ndarray) -> np.ndarray:
    d1 = 2.0 * math.pi * np.cos(2.0 * math.pi * x)
    d2 = 0.2 * 20.0 * math.pi * np.cos(20.0 * math.pi * x)
    d3 = 0.1 * 50.0 * math.pi * np.cos(50.0 * math.pi * x)
    return np.abs(d1 + d2 + d3)


def _lv_lorentzian(x: np.ndarray) -> np.ndarray:
    d = -200.0 * (x - 0.5) / (1.0 + 100.0 * (x - 0.5) ** 2) ** 2
    return np.abs(d)


# ---------------------------------------------------------------------------
# Assemble the library
# ---------------------------------------------------------------------------

FUNCTION_LIBRARY: Dict[str, TestFunction] = {
    # ---- Smooth ----
    "smooth_sinusoid": TestFunction(
        name="smooth_sinusoid",
        label=r"$\sin(2\pi x)$",
        callable=_smooth_sinusoid,
        dense_resolution=200,
        local_variation=_lv_sinusoid,
    ),
    "smooth_poly": TestFunction(
        name="smooth_poly",
        label=r"$4x(1-x)$",
        callable=_smooth_poly,
        dense_resolution=200,
        local_variation=None,
    ),
    "smooth_exp": TestFunction(
        name="smooth_exp",
        label=r"$e^{x}\sin(\pi x)$",
        callable=_smooth_exp,
        dense_resolution=400,
        local_variation=None,
    ),
    # ---- Oscillatory ----
    "oscillatory_high_freq": TestFunction(
        name="oscillatory_high_freq",
        label=r"$\sin(8\pi x) + 0.5\cos(16\pi x)$",
        callable=_oscillatory_high_freq,
        dense_resolution=800,
        local_variation=_lv_high_freq,
    ),
    "oscillatory_chirp": TestFunction(
        name="oscillatory_chirp",
        label=r"$\sin\left(\frac{1}{x+0.02}\right)$",
        callable=_oscillatory_chirp,
        dense_resolution=2000,
        local_variation=None,
    ),
    "multiscale": TestFunction(
        name="multiscale",
        label=r"$\sum_{k\in\{1,10,25\}} a_k\sin(2k\pi x)$",
        callable=_multiscale,
        dense_resolution=1600,
        local_variation=_lv_multiscale,
    ),
    # ---- Localised ----
    "local_gaussians": TestFunction(
        name="local_gaussians",
        label=r"$e^{-100(x-0.3)^2}+e^{-200(x-0.7)^2}$",
        callable=_local_gaussians,
        dense_resolution=1200,
        local_variation=_lv_gaussians,
    ),
    "local_discontinuity": TestFunction(
        name="local_discontinuity",
        label=r"$\tanh(40(x-0.5))$",
        callable=_local_discontinuity,
        dense_resolution=1000,
        local_variation=None,
    ),
    "boundary_layer": TestFunction(
        name="boundary_layer",
        label=r"$e^{-x/\varepsilon}+e^{(x-1)/\varepsilon},\;\varepsilon=0.02$",
        callable=_boundary_layer,
        dense_resolution=2000,
        local_variation=_lv_boundary,
    ),
    "lorentzian": TestFunction(
        name="lorentzian",
        label=r"$\frac{1}{1+100(x-0.5)^2}$",
        callable=_lorentzian,
        dense_resolution=800,
        local_variation=_lv_lorentzian,
    ),
    # ---- Mixed / composite ----
    "mixed_osc_decay": TestFunction(
        name="mixed_osc_decay",
        label=r"$\sin(10\pi x)e^{-3x}$",
        callable=_mixed_osc_decay,
        dense_resolution=800,
        local_variation=None,
    ),
    "cusp": TestFunction(
        name="cusp",
        label=r"$\sqrt{|x-0.4|}\;\sin(6\pi x)$",
        callable=_cusp,
        dense_resolution=3000,
        local_variation=None,
    ),
    "wide_bump": TestFunction(
        name="wide_bump",
        label=r"$e^{-20(x-0.55)^2}\sin(12\pi x)$",
        callable=_wide_bump,
        dense_resolution=800,
        local_variation=None,
    ),
    "spike_bg_demo": TestFunction(
        name="spike_bg_demo",
        label=r"$\sin(2\pi x)+2e^{-200(x-0.3)^2}-1.5e^{-300(x-0.7)^2}$",
        callable=lambda x: (np.sin(2*np.pi*x) + 2*np.exp(-200*(x-0.3)**2) - 1.5*np.exp(-300*(x-0.7)**2)) / 1.3243,
        dense_resolution=2000,
        local_variation=None,
    ),
}


def list_function_names() -> List[str]:
    """Return sorted list of available function names."""
    return sorted(FUNCTION_LIBRARY.keys())


def get_function(name: str) -> TestFunction:
    """Retrieve a :class:`TestFunction` by name.

    Raises
    ------
    KeyError
        If *name* is not in the library.
    """
    if name not in FUNCTION_LIBRARY:
        available = ", ".join(FUNCTION_LIBRARY.keys())
        raise KeyError(f"Unknown function '{name}'. Available: {available}")
    return FUNCTION_LIBRARY[name]


def get_dense_grid(name: str) -> np.ndarray:
    """Return a dense equispaced grid appropriate for *name*."""
    func = get_function(name)
    return np.linspace(0.0, 1.0, func.dense_resolution)


def compute_l2_error(
    f_target: Callable[[np.ndarray], np.ndarray],
    f_approx: Callable[[np.ndarray], np.ndarray],
    grid: Optional[np.ndarray] = None,
    n_points: int = 2000,
) -> float:
    """Estimate the L^2[0, 1] error between *f_target* and *f_approx*.

    Parameters
    ----------
    f_target : callable
        Reference function.
    f_approx : callable
        Approximating function (e.g., a trained PyTorch model).
    grid : np.ndarray, optional
        Evaluation grid.  If ``None``, a uniform grid of size *n_points*
        is used.
    n_points : int
        Number of points when *grid* is ``None``.

    Returns
    -------
    float
        Approximate L^2 error ``sqrt(∫ (f_target − f_approx)^2 dx)``.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, n_points)
    y_true = f_target(grid)
    y_pred = f_approx(grid)
    # Trapezoidal integration
    return float(np.sqrt(trapezoid((y_true - y_pred) ** 2, grid)))
