"""
Programmatic generation of diverse 1D test functions on [0, 1].

Generates three classes (200 functions each, unless otherwise specified):

* **smooth**       — polynomials, trigonometric polynomials, rationals,
                     exponentials; all smooth and slowly varying.
* **oscillatory**  — chirps, amplitude-modulated sinusoids, multi-frequency
                     combinations; high-frequency content everywhere.
* **sharp**        — localised Gaussian bumps, mollified steps, boundary
                     layers; smooth except for a few narrow features.

Every function is L^2-normalised so that

    ∫_0^1 f(x)^2 dx ≈ 1

computed via dense trapezoidal integration (4096 points).  The resolution
flag attached to each function is the number of equispaced points thought
sufficient to resolve it in the L^2 sense.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from data.function_library import TestFunction

# ---------------------------------------------------------------------------
# L^2 normalisation helper
# ---------------------------------------------------------------------------

_REF_GRID = np.linspace(0.0, 1.0, 4096)


def _l2_norm(f: Callable[[np.ndarray], np.ndarray]) -> float:
    """Estimate ``||f||_L^2[0,1]`` via trapezoidal rule on a fine grid."""
    y = f(_REF_GRID)
    return float(np.sqrt(np.trapz(y * y, _REF_GRID)))


def _normalise(f: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``f / ||f||_L^2`` (returns *f* unchanged if norm < 1e-12)."""
    norm = _l2_norm(f)
    if norm < 1e-12:
        return f

    def _fn(x: np.ndarray) -> np.ndarray:
        return f(x) / norm

    return _fn


def _estimate_resolution(
    f: Callable[[np.ndarray], np.ndarray],
    min_points: int = 100,
    max_points: int = 4000,
    tol: float = 1e-4,
) -> int:
    """Heuristic: smallest power-of-2 >= *min_points* where the L^2
    integral stabilises."""
    best = min_points
    ref = _l2_norm(f)
    for n in [128, 256, 512, 1024, 2048, 4000]:
        g = np.linspace(0.0, 1.0, n)
        y = f(g)
        approx = float(np.sqrt(np.trapz(y * y, g)))
        if abs(approx - ref) / (ref + 1e-12) < tol:
            best = n
            break
    return best


# ---------------------------------------------------------------------------
# Smooth class generators
# ---------------------------------------------------------------------------


def _make_smooth(i: int, rng: np.random.RandomState) -> Tuple[Callable, str]:
    """Return (f, label) for smooth function *i*."""
    t = i % 5
    if t == 0:  # polynomial
        degree = rng.randint(2, 7)
        coeffs = rng.uniform(-2, 2, degree + 1)
        lbl = f"poly deg={degree}"

        def f(x: np.ndarray) -> np.ndarray:
            return np.polyval(coeffs, x)
    elif t == 1:  # trigonometric polynomial
        K = rng.randint(1, 4)
        a = rng.uniform(-1, 1, K)
        b = rng.uniform(-1, 1, K)
        lbl_parts = []

        def f(x: np.ndarray) -> np.ndarray:
            s = np.zeros_like(x)
            for k in range(K):
                s += a[k] * np.sin(2 * math.pi * (k + 1) * x)
                s += b[k] * np.cos(2 * math.pi * (k + 1) * x)
            return s
        lbl = f"trig K={K}"
    elif t == 2:  # exponential
        a = rng.uniform(-3, 3)
        lbl = f"exp({a:.1f}x)"

        def f(x: np.ndarray) -> np.ndarray:
            return np.exp(a * x)
    elif t == 3:  # rational
        a = rng.uniform(0.5, 2)
        b = rng.uniform(-0.5, 0.5)
        c = rng.uniform(1, 4)
        lbl = f"1/(c+(x-b)^2)"

        def f(x: np.ndarray) -> np.ndarray:
            return 1.0 / (c + (x - b) ** 2) * a
    else:  # smooth composite
        a = rng.uniform(0.5, 2)
        b = rng.uniform(0, 2 * math.pi)
        lbl = f"sin(ax+b)·exp(-x)"

        def f(x: np.ndarray) -> np.ndarray:
            return np.sin(a * x + b) * np.exp(-x)
    return f, lbl


# ---------------------------------------------------------------------------
# Oscillatory class generators
# ---------------------------------------------------------------------------


def _make_oscillatory(i: int, rng: np.random.RandomState) -> Tuple[Callable, str]:
    t = i % 6
    if t == 0:  # chirp
        c = rng.uniform(0.3, 1.0)
        eps = rng.uniform(0.06, 0.15)
        lbl = f"sin(c/(x+eps))"

        def f(x: np.ndarray) -> np.ndarray:
            return np.sin(c / (x + eps))
    elif t == 1:  # amplitude-modulated
        omega = rng.uniform(6, 20) * math.pi
        a0 = rng.uniform(0.2, 1.5)
        b0 = rng.uniform(0.5, 2)
        lbl = f"exp(-bx)·sin(ωx)"

        def f(x: np.ndarray) -> np.ndarray:
            return a0 * np.exp(-b0 * x) * np.sin(omega * x)
    elif t == 2:  # multi-frequency superposition
        n_modes = rng.randint(3, 8)
        freqs = rng.uniform(1, 30, n_modes) * math.pi
        amps = rng.uniform(0.5, 1.5, n_modes)
        phases = rng.uniform(0, 2 * math.pi, n_modes)
        lbl = f"Σ a_k sin(ω_k x+φ_k), {n_modes} modes"

        def f(x: np.ndarray) -> np.ndarray:
            s = np.zeros_like(x)
            for k in range(n_modes):
                s += amps[k] * np.sin(freqs[k] * x + phases[k])
            return s
    elif t == 3:  # frequency-modulated
        fm = rng.uniform(3, 10)
        fc = rng.uniform(5, 15) * math.pi
        lbl = f"sin(ω_c x + fm·sin(πx))"

        def f(x: np.ndarray) -> np.ndarray:
            return np.sin(fc * x + fm * np.sin(math.pi * x))
    elif t == 4:  # product of high-freq modes
        omega1 = rng.uniform(5, 12) * math.pi
        omega2 = rng.uniform(5, 12) * math.pi
        lbl = f"sin(ω1 x)·cos(ω2 x)"

        def f(x: np.ndarray) -> np.ndarray:
            return np.sin(omega1 * x) * np.cos(omega2 * x)
    else:  # oscillatory on a nonlinear grid (compressed oscillations)
        k = rng.uniform(5, 12)
        alpha = rng.uniform(0.5, 0.9)
        lbl = f"sin(k·x^α)"

        def f(x: np.ndarray) -> np.ndarray:
            return np.sin(k * np.power(x, alpha))
    return f, lbl


# ---------------------------------------------------------------------------
# Sharp (localised) class generators
# ---------------------------------------------------------------------------


def _make_sharp(i: int, rng: np.random.RandomState) -> Tuple[Callable, str]:
    t = i % 5
    if t == 0:  # multiple Gaussian bumps
        n_bumps = rng.randint(1, 4)
        mus = rng.uniform(0.1, 0.9, n_bumps)
        sigmas = rng.uniform(0.01, 0.04, n_bumps)
        amps = rng.choice([-1, 1], n_bumps) * rng.uniform(0.5, 2, n_bumps)
        signs = "+" if all(a > 0 for a in amps) else "±"
        lbl = f"{n_bumps} Gaussian bumps ({signs}), σ~{sigmas.mean():.3f}"

        def f(x: np.ndarray) -> np.ndarray:
            s = np.zeros_like(x)
            for mu, sig, amp in zip(mus, sigmas, amps):
                s += amp * np.exp(-((x - mu) ** 2) / (2 * sig * sig))
            return s
    elif t == 1:  # mollified step
        x0 = rng.uniform(0.2, 0.8)
        sharpness = rng.uniform(15, 45)
        lbl = f"tanh(s·(x-x0)), s={sharpness:.0f}"

        def f(x: np.ndarray) -> np.ndarray:
            return np.tanh(sharpness * (x - x0))
    elif t == 2:  # boundary layers
        eps = rng.uniform(0.02, 0.06)
        a = rng.uniform(0.5, 2)
        b = rng.uniform(0.5, 2)
        lbl = f"boundary layer ε={eps:.3f}"

        def f(x: np.ndarray) -> np.ndarray:
            return a * np.exp(-x / eps) + b * np.exp((x - 1) / eps)
    elif t == 3:  # cusp-like (square-root singularity)
        x0 = rng.uniform(0.2, 0.8)
        lbl = f"sqrt(|x-x0|)"

        def f(x: np.ndarray) -> np.ndarray:
            return np.sqrt(np.abs(x - x0))
    else:  # sharp spike on smooth background
        x0 = rng.uniform(0.15, 0.85)
        sigma = rng.uniform(0.005, 0.03)
        amp = rng.uniform(2, 8)
        bg_freq = rng.uniform(1, 4) * math.pi
        lbl = f"smooth bg + Gaussian spike, σ={sigma:.4f}"

        def f(x: np.ndarray) -> np.ndarray:
            bg = np.sin(bg_freq * x)
            spike = amp * np.exp(-((x - x0) ** 2) / (2 * sigma * sigma))
            return bg + spike
    return f, lbl


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_function_library(
    n_per_class: int = 200,
    seed: int = 42,
    split: str = "all",
    split_frac: float = 0.7,
) -> Dict[str, TestFunction]:
    """Generate a dictionary of L^2-normalised test functions.

    Parameters
    ----------
    n_per_class : int
        Number of functions per class.
    seed : int
        Random seed for reproducibility.
    split : str
        ``"train"`` → first ``split_frac`` fraction of each class;
        ``"test"``  → remaining fraction;
        ``"all"``   → all functions.
    split_frac : float
        Fraction used for training (default 0.7).

    Returns
    -------
    dict[str, TestFunction]
    """
    rng = np.random.RandomState(seed)
    functions: Dict[str, TestFunction] = {}
    n_digits = max(3, len(str(n_per_class)))

    n_train = int(n_per_class * split_frac)

    for cls_name, generator in [
        ("smooth", _make_smooth),
        ("oscillatory", _make_oscillatory),
        ("sharp", _make_sharp),
    ]:
        for idx in range(n_per_class):
            if split == "train" and idx >= n_train:
                continue
            if split == "test" and idx < n_train:
                continue
            raw_f, label = generator(idx, rng)
            f_norm = _normalise(raw_f)
            resolution = _estimate_resolution(f_norm)
            key = f"{cls_name}_{idx:0{n_digits}d}"
            functions[key] = TestFunction(
                name=key, label=label, callable=f_norm,
                dense_resolution=resolution,
            )

    return functions


def get_class(name: str) -> str:
    """Return ``"smooth"``, ``"oscillatory"``, or ``"sharp"``
    given a generated function name."""
    for cls in ("smooth", "oscillatory", "sharp"):
        if name.startswith(cls):
            return cls
    return "unknown"
