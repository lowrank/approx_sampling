#!/usr/bin/env python3
r"""
Generate publication-quality figures from experiment results.

Produces, for each test function:
  - **Convergence curves**: L^2 error vs training progress.
  - **Sampling-point distributions**: rug-plot + histogram overlay showing
    where each algorithm allocated its budget on [0, 1].
  - **Summary bar chart** comparing final L^2 errors across all
    (function, algorithm) pairs.

Usage::

    python plot_results.py results latex/figures
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette for algorithms (consistent across all figures)
# ---------------------------------------------------------------------------
ALG_COLORS: Dict[str, str] = {
    "uniform": "#1f77b4",
    "chebyshev": "#ff7f0e",
    "qmc_sobol": "#2ca02c",
    "qmc_halton": "#17becf",
    "adaptive_residual": "#d62728",
    "adversarial": "#8c564b",
    "importance_sampling": "#e377c2",
    "diffusion": "#7f7f7f",
}

ALG_LABELS: Dict[str, str] = {
    "uniform": "Uniform",
    "chebyshev": "Chebyshev",
    "qmc_sobol": "QMC Sobol'",
    "qmc_halton": "QMC Halton",
    "adaptive_residual": "Adaptive Residual",
    "adversarial": "Adversarial",
    "importance_sampling": "Importance Sampling",
    "diffusion": "Diffusion (score)",
}

ALG_LABELS: Dict[str, str] = {
    "uniform": "Uniform",
    "chebyshev": "Chebyshev",
    "qmc_sobol": "QMC Sobol'",
    "qmc_halton": "QMC Halton",
    "adaptive_residual": "Adaptive Residual",
    "energy_based": "Energy-based",
    "adversarial": "Adversarial",
    "importance_sampling": "Importance Sampling",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_results(results_dir: str) -> Tuple[Dict, Dict, List[str], List[str]]:
    """Load experiment results and return metadata."""
    with open(os.path.join(results_dir, "all_results.json")) as f:
        data: Dict = json.load(f)

    functions = sorted(data.keys())
    algorithms = sorted(
        {alg for fname in functions for alg in data[fname]}
    )
    return data, functions, algorithms


def load_points(results_dir: str, func_name: str, alg_name: str) -> np.ndarray:
    """Load saved sampling points for a (function, algorithm) pair."""
    path = os.path.join(results_dir, f"pts_{func_name}_{alg_name}.npz")
    if os.path.exists(path):
        return np.load(path)["points"]
    return np.array([])


# ---------------------------------------------------------------------------
# Plot 1: Convergence curves per function
# ---------------------------------------------------------------------------


def plot_convergence(
    data: Dict,
    functions: List[str],
    algorithms: List[str],
    out_dir: str,
) -> None:
    """For each function, plot L^2 error history over checkpoints."""
    for func_name in functions:
        fig, ax = plt.subplots(figsize=(9, 4))
        for alg_name in algorithms:
            result = data[func_name].get(alg_name, {})
            history = result.get("l2_error_history", [])
            if not history:
                continue
            color = ALG_COLORS.get(alg_name, "#333333")
            label = ALG_LABELS.get(alg_name, alg_name)
            ax.plot(history, label=label, color=color, linewidth=1.5)

        ax.set_xlabel("Checkpoint", fontsize=10)
        ax.set_ylabel(r"$L^2$ error", fontsize=10)
        ax.set_title(f"Convergence — {func_name}", fontsize=11)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"conv_{func_name}.pdf"), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Sampling-point distribution per function
# ---------------------------------------------------------------------------


def plot_distributions(
    data: Dict,
    functions: List[str],
    algorithms: List[str],
    results_dir: str,
    out_dir: str,
) -> None:
    """For each function, show where each algorithm placed its samples."""
    n_alg = len(algorithms)

    for func_name in functions:
        fig, axes = plt.subplots(
            n_alg, 1, figsize=(10, 1.6 * n_alg), sharex=True
        )
        if n_alg == 1:
            axes = [axes]
        fig.suptitle(
            f"Sampling-point distributions — {func_name}",
            fontsize=12, y=1.0,
        )

        # Load target function for background
        target_grid = np.linspace(0, 1, 2000)
        from data.function_library import get_function
        tf = get_function(func_name)
        y_target = tf.evaluate(target_grid)
        y_norm = (y_target - y_target.min()) / (y_target.max() - y_target.min() + 1e-8)

        for i, alg_name in enumerate(algorithms):
            ax = axes[i]
            color = ALG_COLORS.get(alg_name, "#333333")
            label = ALG_LABELS.get(alg_name, alg_name)

            # Light background: target function shape
            ax.fill_between(
                target_grid, 0, y_norm,
                alpha=0.08, color="gray",
            )
            ax.plot(
                target_grid, y_norm,
                color="gray", alpha=0.3, linewidth=0.5,
            )

            # Load sampling points
            pts = load_points(results_dir, func_name, alg_name)

            # Rug plot + histogram
            if len(pts) > 0:
                # Histogram
                bins = np.linspace(0, 1, 51)
                ax.hist(
                    pts, bins=bins, color=color, alpha=0.6,
                    density=True, edgecolor="white", linewidth=0.3,
                )
                # Rug ticks at bottom
                ax.scatter(
                    pts, np.zeros_like(pts) - 0.02,
                    marker="|", s=20, color=color, alpha=0.4,
                    linewidths=0.5,
                )

            ax.set_ylabel(label, fontsize=8, rotation=0,
                          ha="right", va="center", labelpad=60)
            ax.set_yticks([])
            ax.set_ylim(-0.05, 1.15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

        axes[-1].set_xlabel("$x$", fontsize=10)
        axes[-1].set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"dist_{func_name}.pdf"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Combined summary (function approximation + sampling rug plot)
# ---------------------------------------------------------------------------


def plot_combined_summary(
    data: Dict,
    functions: List[str],
    algorithms: List[str],
    results_dir: str,
    models_dir: str,
    out_dir: str,
) -> None:
    """
    For each function produce a compact multi-panel figure:
      Left: target and best approximation
      Right: sampling-point rug plots for every algorithm.
    """
    n_alg = len(algorithms)
    from data.function_library import get_function

    for func_name in functions:
        tf = get_function(func_name)
        target_grid = np.linspace(0, 1, 500)
        y_target = tf.evaluate(target_grid)

        best_alg = None
        best_err = float("inf")
        for alg_name in algorithms:
            v = data[func_name].get(alg_name, {}).get("final_l2_error", float("inf"))
            if v == v and v < best_err:  # not NaN
                best_err = v
                best_alg = alg_name

        # Build figure: 1 row for function + approximation, N rows for distributions
        total_rows = 1 + n_alg
        fig = plt.figure(figsize=(12, 1.4 * total_rows))
        gs = fig.add_gridspec(total_rows, 1, hspace=0.15)

        # ---- Row 0: target function ----
        ax_main = fig.add_subplot(gs[0])
        ax_main.plot(target_grid, y_target, "k-", linewidth=1.2, label="Target")
        ax_main.set_ylabel(r"$f(x)$", fontsize=9)
        ax_main.set_title(f"{func_name}", fontsize=10, loc="left")
        ax_main.legend(fontsize=7, loc="upper right")
        ax_main.set_xlim(0, 1)
        ax_main.set_xticklabels([])

        # ---- Rows 1..N: sampling distributions ----
        for i, alg_name in enumerate(algorithms):
            ax = fig.add_subplot(gs[i + 1])
            color = ALG_COLORS.get(alg_name, "#333333")
            label = ALG_LABELS.get(alg_name, alg_name)
            err_val = data[func_name].get(alg_name, {}).get(
                "final_l2_error", float("nan")
            )
            err_str = f"{err_val:.4f}" if err_val == err_val else "N/A"

            pts = load_points(results_dir, func_name, alg_name)
            if len(pts) > 0:
                bins = np.linspace(0, 1, 41)
                ax.hist(
                    pts, bins=bins, color=color, alpha=0.55,
                    density=True, edgecolor="white", linewidth=0.3,
                )
                ax.scatter(
                    pts, np.zeros_like(pts) - 0.03,
                    marker="|", s=15, color=color, alpha=0.35, linewidths=0.5,
                )

            ax.set_ylabel(f"{label}", fontsize=7, rotation=0,
                          ha="right", va="center", labelpad=70)
            ax.set_yticks([])
            ax.set_ylim(-0.06, 1.2)
            ax.set_xlim(0, 1)
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)

            # Add L^2 error text
            ax.text(
                0.99, 0.85, f"L²={err_str}",
                transform=ax.transAxes, fontsize=7,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

        axes = fig.get_axes()
        axes[-1].set_xlabel("$x$", fontsize=10)

        fig.savefig(
            os.path.join(out_dir, f"summary_{func_name}.pdf"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Summary bar chart
# ---------------------------------------------------------------------------


def plot_bar_chart(
    data: Dict,
    functions: List[str],
    algorithms: List[str],
    out_dir: str,
) -> None:
    """Grouped bar chart of final L^2 error per (function, algorithm)."""
    n_func = len(functions)
    n_alg = len(algorithms)
    x = np.arange(n_func)
    width = 0.8 / n_alg

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, alg_name in enumerate(algorithms):
        errors = []
        for func_name in functions:
            v = data[func_name].get(alg_name, {}).get("final_l2_error", float("nan"))
            errors.append(v if v == v else 0)
        offset = (i - (n_alg - 1) / 2) * width
        color = ALG_COLORS.get(alg_name, "#333333")
        label = ALG_LABELS.get(alg_name, alg_name)
        ax.bar(x + offset, errors, width, label=label, color=color, alpha=0.85)

    ax.set_ylabel(r"$L^2[0,1]$ error", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(functions, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_barchart.pdf"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <results_dir> [out_dir]")
        sys.exit(1)

    results_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "latex/figures"
    os.makedirs(out_dir, exist_ok=True)

    data, functions, algorithms = load_results(results_dir)
    functions = sorted(functions)

    print(f"Functions: {len(functions)}, Algorithms: {len(algorithms)}")
    print(f"Output: {out_dir}")

    # Convergence curves
    print("  [1/4] Convergence curves ...")
    plot_convergence(data, functions, algorithms, out_dir)

    # Distribution plots
    print("  [2/4] Sampling-point distributions ...")
    plot_distributions(data, functions, algorithms, results_dir, out_dir)

    # Combined summary
    print("  [3/4] Combined summary figures ...")
    plot_combined_summary(data, functions, algorithms, results_dir, results_dir, out_dir)

    # Bar chart
    print("  [4/4] Summary bar chart ...")
    plot_bar_chart(data, functions, algorithms, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
