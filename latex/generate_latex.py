#!/usr/bin/env python3
r"""
Generate all assets required by the LaTeX document:

* ``results_table.tex`` — final-error table (from curves.json or dummy).
* ``figures/`` — convergence curves, sampling distributions, and
  function-illustration plots.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Colour palette (14 algorithms)
# ---------------------------------------------------------------------------
ALG_COLORS = {
    "adaptive_residual": "#000000",
    "uniform": "#1f77b4", "chebyshev": "#ff7f0e",
    "qmc_sobol": "#2ca02c", "qmc_halton": "#17becf",
    "iter_uniform": "#1f77b4", "iter_chebyshev": "#ff7f0e",
    "adversarial": "#8c564b", "importance_sampling": "#e377c2",
    "normalizing_flow": "#bcbd22", "mdn": "#9467bd",
    "svgd": "#7f7f7f", "ensemble": "#d62728",
    "gp_ucb": "#2ca02c", "policy": "#e377c2",
    "neural_process": "#17becf",
    "diffusion": "#ff7f0e",
}

ALG_LABELS = {
    "adaptive_residual": "Adaptive Res. (baseline)",
    "uniform": "Uniform", "chebyshev": "Chebyshev",
    "qmc_sobol": "QMC Sobol", "qmc_halton": "QMC Halton",
    "iter_uniform": "Iter. Uniform", "iter_chebyshev": "Iter. Chebyshev",
    "adversarial": "Adversarial", "importance_sampling": "Importance S.",
    "normalizing_flow": "Norm. Flow", "mdn": "MDN",
    "svgd": "SVGD", "ensemble": "Ensemble",
    "gp_ucb": "GP-UCB", "policy": "Policy",
    "neural_process": "Neural Process",
    "diffusion": "Diffusion",
}

RESULTS_DIR = "results" if len(sys.argv) < 2 else sys.argv[1]
OUT_DIR = "latex/figures" if len(sys.argv) < 3 else sys.argv[2]
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Results table (TeX)
# ---------------------------------------------------------------------------


def generate_results_table() -> None:
    curves_path = os.path.join(RESULTS_DIR, "curves.json")
    if os.path.exists(curves_path):
        with open(curves_path) as f:
            data = json.load(f)
    else:
        data = {}

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{$L^2[0,1]$ error at the largest budget.}")
    lines.append(r"  \label{tab:results}")
    lines.append(r"  \small")

    if not data:
        lines.append(r"  (No results yet — run \texttt{python main.py} first.)")
    else:
        # Sample evenly across classes: smooth, oscillatory, sharp, other
        all_fns = sorted(data.keys())
        classes: dict[str, list] = {"smooth": [], "oscillatory": [], "sharp": [], "other": []}
        for fn in all_fns:
            for cls in ("smooth", "oscillatory", "sharp"):
                if fn.startswith(cls):
                    classes[cls].append(fn)
                    break
            else:
                classes["other"].append(fn)
        functions = []
        per_class = 4
        for cls in ("smooth", "oscillatory", "sharp", "other"):
            functions.extend(classes[cls][:per_class])
        if not functions:
            functions = all_fns[:10]

        # Sort: adaptive_residual first (baseline), then alphabetical
        def _sort_key(a):
            return (0 if a == "adaptive_residual" else 1, a)
        algs = sorted({a for f in functions for a in data[f]}, key=_sort_key)[:8]
        col_spec = "l" + "c" * len(algs)
        lines.append(r"  \begin{tabular}{@{}" + col_spec + "@{}}")
        lines.append(r"    \toprule")
        hdr = "    Function"
        for a in algs:
            hdr += f" & \\textbf{{{ALG_LABELS.get(a, a)}}}"
        hdr += r" \\"
        lines.append(hdr)
        lines.append(r"    \midrule")
        for fn in functions:
            fn_esc = fn[:30].replace("_", r"\_")
            row = f"    {fn_esc}"
            best = min(
                (data[fn].get(a, {}).get("errors", [1])[-1] for a in algs),
                default=1,
            )
            for a in algs:
                errs = data[fn].get(a, {}).get("errors", [])
                if errs:
                    v = errs[-1]
                    m = f"$\\mathbf{{{v:.4f}}}$" if abs(v - best) < 1e-10 else f"{v:.4f}"
                    row += f" & {m}"
                else:
                    row += " & ---"
            row += r" \\"
            lines.append(row)
        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")

    lines.append(r"\end{table}")

    # ---- Per-class summary tables ----
    for cls_label, prefix in [("Smooth", "smooth_"), ("Oscillatory", "oscillatory_"),
                               ("Sharp", "sharp_"), ("Hardcoded", None)]:
        if prefix is not None:
            cls_fns = [fn for fn in all_fns if fn.startswith(prefix)]
        else:
            cls_fns = [fn for fn in all_fns
                       if not any(fn.startswith(p) for p in ("smooth_", "oscillatory_", "sharp_"))]
        if len(cls_fns) < 2:
            continue

        lines.append(r"\begin{table}[htbp]")
        lines.append(r"  \centering")
        lines.append(rf"  \caption{{{cls_label} functions — mean $L^2$ error at largest budget "
                     rf"({len(cls_fns)} functions).}}")
        lines.append(r"  \small")
        col_spec = "l" + "c" * len(algs)
        lines.append(r"  \begin{tabular}{@{}" + col_spec + "@{}}")
        lines.append(r"    \toprule")
        hdr = "    Algorithm"
        for a in algs:
            hdr += f" & \\textbf{{{ALG_LABELS.get(a, a)}}}"
        hdr += r" \\"
        lines.append(hdr)
        lines.append(r"    \midrule")

        # Average error per algorithm
        row_avg = "    Mean"
        row_std = "    Std"
        best_mean = float("inf")
        for a in algs:
            vals = []
            for fn in cls_fns:
                errs = data[fn].get(a, {}).get("errors", [])
                if errs and errs[-1] == errs[-1]:
                    vals.append(errs[-1])
            if vals:
                m = np.mean(vals)
                s = np.std(vals)
                if m < best_mean:
                    best_mean = m
                row_avg += f" & {m:.4f}"
                row_std += f" & $\\pm${s:.4f}"
            else:
                row_avg += " & ---"
                row_std += " & ---"
        row_avg += r" \\"; row_std += r" \\"
        lines.append(row_avg)
        lines.append(row_std)
        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")

    lines.append("")  # ensure trailing newline

    with open(os.path.join(os.path.dirname(OUT_DIR), "results_table.tex"), "w") as f:
        f.write("\n".join(lines))
    print("  → results_table.tex")


# ---------------------------------------------------------------------------
# 2. Convergence curves
# ---------------------------------------------------------------------------


def generate_convergence_curves() -> None:
    curves_path = os.path.join(RESULTS_DIR, "curves.json")
    if not os.path.exists(curves_path):
        print("  (no curves.json, skipping convergence plot)")
        return
    with open(curves_path) as f:
        data = json.load(f)

    functions = sorted(data.keys())[:6]
    for fn in functions:
        fig, ax = plt.subplots(figsize=(10, 5))
        for an, rec in data[fn].items():
            b, e = rec["budgets"], rec["errors"]
            if len(b) > 0:
                c = ALG_COLORS.get(an, "#333")
                ax.plot(b, e, "o-", color=c, label=ALG_LABELS.get(an, an),
                        markersize=4, linewidth=1.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("# samples", fontsize=11); ax.set_ylabel("L2 error", fontsize=11)
        ax.set_title(fn.replace("_", r"\_"), fontsize=12)
        ax.legend(fontsize=5.5, ncol=3, loc="upper right",
                  framealpha=0.8, borderpad=0.3, handlelength=1.2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"conv_{fn}.pdf"), dpi=150)
        plt.close(fig)

    # Summary multi-panel
    if len(functions) >= 2:
        n_cols = min(3, len(functions))
        n_rows = (len(functions) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for i, fn in enumerate(functions[:n_rows*n_cols]):
            ax = axes[i]
            for an, rec in data[fn].items():
                b, e = rec["budgets"], rec["errors"]
                if len(b) > 0:
                    ax.plot(b, e, "o-", color=ALG_COLORS.get(an, "#333"),
                            markersize=3, linewidth=1)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(fn, fontsize=9)
            ax.grid(True, alpha=0.3)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        handles, labels = [], set()
        for fn in functions[:n_rows*n_cols]:
            for an, rec in data[fn].items():
                lbl = ALG_LABELS.get(an, an)
                if lbl not in labels:
                    labels.add(lbl)
                    c = ALG_COLORS.get(an, "#333")
                    handles.append(plt.Line2D([0],[0], color=c, marker="o", markersize=3, linewidth=1))

        leg = fig.legend(handles, list(labels), fontsize=5.5, ncol=5,
                         loc="lower center", framealpha=0.8, borderpad=0.3,
                         handlelength=1.0)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        fig.savefig(os.path.join(OUT_DIR, "convergence_summary.pdf"), dpi=150,
                    bbox_extra_artists=(leg,), bbox_inches="tight")
        plt.close(fig)
    print("  → convergence plots")

    # Class-average convergence curves
    for cls_label, prefix in [("smooth", "smooth_"), ("oscillatory", "oscillatory_"),
                               ("sharp", "sharp_")]:
        cls_fns = [fn for fn in functions if fn.startswith(prefix)]
        if len(cls_fns) < 2:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        alg_to_errors: dict[str, list] = {}
        for fn in cls_fns:
            for an, rec in data[fn].items():
                if an not in alg_to_errors:
                    alg_to_errors[an] = []
                errs = rec.get("errors", [])
                if errs and errs[-1] == errs[-1]:
                    alg_to_errors[an].append(errs)
        # Plot mean ± std band
        common_budgets = None
        for an, err_lists in alg_to_errors.items():
            if not err_lists:
                continue
            max_len = max(len(e) for e in err_lists)
            padded = []
            for e in err_lists:
                if len(e) < max_len:
                    e2 = list(e) + [e[-1]] * (max_len - len(e))
                else:
                    e2 = list(e)
                padded.append(e2)
            arr = np.array(padded)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            b_vals = data[cls_fns[0]].get(an, {}).get("budgets", [])
            if common_budgets is None and b_vals:
                common_budgets = b_vals[:max_len]
            c = ALG_COLORS.get(an, "#333")
            ax.plot(common_budgets[:max_len], mean, color=c, linewidth=1.5,
                    label=ALG_LABELS.get(an, an))
            ax.fill_between(common_budgets[:max_len], mean - std, mean + std,
                            color=c, alpha=0.1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("# samples"); ax.set_ylabel("L2 error")
        ax.set_title(f"{cls_label} class — mean ± std ({len(cls_fns)} functions)")
        ax.legend(fontsize=5.5, ncol=3, loc="upper right", framealpha=0.8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"conv_class_{cls_label}.pdf"), dpi=150)
        plt.close(fig)
    print("  → class-average convergence plots")


# ---------------------------------------------------------------------------
# 3. Function-illustration plots (sharp class)
# ---------------------------------------------------------------------------


def generate_function_illustrations() -> None:
    from data.function_generator import generate_function_library
    lib = generate_function_library(n_per_class=200, seed=42)

    sharp_names = sorted([k for k in lib if k.startswith("sharp_")])

    # Pick diverse function types by scanning labels
    selected: list[str] = []
    categories = {
        "1 bump (+)": [], "1 bump (±)": [], "2 bumps": [], "3 bumps": [],
        "tanh step": [], "boundary": [], "cusp": [], "spike+bg": [],
    }
    for fn in sharp_names:
        lbl = lib[fn].label
        if "1 Gaussian" in lbl and "(+)" in lbl:       categories["1 bump (+)"].append(fn)
        elif "1 Gaussian" in lbl and "(±)" in lbl:      categories["1 bump (±)"].append(fn)
        elif "2 Gaussian" in lbl:                        categories["2 bumps"].append(fn)
        elif "3 Gaussian" in lbl:                        categories["3 bumps"].append(fn)
        elif "tanh" in lbl:                              categories["tanh step"].append(fn)
        elif "boundary" in lbl:                          categories["boundary"].append(fn)
        elif "sqrt" in lbl:                              categories["cusp"].append(fn)
        elif "spike" in lbl or "bg +" in lbl:            categories["spike+bg"].append(fn)

    for cat in ["1 bump (+)", "1 bump (±)", "2 bumps", "3 bumps",
                 "tanh step", "boundary", "cusp", "spike+bg"]:
        if categories[cat]:
            selected.append(categories[cat][0])
    if len(selected) < 4:
        selected = [sharp_names[i] for i in [0, 5, 21, 42, 88, 133, 177] if i < len(sharp_names)]

    x = np.linspace(0, 1, 2000)

    for fn in selected:
        func = lib[fn]
        y = func.evaluate(x)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.5),
                                        gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(x, y, "k-", linewidth=0.8)
        ax1.set_ylabel("f(x)")
        ax1.set_title(f"{fn} — {func.label}", fontsize=9)
        ax1.set_xlim(0, 1)

        # Local variation (|f'| via finite diff)
        dy = np.abs(np.gradient(y, x))
        dy = dy / (dy.max() + 1e-10)
        ax2.fill_between(x, 0, dy, alpha=0.5, color="tab:red")
        ax2.set_ylabel(r"$|f'|$ (norm.)", fontsize=8)
        ax2.set_xlabel("$x$")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.1)

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"illus_{fn}.pdf"), dpi=150)
        plt.close(fig)

    # Summary grid of sharp functions
    n_show = min(9, len(selected))
    n_cols = 3
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i in range(n_show):
        fn = selected[i]
        func = lib[fn]
        y = func.evaluate(x)
        ax = axes[i]
        ax.plot(x, y, "k-", linewidth=1)
        ax.set_title(fn, fontsize=8)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([])
    for j in range(n_show, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Sharp-function gallery (L²-normalised)", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sharp_gallery.pdf"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {n_show} sharp-function illustrations")

    # Also generate smooth and oscillatory galleries
    for cls_name, cls_label in [("smooth", "Smooth"), ("oscillatory", "Oscillatory")]:
        names = sorted([k for k in lib if k.startswith(f"{cls_name}_")])
        # Pick diverse: spread evenly across indices with some randomness
        n_avail = len(names)
        sel = [names[i] for i in range(0, n_avail, max(1, n_avail // 9))][:9]
        if len(sel) < 3:
            sel = names[:9]
        n_show2 = min(9, len(sel))
        fig2, axes2 = plt.subplots((n_show2 + 2) // 3, 3, figsize=(12, 3 * ((n_show2 + 2) // 3)))
        if isinstance(axes2, np.ndarray):
            axes2 = axes2.flatten()
        else:
            axes2 = [axes2]
        for i in range(n_show2):
            func = lib[sel[i]]
            axes2[i].plot(x, func.evaluate(x), "k-", linewidth=1)
            axes2[i].set_title(f"{sel[i]}  — {func.label}", fontsize=7)
            axes2[i].set_xticks([0, 0.5, 1])
            axes2[i].set_yticks([])
        for j in range(n_show2, len(axes2)):
            axes2[j].set_visible(False)
        fig2.suptitle(f"{cls_label}-function gallery (L²-normalised)", fontsize=11, y=1.01)
        fig2.tight_layout()
        fig2.savefig(os.path.join(OUT_DIR, f"{cls_name}_gallery.pdf"), dpi=150, bbox_inches="tight")
        plt.close(fig2)
    print("  → smooth + oscillatory galleries")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Generating LaTeX assets …")
    generate_results_table()
    generate_convergence_curves()
    generate_function_illustrations()
    print("Done.")


if __name__ == "__main__":
    main()
