"""
Visualize RL experiment results.

Loads all .pkl files from results/kl/ and results/euclidean/,
then plots the suboptimality gap  E_{p_in}[v* - v^pi_k]  vs outer iterations.

Usage (from project root):
    python rl-experiment/visualize_rl.py
"""
import os
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_here = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------
# Config: colour / linestyle per mode  (shared across KL and Euclidean)
# -----------------------------------------------------------------------
MODE_STYLE = {
    "adaptive": dict(color="tab:blue",   linestyle="-",  linewidth=2.0, label="Adaptive BiOpt"),
    "theory":   dict(color="tab:orange", linestyle="--", linewidth=1.8, label="Theory step-sizes"),
    "fixed":    dict(color="tab:red",    linestyle=":",  linewidth=1.8, label="Fixed step-sizes"),
}


def load_results(subdir: str) -> dict:
    """Return {mode: values_array} for all .pkl files under subdir."""
    pattern = os.path.join(_here, "results", subdir, "*.pkl")
    results = {}
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        # fname like  kl_adaptive_S10_G0.pkl  or  euclidean_fixed_S10_G0.pkl
        parts = fname.replace(".pkl", "").split("_")
        # parts[0] = method (kl / euclidean), parts[1] = mode, rest = params
        mode = parts[1]  # adaptive / theory / fixed
        with open(path, "rb") as f:
            d = pickle.load(f)
        values = np.array(d["values"])
        if mode not in results:
            results[mode] = values
        else:
            # keep the longer run if re-run with different --max-iter
            if len(values) > len(results[mode]):
                results[mode] = values
    return results


def plot_gap(ax, results: dict, title: str):
    """Plot suboptimality gap vs outer iterations on ax."""
    for mode, style in MODE_STYLE.items():
        if mode not in results:
            continue
        values = results[mode]
        iters  = np.arange(len(values))
        # clip negative / zero for log scale
        pos = np.where(values > 0, values, np.nan)
        ax.semilogy(iters, pos, **style)

    ax.set_xlabel("Outer iteration $k$", fontsize=11)
    ax.set_ylabel(r"$\mathbb{E}_{p_0}[v^* - v^{\pi_k}]$", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, which="both", alpha=0.25)


def main():
    kl_results  = load_results("kl")
    euc_results = load_results("euclidean")

    if not kl_results and not euc_results:
        print("No result files found. Run the experiments first:\n"
              "  bash rl-experiment/run_rl.sh")
        return

    n_plots = (1 if kl_results else 0) + (1 if euc_results else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)
    axes = axes[0]

    idx = 0
    if kl_results:
        plot_gap(axes[idx], kl_results,
                 "KL / Mirror-descent\nSuboptimality gap vs iterations")
        idx += 1
    if euc_results:
        plot_gap(axes[idx], euc_results,
                 "Euclidean / Projection\nSuboptimality gap vs iterations")
        idx += 1

    plt.suptitle("Adaptive Bilevel Optimization — RL Experiment",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(_here, "rl_visualization.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
