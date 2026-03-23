import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------------------------------------------------------------
# Configurations matching Figure 1 of the paper.
# Legend shows the MISSPECIFIED value; the other step-size is set correctly:
#   eta_correct = 2/(H_g + L_g) = 2/20002  (default in toy.py)
#   gamma_correct = 2/(L_f + H_f) = 2/(2+0.1) ~ 0.95
# -----------------------------------------------------------------------
configs = [
    ("toy_adaptive",
     "Adaptive BiOpt",
     "black", "-"),
    ("eta0.00011gamma0.95",
     r"$\eta=1.1\times10^{-4}$",
     "saddlebrown", "--"),
    ("eta0.0002gamma0.95",
     r"$\eta=2\times10^{-4}$",
     "tab:blue", "-."),
    ("eta9.999000099990002e-05gamma1.0",
     r"$\gamma=1$",
     "tab:green", ":"),
    ("eta9.999000099990002e-05gamma0.96",
     r"$\gamma=0.96$",
     "tab:pink", (0, (3, 1, 1, 1))),
]

data = {}
for fname, *_ in configs:
    with open(fname, "rb") as f:
        data[fname] = pickle.load(f)

fig = plt.figure(figsize=(13, 5.5))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

# -----------------------------------------------------------------------
# Chart 1: f(x_t; y*(x_t)) - f(x*; y*(x*))  vs  Iterations  [log-log]
# y*(x_t) = y* = (0.1, 2) for all t  =>  f(x*; y*(x*)) = 0
# Stored values = f(x_t; y*), plotted from t=1
# -----------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0])
for fname, label, color, ls in configs:
    values = data[fname]["values"]          # values[t-1] = f(x_t; y*)
    iters  = np.arange(1, len(values) + 1)  # t = 1, 2, …
    ax1.loglog(iters, values, label=label, color=color,
               linestyle=ls, linewidth=1.8)

ax1.set_xlabel("Iterations", fontsize=12)
ax1.set_ylabel(r"$f(x_t;\,y^*(x_t)) - f(x^*;\,y^*(x^*))$", fontsize=11)
ax1.set_title("(a)", fontsize=12, loc="left")
ax1.legend(fontsize=9, framealpha=0.85)
ax1.grid(True, which="both", alpha=0.25)
ax1.set_xlim(1, 1000)

# -----------------------------------------------------------------------
# Chart 2: Trajectory {x_t}
# xs = [x_0, x_0, x_1, x_2, …]  =>  outer iterates: xs[1:]
# Only show valid iterates (before divergence cutoff)
# -----------------------------------------------------------------------
ax2 = fig.add_subplot(gs[1])

# Background contour of f(x; y*) = 0.1*x1^2 + 2*x2^2
x1g = np.linspace(-8, 8, 300)
x2g = np.linspace(-8, 8, 300)
X1, X2 = np.meshgrid(x1g, x2g)
F_grid = 0.1 * X1**2 + 2 * X2**2
ax2.contourf(X1, X2, F_grid, levels=30, cmap="YlOrRd", alpha=0.3)
ax2.contour(X1, X2, F_grid, levels=30, colors="gray", linewidths=0.4, alpha=0.5)

XLIM, YLIM = (-8, 8), (-8, 8)
for fname, label, color, ls in configs:
    d = data[fname]
    n_valid = len(d["values"])               # number of valid outer steps
    xs = np.array(d["xs"][1:n_valid + 2])   # x_0, x_1, …, x_{n_valid}
    xs_c = np.clip(xs, [XLIM[0], YLIM[0]], [XLIM[1], YLIM[1]])
    ax2.plot(xs_c[:, 0], xs_c[:, 1], color=color,
             linestyle=ls, linewidth=1.6, alpha=0.9, label=label)
    ax2.plot(xs_c[0, 0],  xs_c[0, 1],  "o", color=color, markersize=5, zorder=4)
    ax2.plot(xs_c[-1, 0], xs_c[-1, 1], "s", color=color, markersize=5, zorder=4)

ax2.plot(0, 0, "k*", markersize=13, label=r"$x^*$", zorder=5)
ax2.set_xlabel(r"$x_1$", fontsize=12)
ax2.set_ylabel(r"$x_2$", fontsize=12)
ax2.set_title("(b) Sequence $\\{x_t\\}_t$", fontsize=12, loc="left")
ax2.set_xlim(*XLIM)
ax2.set_ylim(*YLIM)
ax2.legend(fontsize=8, loc="upper right", framealpha=0.85)
ax2.grid(True, alpha=0.25)

plt.suptitle("Adaptive Bilevel Optimization — Toy Example", fontsize=13, fontweight="bold")
plt.savefig("toy_visualization.png", dpi=150, bbox_inches="tight")
print("Saved: toy_visualization.png")
plt.show()
