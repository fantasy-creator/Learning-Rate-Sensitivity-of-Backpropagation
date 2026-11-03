"""Figure: one-step sensitivity. Run after src/onestep.py and src/dump_modes.py.

(a) per-mode responses of BP and ProxProp vs. mode energy (first hidden layer, init);
(b) measured ratio rho = ||Delta_Prox|| / ||Delta_BP|| against the three bounds.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
COL = {"drybean": "#2a78d6", "optdigits": "#eb6834", "har": "#1baf7a"}
DSN = {"drybean": "Dry Bean", "optdigits": "Optdigits", "har": "HAR"}
MK = {"drybean": "o", "optdigits": "s", "har": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": INK2,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "mathtext.fontset": "stix"})

modes = json.load(open(os.path.join(ROOT, "results", "onestep", "modes.json")))
rec = pd.DataFrame(json.load(open(os.path.join(ROOT, "results", "onestep", "records.json")))["records"])

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.25), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
for ds in ("drybean", "optdigits", "har"):
    m = modes[f"{ds}_L0"]
    lam, bp, px = np.array(m["lam"]), np.array(m["bp"]), np.array(m["prox"])
    ax.scatter(lam, bp, s=10, marker=MK[ds], color=COL[ds], linewidths=0, alpha=0.9,
               label=f"{DSN[ds]}, BP")
    ax.scatter(lam, px, s=10, marker=MK[ds], facecolors="white", edgecolors=COL[ds], linewidths=0.7,
               label=f"{DSN[ds]}, ProxProp")
ax.axvline(1.0, color=INK2, lw=0.6, ls=":")
ax.text(1.15, 3e-6, r"$\lambda_i=1$", color=INK2, fontsize=6.5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"mode energy $\lambda_i=\sigma_i^2$ of $\widehat A\widehat A^{\top}$")
ax.set_ylabel(r"response per unit $\delta$ along mode $i$")
ax.set_title("(a) BP: $\\sigma_i\\|Gu_i\\|$ (filled); ProxProp: $\\frac{\\sigma_i}{1+\\sigma_i^2}\\|Gu_i\\|$ (open)",
             fontsize=7, color=INK)
ax.grid(True, which="major", color=GRID, lw=0.4)
ax.legend(ncol=2, loc="upper left", frameon=False, handletextpad=0.2, columnspacing=0.6, borderaxespad=0.2)

ax = axes[1]
rec["b2c"] = rec["bound2_conservative"]
groups = [(ds, l) for ds in ("drybean", "optdigits", "har") for l in (0, 1, 2)]
x = np.arange(len(groups))
for i, (ds, l) in enumerate(groups):
    r = rec[(rec.dataset == ds) & (rec.layer == l)]
    q = np.percentile(r.rho, [5, 25, 50, 75, 95])
    ax.vlines(i, q[0], q[4], color=COL[ds], lw=0.8)
    ax.add_patch(plt.Rectangle((i - 0.18, q[1]), 0.36, q[3] - q[1], facecolor=COL[ds], edgecolor=COL[ds],
                               alpha=0.35, lw=0.6))
    ax.hlines(q[2], i - 0.18, i + 0.18, color=COL[ds], lw=1.4)
    ax.scatter([i], [r.bound1.median()], marker="x", s=14, color=INK, linewidths=0.8,
               label=r"$1/(1+\mu)$ (Prop. 1)" if i == 0 else None, zorder=3)
    ax.scatter([i + 0.28], [r.b2c.median()], marker="+", s=20, color=INK2, linewidths=0.8,
               label=r"$1/(1+\mu_+)$ (Cor. 1)" if i == 0 else None, zorder=3)
    ax.scatter([i], [r.bound3.median()], marker="D", s=9, facecolors="white", edgecolors=INK, linewidths=0.7,
               label=r"$\|G\|_F/(2\|g\|_F)$ (Prop. 2)" if i == 0 else None, zorder=3)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([f"L{l+1}" for _, l in groups])
for j, ds in enumerate(("drybean", "optdigits", "har")):
    ax.text(3 * j + 1, 1.9, DSN[ds], ha="center", va="bottom", color=INK, fontsize=7)
ax.set_ylim(8e-4, 3.5)
ax.set_xlim(-0.6, len(groups) - 0.4)
for b in (2.5, 5.5):
    ax.axvline(b, color=GRID, lw=0.6)
ax.set_ylabel(r"$\rho=\|\Delta_{\mathrm{Prox}}\|_F/\|\Delta_{\mathrm{BP}}\|_F$")
ax.set_xlabel("hidden layer (all checkpoints, batches, architectures)")
ax.set_title(r"(b) measured $\rho$ (box: 25-75%, whiskers: 5-95%) and median bounds", fontsize=7, color=INK)
ax.grid(True, axis="y", which="major", color=GRID, lw=0.4)
ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor=GRID, ncol=3, handletextpad=0.1,
          columnspacing=0.6, borderaxespad=0.3)
fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(FIG, "onestep.pdf"), bbox_inches="tight")
print("saved", os.path.join(FIG, "onestep.pdf"))
