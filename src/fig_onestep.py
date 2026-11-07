"""Figure: one-step sensitivity. Run after src/onestep.py and src/dump_modes.py.

(a) per-mode responses of BP and ProxProp vs. mode energy (first hidden layer, init);
(b) measured ratio rho = ||Delta_Prox|| / ||Delta_BP|| against the three bounds.
"""
import json
import os

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_style as ps
import matplotlib.pyplot as plt  # noqa: E402  (plot_style selects the backend)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
DS = ("drybean", "optdigits", "har")
MK = {"drybean": "o", "optdigits": "s", "har": "^"}
BP_C, PX_C = ps.METHOD["bp_sgd"][1], ps.METHOD["prox_sgd"][1]
BOX_EDGE, MEDIAN = ps.METHOD["prox_sgd"][1], ps.METHOD["prox_all"][1]

ps.apply(7.0)
modes = json.load(open(os.path.join(ROOT, "results", "onestep", "modes.json")))
rec = pd.DataFrame(json.load(open(os.path.join(ROOT, "results", "onestep", "records.json")))["records"])

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.05), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
for ds in DS:  # BP filled, ProxProp open; the marker shape gives the dataset
    m = modes[f"{ds}_L0"]
    lam, bp, px = np.array(m["lam"]), np.array(m["bp"]), np.array(m["prox"])
    ax.scatter(lam, bp, s=11, marker=MK[ds], color=BP_C, linewidths=0, alpha=0.85, zorder=3)
    ax.scatter(lam, px, s=11, marker=MK[ds], facecolors="white", edgecolors=PX_C, linewidths=0.75, zorder=3)
ax.axvline(1.0, color=ps.INK2, lw=0.7, ls=(0, (2, 2)), zorder=2)
ax.text(1.5, 4.5e-6, r"$\lambda_i=1$", color=ps.INK2, fontsize=6.8, style="italic")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(top=3.0)
ax.set_xlabel(r"Mode energy $\lambda_i=\sigma_i^2$ of $\widehat A\widehat A^{\top}$")
ax.set_ylabel(r"Response per unit $\delta$")
handles = [Line2D([], [], ls="none", marker="o", ms=4, color=BP_C, label=r"BP, gain $\sigma_i$"),
           Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=PX_C, mew=0.9,
                  label=r"ProxProp, gain $\sigma_i/(1+\sigma_i^2)$")]
handles += [Line2D([], [], ls="none", label=" ")]  # keeps the methods in the first legend column
handles += [Line2D([], [], ls="none", marker=MK[ds], ms=4, mfc="white", mec=ps.INK2, mew=0.8, label=ps.DSN[ds])
            for ds in DS]
ax.legend(handles=handles, loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.3, borderaxespad=0.4)

ax = axes[1]
rec["b2c"] = rec["bound2_conservative"]
groups = [(ds, l) for ds in DS for l in (0, 1, 2)]
x = np.arange(len(groups))
for i, (ds, l) in enumerate(groups):
    r = rec[(rec.dataset == ds) & (rec.layer == l)]
    q = np.percentile(r.rho, [5, 25, 50, 75, 95])
    ax.vlines(i, q[0], q[4], color=BOX_EDGE, lw=0.8, zorder=2)
    ax.add_patch(plt.Rectangle((i - 0.2, q[1]), 0.4, q[3] - q[1], facecolor=ps.BOX_FILL, edgecolor=BOX_EDGE,
                               lw=0.7, zorder=2))
    ax.hlines(q[2], i - 0.2, i + 0.2, color=MEDIAN, lw=1.5, zorder=3)
    ax.scatter([i], [r.bound1.median()], marker="x", s=15, color=ps.INK, linewidths=0.9,
               label=r"$1/(1+\mu)$ (Prop. 1)" if i == 0 else None, zorder=4)
    ax.scatter([i + 0.3], [r.b2c.median()], marker="+", s=22, color=ps.INK2, linewidths=0.9,
               label=r"$1/(1+\mu_+)$ (Cor. 1)" if i == 0 else None, zorder=4)
    ax.scatter([i], [r.bound3.median()], marker="D", s=11, facecolors="white", edgecolors=ps.INK, linewidths=0.8,
               label=r"$\|G\|_F/(2\|g\|_F)$ (Prop. 2)" if i == 0 else None, zorder=4)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([f"L{l+1}" for _, l in groups])
ax.tick_params(axis="x", which="both", length=0)
ax.grid(False, axis="x")
for j, ds in enumerate(DS):
    ax.text(3 * j + 1, 1.75, ps.DSN[ds], ha="center", va="bottom", color=PX_C, fontsize=7.2, weight="bold",
            style="italic")
ax.set_ylim(2.5e-4, 3.5)
ax.set_xlim(-0.6, len(groups) - 0.4)
for b in (2.5, 5.5):
    ax.axvline(b, color=ps.MUTED, lw=0.6, ls=(0, (3, 3)))
ax.set_ylabel(r"$\rho=\|\Delta_{\mathrm{Prox}}\|_F\,/\,\|\Delta_{\mathrm{BP}}\|_F$")
ax.set_xlabel("Hidden layer (all checkpoints, batches and architectures)")
ax.legend(loc="lower right", ncol=3, handletextpad=0.15, columnspacing=0.7, borderaxespad=0.4)
fig.tight_layout(w_pad=1.6)
ps.subcaptions(fig, axes, ["(a) Per-mode response at initialization",
                           "(b) Measured ratio (box: 25-75%, whiskers: 5-95%) and median bounds"])
fig.savefig(os.path.join(FIG, "onestep.pdf"), bbox_inches="tight")
print("saved", os.path.join(FIG, "onestep.pdf"))
