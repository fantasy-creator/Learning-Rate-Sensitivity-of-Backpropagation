"""Figure: ProxProp's per-layer step-length ratio s_l(t) = ||g M^-1|| / ||g|| over training.

Deep MLP, ProxProp at its validation-selected learning rate, mean over 5 seeds (band: min-max).
Run after src/analyze.py --exp main.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
DSN = {"drybean": "Dry Bean", "optdigits": "Optdigits", "har": "HAR"}
LCOL = ["#2a78d6", "#eb6834", "#1baf7a"]
LST = ["-", "--", ":"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 6.5, "axes.labelsize": 6.5, "legend.fontsize": 5.8,
                     "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "axes.edgecolor": INK2,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "mathtext.fontset": "stix"})

S = json.load(open(os.path.join(ROOT, "results", "summary_main.json")))
fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.55), sharey=True)
for ax, ds in zip(axes, ("drybean", "optdigits", "har")):
    best = next(r["best_lr"] for r in S if r["dataset"] == ds and r["arch"] == "deep" and r["method"] == "prox_sgd")
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results", "main", f"{ds}_deep_prox_sgd_lr{best:g}_s*.json"))]
    runs = [r for r in runs if not r["diverged"]]
    for l in range(3):
        D = np.array([[e[l] for e in r["history"]["damp"]] for r in runs])
        ep = np.arange(1, D.shape[1] + 1)
        ax.fill_between(ep, D.min(0), D.max(0), color=LCOL[l], alpha=0.15, lw=0)
        ax.plot(ep, D.mean(0), color=LCOL[l], ls=LST[l], lw=1.0, label=f"layer {l + 1}")
    ax.set_yscale("log")
    ax.set_title(f"{DSN[ds]} ($\\eta^\\star={best:g}$)", fontsize=6.5, color=INK)
    ax.set_xlabel("epoch")
    ax.set_xticks([0, 50, 100])
    ax.grid(True, which="major", color=GRID, lw=0.35)
axes[0].set_ylabel(r"$s_l(t)=\|gM^{-1}\|_F/\|g\|_F$")
axes[0].legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor=GRID, handlelength=1.6, borderpad=0.3, labelspacing=0.2)
fig.tight_layout(w_pad=0.3)
fig.savefig(os.path.join(FIG, "damping.pdf"), bbox_inches="tight")
print("saved damping")
