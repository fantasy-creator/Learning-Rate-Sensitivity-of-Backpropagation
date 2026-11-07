"""Figure: ProxProp's per-layer step-length ratio s_l(t) = ||g M^-1|| / ||g|| over training.

Deep MLP, ProxProp at its validation-selected learning rate, mean over 5 seeds (band: min-max).
Run after src/analyze.py --exp main.
"""
import glob
import json
import os

import numpy as np

import plot_style as ps
import matplotlib.pyplot as plt  # noqa: E402  (plot_style selects the backend)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
LST = ["-", "--", ":"]

ps.apply(6.5)
S = json.load(open(os.path.join(ROOT, "results", "summary_main.json")))
fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.56), sharey=True)
for ax, ds in zip(axes, ("drybean", "optdigits", "har")):
    best = next(r["best_lr"] for r in S if r["dataset"] == ds and r["arch"] == "deep" and r["method"] == "prox_sgd")
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results", "main", f"{ds}_deep_prox_sgd_lr{best:g}_s*.json"))]
    runs = [r for r in runs if not r["diverged"]]
    for l in range(3):
        D = np.array([[e[l] for e in r["history"]["damp"]] for r in runs])
        ep = np.arange(1, D.shape[1] + 1)
        ax.fill_between(ep, D.min(0), D.max(0), color=ps.RAMP3[l], alpha=0.25, lw=0)
        ax.plot(ep, D.mean(0), color=ps.RAMP3[l], ls=LST[l], lw=1.1, label=f"Layer {l + 1}")
    ax.set_yscale("log")
    ax.set_title(f"{ps.DSN[ds]} ($\\eta^\\star={best:g}$)", fontsize=6.5)
    ax.set_xlabel("Epoch")
    ax.set_xticks([0, 50, 100])
    ax.set_xlim(0, 100)
axes[0].set_ylabel(r"$s_l(t)=\|gM^{-1}\|_F/\|g\|_F$", fontsize=6.5)
axes[0].legend(loc="lower right", handlelength=1.8, borderpad=0.3, labelspacing=0.2, fontsize=5.6)
fig.tight_layout(w_pad=0.3)
fig.savefig(os.path.join(FIG, "damping.pdf"), bbox_inches="tight")
print("saved damping")
