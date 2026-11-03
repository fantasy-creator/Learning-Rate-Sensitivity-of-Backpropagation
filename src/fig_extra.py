"""Figure: input-scale stress test.

Plots the usable window [lo, hi] (Delta = 2 pp w.r.t. the best accuracy among BP, ProxProp and
ProxProp-all at that scale) and the best learning rate for each method and scale.
Run after src/analyze.py --exp main and --exp scale.
"""
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
COL = {"bp_sgd": "#2a78d6", "prox_sgd": "#eb6834", "prox_all": "#e87ba4"}
NAME = {"bp_sgd": "BP+SGD", "prox_sgd": "ProxProp", "prox_all": "ProxProp-all"}
MK = {"bp_sgd": "o", "prox_sgd": "s", "prox_all": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "axes.edgecolor": INK2,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "mathtext.fontset": "stix"})


def load(exp):
    p = os.path.join(ROOT, "results", f"summary_{exp}.json")
    return json.load(open(p)) if os.path.exists(p) else []


def scale_figure():
    S = load("scale") + [r for r in load("main") if r["arch"] == "medium" and r["method"] in COL]
    # the main sweep's medium MLP is the s = 1 point; its window reference includes the NM methods,
    # so recompute the s = 1 reference from bp/prox/prox_all only for comparability
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.95), sharey=True)
    for ax, ds in zip(axes, ("drybean", "optdigits", "har")):
        for k, m in enumerate(("bp_sgd", "prox_sgd", "prox_all")):
            xs, lo, hi, best = [], [], [], []
            for s in (0.1, 1.0, 10.0):
                rs = [r for r in S if r["dataset"] == ds and r["scale"] == s]
                if not rs:
                    continue
                ref = max(max(r["curve"].values()) for r in rs if r["method"] in COL)
                r = next((r for r in rs if r["method"] == m), None)
                if r is None:
                    continue
                ok = [float(l) for l, a in r["curve"].items() if a >= ref - 0.02]
                xs.append(s * (1.25 ** (k - 1)))
                lo.append(min(ok) if ok else np.nan)
                hi.append(max(ok) if ok else np.nan)
                best.append(r["best_lr"])
            xs, lo, hi = np.array(xs), np.array(lo), np.array(hi)
            ax.vlines(xs, lo, hi, color=COL[m], lw=2.2, alpha=0.8)
            cens = hi >= 100.0  # window reaches the top of the grid: true upper edge unknown
            ax.scatter(xs[cens], hi[cens] * 1.6, marker="^", s=12, color=COL[m], linewidths=0, zorder=3)
            ax.scatter(xs, best, marker=MK[m], s=14, color=COL[m], edgecolors="white", linewidths=0.5,
                       zorder=3, label=NAME[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.1, 1, 10])
        ax.set_xticklabels(["0.1", "1", "10"])
        ax.set_ylim(5e-5, 400)
        ax.set_title(DSN[ds], fontsize=7.5, color=INK)
        ax.set_xlabel("input scale $s$")
        ax.grid(True, which="major", color=GRID, lw=0.35)
    axes[0].set_ylabel(r"usable $\eta$ (bar) and best $\eta$ (marker)")
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor=GRID, handletextpad=0.2)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIG, "scale.pdf"), bbox_inches="tight")
    print("saved scale")


if __name__ == "__main__":
    if load("scale"):
        scale_figure()
