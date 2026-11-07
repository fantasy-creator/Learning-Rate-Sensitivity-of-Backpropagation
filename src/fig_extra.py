"""Figure: input-scale stress test.

Plots the usable window [lo, hi] (Delta = 2 pp w.r.t. the best accuracy among BP, ProxProp and
ProxProp-all at that scale) and the best learning rate for each method and scale.
Run after src/analyze.py --exp main and --exp scale.
"""
import json
import os

import numpy as np

import plot_style as ps
import matplotlib.pyplot as plt  # noqa: E402  (plot_style selects the backend)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
METHODS = ("bp_sgd", "prox_sgd", "prox_all")


def load(exp):
    p = os.path.join(ROOT, "results", f"summary_{exp}.json")
    return json.load(open(p)) if os.path.exists(p) else []


def scale_figure():
    ps.apply(7.0)
    S = load("scale") + [r for r in load("main") if r["arch"] == "medium" and r["method"] in METHODS]
    # the main sweep's medium MLP is the s = 1 point; its window reference includes the NM methods,
    # so recompute the s = 1 reference from bp/prox/prox_all only for comparability
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.95), sharey=True)
    for ax, ds in zip(axes, ("drybean", "optdigits", "har")):
        for s in (0.3, 3.0):  # separate the three input scales
            ax.axvline(s, color=ps.MUTED, lw=0.6, ls=(0, (3, 3)))
        for k, m in enumerate(METHODS):
            name, col, mk, _ = ps.METHOD[m]
            xs, lo, hi, best = [], [], [], []
            for s in (0.1, 1.0, 10.0):
                rs = [r for r in S if r["dataset"] == ds and r["scale"] == s]
                if not rs:
                    continue
                ref = max(max(r["curve"].values()) for r in rs if r["method"] in METHODS)
                r = next((r for r in rs if r["method"] == m), None)
                if r is None:
                    continue
                ok = [float(l) for l, a in r["curve"].items() if a >= ref - 0.02]
                xs.append(s * (1.3 ** (k - 1)))
                lo.append(min(ok) if ok else np.nan)
                hi.append(max(ok) if ok else np.nan)
                best.append(r["best_lr"])
            xs, lo, hi = np.array(xs), np.array(lo), np.array(hi)
            ax.vlines(xs, lo, hi, color=col, lw=3.2, alpha=0.9, zorder=2)
            cens = hi >= 100.0  # window reaches the top of the grid: true upper edge unknown
            ax.scatter(xs[cens], hi[cens] * 1.7, marker="^", s=13, color=col, linewidths=0, zorder=3)
            ax.scatter(xs, best, marker=mk, s=17, color=col, edgecolors="white", linewidths=0.6,
                       zorder=4, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.1, 1, 10])
        ax.set_xticklabels(["0.1", "1", "10"])
        ax.minorticks_off()
        ax.grid(False, axis="x")
        ax.set_xlim(0.06, 17)
        ax.set_ylim(5e-5, 400)
        ax.set_title(ps.DSN[ds])
        ax.set_xlabel("Input scale $s$")
    axes[0].set_ylabel(r"Learning rate $\eta$")
    axes[0].legend(loc="lower left", handletextpad=0.2, borderaxespad=0.4)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIG, "scale.pdf"), bbox_inches="tight")
    print("saved scale")


if __name__ == "__main__":
    if load("scale"):
        scale_figure()
