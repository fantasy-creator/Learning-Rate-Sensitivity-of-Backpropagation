"""README figure: usable learning rates of BP+SGD, ProxProp and ProxProp-all in all nine settings.

A learning rate is usable when its seed-mean final validation accuracy is within 2 pp of the best
SGD-family accuracy of that setting, the window definition of the paper. Bars join usable rates
that are neighbours on the grid, rings mark the best rate, crosses mark rates where at least one
of the 5 seeds failed numerically. Run after src/analyze.py --exp main.
Output: assets/lr_windows.png
"""
import json
import os

from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

import plot_style as ps
import matplotlib.pyplot as plt  # noqa: E402  (plot_style selects the backend)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets")
os.makedirs(OUT, exist_ok=True)
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
DELTA = 2.0  # pp
DSN = {"drybean": "Dry Bean", "optdigits": "Optdigits", "har": "HAR"}
ARCHN = {"shallow": "1 hidden\nlayer", "medium": "2 hidden\nlayers", "deep": "3 hidden\nlayers"}
# colors, markers and fonts of the report figures (src/plot_style.py)
METHODS = [(m, ps.METHOD[m][0], ps.METHOD[m][1], ps.METHOD[m][2]) for m in ("bp_sgd", "prox_sgd", "prox_all")]
INK, INK2, MUTED, GRID = ps.INK, ps.INK2, ps.MUTED, ps.GRID


def runs_of(flags):
    """Index ranges [i, j] of consecutive True entries."""
    out, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def lane(ax, y, rec, color, marker):
    acc = [rec["curve"][f"{lr:g}"] for lr in LRS]
    ok = [a >= rec["ref"] - DELTA / 100 for a in acc]
    for i, j in runs_of(ok):
        if i == j:
            ax.plot(LRS[i], y, "o", ms=6, color=color, zorder=3)
        else:
            ax.plot(LRS[i:j + 1], [y] * (j - i + 1), color=color, lw=6, solid_capstyle="round", zorder=3)
    fails = [lr for lr in LRS if rec["nonfinite_by_lr"].get(f"{lr:g}", 0) > 0]
    ax.plot(fails, [y] * len(fails), "x", ms=6, mew=1.3, color=MUTED, zorder=2)
    ax.plot(rec["best_lr"], y, marker, ms=8.5, mfc="white", mec=color, mew=2, zorder=4)
    if ok[-1]:  # usable up to the largest rate tested: the true upper edge is unknown
        ax.plot(LRS[-1] * 1.9, y, ">", ms=6, color=color, zorder=3)


def main():
    ps.apply(10.5)
    S = json.load(open(os.path.join(ROOT, "results", "summary_main.json")))
    by = {(r["dataset"], r["arch"], r["method"]): r for r in S if r.get("scale", 1.0) == 1.0}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.5), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.745, bottom=0.12, wspace=0.06)
    centers = []
    for ax, ds in zip(axes, DSN):
        for g, arch in enumerate(ARCHN):
            for k, (m, name, color, marker) in enumerate(METHODS):
                lane(ax, 4 * g + k, by[(ds, arch, m)], color, marker)
            centers.append(4 * g + 1)
        ax.set_xscale("log")
        ax.set_xlim(LRS[0] / 2.2, LRS[-1] * 3.2)
        ax.set_xticks([1e-4, 1e-2, 1, 100])
        ax.set_xticklabels(["$10^{-4}$", "$10^{-2}$", "$1$", "$10^{2}$"])
        ax.set_xticks(LRS, minor=True)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)
        ax.tick_params(axis="y", length=0)
        ax.grid(False, axis="y")
        ax.grid(False, axis="x", which="major")
        ax.grid(True, axis="x", which="minor")
        ax.set_title(DSN[ds], fontsize=12, pad=6)
        ax.set_xlabel(r"Learning rate $\eta$")
    axes[0].set_yticks(centers[:3])
    axes[0].set_yticklabels(list(ARCHN.values()), color=INK)
    axes[0].set_ylim(10.8, -0.8)
    for k, (_, name, _, _) in enumerate(METHODS):  # direct labels on the first group
        axes[0].text(LRS[0] / 1.7, k, name, va="center", ha="left", fontsize=8.5, color=INK2)

    fig.text(0.012, 0.975, "Usable learning rates in all nine settings", fontsize=13, weight="bold",
             color=INK, va="top")
    fig.text(0.012, 0.915, f"Final validation accuracy (mean of 5 seeds) within {DELTA:g} pp of the best "
             "SGD-family run of that setting", fontsize=9.5, color=INK2, va="top")
    handles = [Line2D([], [], color=c, lw=6, marker=mk, ms=8.5, mfc="white", mec=c, mew=2, label=n)
               for _, n, c, mk in METHODS]
    handles += [Line2D([], [], ls="none", marker="o", ms=8, mfc="white", mec=INK2, mew=1.6,
                       label="best learning rate"),
                Line2D([], [], ls="none", marker="x", ms=6, mew=1.3, color=MUTED,
                       label="numerical failure in a seed"),
                Line2D([], [], ls="none", marker=">", ms=6, color=INK2, label="usable up to the grid edge")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, 0.875), ncol=6, frameon=False,
               fontsize=8.6, handlelength=2.0, columnspacing=1.1, handletextpad=0.6)
    fig.savefig(os.path.join(OUT, "lr_windows.png"), dpi=200, facecolor="white")
    print("saved", os.path.join(OUT, "lr_windows.png"))


if __name__ == "__main__":
    main()
