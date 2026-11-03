"""Figure: learning-rate sweeps (validation accuracy vs. lr), 3 datasets x 3 depths.

Run after src/analyze.py --exp main.
"""
import argparse
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
ARCHN = {"shallow": "1 hidden layer", "medium": "2 hidden layers", "deep": "3 hidden layers"}
STYLE = {  # fixed categorical order (validated palette) + marker/linestyle as secondary encoding
    "bp_sgd": ("BP+SGD", "#2a78d6", "o", "-"),
    "prox_sgd": ("ProxProp", "#eb6834", "s", "-"),
    "nm_dynamic": ("NM-dynamic", "#1baf7a", "^", "--"),
    "nm_static": ("NM-static", "#eda100", "v", ":"),
    "prox_all": ("ProxProp-all", "#e87ba4", "D", "-."),
    "bp_adam": ("BP+Adam", "#008300", "x", (0, (1, 1))),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6.8,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.edgecolor": INK2,
                     "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "mathtext.fontset": "stix"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main")
    ap.add_argument("--methods", nargs="+", default=["bp_sgd", "prox_sgd", "nm_dynamic", "nm_static", "prox_all"])
    ap.add_argument("--out", default="sweep")
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--height", type=float, default=5.0)
    args = ap.parse_args()
    S = json.load(open(os.path.join(ROOT, "results", f"summary_{args.exp}.json")))
    fig, axes = plt.subplots(3, 3, figsize=(7.16, args.height), sharex=True)
    for i, ds in enumerate(("drybean", "optdigits", "har")):
        for j, arch in enumerate(("shallow", "medium", "deep")):
            ax = axes[i, j]
            recs = {r["method"]: r for r in S if r["dataset"] == ds and r["arch"] == arch}
            if not recs:
                ax.set_visible(False)
                continue
            ref = next(iter(recs.values()))["ref"]
            ax.axhspan(ref - args.delta / 100, ref, color="#ecebe8", lw=0)
            for m in args.methods:
                if m not in recs:
                    continue
                name, col, mk, ls = STYLE[m]
                cur = recs[m]["curve"]
                sd = recs[m]["curve_std"]
                lrs = np.array(sorted(float(k) for k in cur))
                mu = np.array([cur[f"{x:g}"] for x in lrs])
                s = np.array([sd.get(f"{x:g}", 0.0) or 0.0 for x in lrs])
                ax.fill_between(lrs, mu - s, mu + s, color=col, alpha=0.12, lw=0)
                ax.plot(lrs, mu, ls=ls, color=col, lw=1.0, marker=mk, ms=2.6, mew=0.6,
                        mfc="white" if m in ("nm_static", "prox_all") else col, label=name)
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(True, which="major", color=GRID, lw=0.35)
            if i == 0:
                ax.set_title(ARCHN[arch], fontsize=7.5, color=INK)
            if j == 0:
                ax.set_ylabel(f"{DSN[ds]}\nfinal val. accuracy")
            if i == 2:
                ax.set_xlabel(r"learning rate $\eta$")
            ax.set_xticks([1e-4, 1e-2, 1, 100])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 1.015),
               handlelength=2.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.4, w_pad=0.6)
    fig.savefig(os.path.join(FIG, f"{args.out}.pdf"), bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
