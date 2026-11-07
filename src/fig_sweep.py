"""Figure: learning-rate sweeps (validation accuracy vs. lr), 3 datasets x 3 depths.

Run after src/analyze.py --exp main.
"""
import argparse
import json
import os

import numpy as np

import plot_style as ps
import matplotlib.pyplot as plt  # noqa: E402  (plot_style selects the backend)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
ARCHN = {"shallow": "1 hidden layer", "medium": "2 hidden layers", "deep": "3 hidden layers"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main")
    ap.add_argument("--methods", nargs="+", default=["bp_sgd", "nm_static", "nm_dynamic", "prox_sgd", "prox_all"])
    ap.add_argument("--out", default="sweep")
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--height", type=float, default=5.05)
    args = ap.parse_args()
    ps.apply(7.0)
    S = json.load(open(os.path.join(ROOT, "results", f"summary_{args.exp}.json")))
    fig, axes = plt.subplots(3, 3, figsize=(7.16, args.height), sharex=True, sharey=True)
    for i, ds in enumerate(("drybean", "optdigits", "har")):
        for j, arch in enumerate(("shallow", "medium", "deep")):
            ax = axes[i, j]
            recs = {r["method"]: r for r in S if r["dataset"] == ds and r["arch"] == arch}
            if not recs:
                ax.set_visible(False)
                continue
            ref = next(iter(recs.values()))["ref"]
            ax.axhspan(ref - args.delta / 100, ref, color=ps.BAND, lw=0, zorder=0)
            for m in args.methods:
                if m not in recs:
                    continue
                name, col, mk, ls = ps.METHOD[m]
                cur = recs[m]["curve"]
                sd = recs[m]["curve_std"]
                lrs = np.array(sorted(float(k) for k in cur))
                mu = np.array([cur[f"{x:g}"] for x in lrs])
                s = np.array([sd.get(f"{x:g}", 0.0) or 0.0 for x in lrs])
                hero = m == "prox_all"
                ax.fill_between(lrs, mu - s, mu + s, color=col, alpha=0.16, lw=0, zorder=1)
                ax.plot(lrs, mu, ls=ls, color=col, lw=1.45 if hero else 1.0, marker=mk, ms=3.1 if hero else 2.7,
                        mec="white", mew=0.35, label=name, zorder=4 if hero else 3)
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.02)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticks([1e-4, 1e-2, 1, 100])
            ax.set_xlim(6e-5, 1.7e2)
            if i == 0:
                ax.set_title(ARCHN[arch])
            if j == 0:
                ax.set_ylabel("Val. accuracy")
            if i == 2:
                ax.set_xlabel(r"Learning rate $\eta$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.53, 1.0), handlelength=2.4,
               columnspacing=1.4, borderpad=0.4)
    fig.tight_layout(rect=(0.035, 0, 1, 0.95), h_pad=0.5, w_pad=0.5)
    for i, ds in enumerate(("drybean", "optdigits", "har")):  # row headers
        pos = axes[i, 0].get_position()
        fig.text(0.008, (pos.y0 + pos.y1) / 2, ps.DSN[ds], rotation=90, ha="left", va="center",
                 fontsize=8, weight="bold", style="italic", color=ps.METHOD["prox_sgd"][1])
    fig.savefig(os.path.join(FIG, f"{args.out}.pdf"), bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
