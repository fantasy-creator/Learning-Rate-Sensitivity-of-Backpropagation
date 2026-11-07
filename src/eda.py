"""Exploratory data analysis of the three datasets, on the training split only.

Usage: python src/eda.py      (after src/data_prep.py; needs numpy, pandas and matplotlib only)

Reads   data/processed/{drybean,optdigits,har}.npz and data/processed/summary.json
Writes  paper/figures/eda.pdf          the report's EDA figure (three panels, full text width)
        results/eda/*.png              supporting plots for every dataset
        results/eda/eda_summary.txt    the numbers quoted in the report

The validation and test splits are only used for their class counts, which src/data_prep.py
already records; every statistic of the features comes from the training split, so the EDA
does not look at data that later selects or scores a model.
"""
import json
import os

import numpy as np

import plot_style as ps
import matplotlib.pyplot as plt  # noqa: E402  (plot_style selects the backend)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "paper", "figures")
OUT = os.path.join(ROOT, "results", "eda")
DS = ["drybean", "optdigits", "har"]
DSN = {"drybean": "Dry Bean", "optdigits": "Optdigits", "har": "HAR"}
BATCH, N_BATCHES, SEED = 128, 10, 0

INK, INK2, GRID = ps.INK, ps.INK2, ps.GRID
# categorical class colors of the supporting plots (up to 10 classes)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7b5bd6", "#8c6d4f", "#5a9bb0", "#b04a5a"]
# HAR activities in the report figure: the three walking activities in blues, the static postures in oranges
ACTIVITY = ["#08519c", "#3a87c8", "#8fc0e3", "#8c2d04", "#e6550d", "#fdae6b"]
# datasets in order of input dimension (16, 64, 561) on the blue ramp of the other figures
LINE = {"drybean": (ps.RAMP3[0], "-"), "optdigits": (ps.RAMP3[1], "--"), "har": (ps.RAMP3[2], ":")}
ps.apply(7.0)  # same style as src/fig_*.py, so the EDA figure matches the other figures of the report


def load(name):
    z = np.load(os.path.join(DATA, f"{name}.npz"), allow_pickle=False)
    x = z["x_train"]
    return dict(raw=x, std=(x - z["mean"]) / z["std"], y=z["y_train"], mean=z["mean"], sd=z["std"])


def batch_spectra(xs, rng):
    """Eigenvalues of A_hat A_hat^T (bias row included) for random standardized training mini-batches.

    Same quantity as Table I of the report: the second-moment matrix of the first layer's input,
    whose largest eigenvalue sets BP's largest stable learning rate. Returns a (N_BATCHES, d+1)
    array, sorted in decreasing order and zero-padded beyond the batch rank.
    """
    n, d = xs.shape
    out = np.zeros((N_BATCHES, d + 1))
    for k in range(N_BATCHES):
        idx = rng.choice(n, BATCH, replace=False)
        ah = np.hstack([xs[idx], np.ones((BATCH, 1))])
        s = np.linalg.svd(ah, compute_uv=False) ** 2  # eigenvalues of ah^T ah = nonzero ones of ah ah^T
        out[k, :len(s)] = np.sort(s)[::-1]
    return out


def spectrum_stats(sp):
    lam = np.median(sp, 0)
    tol = lam[0] * 1e-10
    ranks = [(row > row[0] * 1e-10).sum() for row in sp]
    mu = [row[row > row[0] * 1e-10].min() for row in sp]
    return dict(lam=lam, lam_max=float(np.median(sp[:, 0])), rank=int(np.median(ranks)),
                mu_plus=float(np.median(mu)), cond=float(np.median(sp[:, 0] / np.array(mu))), tol=tol)


def correlation(x):
    keep = x.std(0) > 1e-8
    return np.corrcoef(x[:, keep], rowvar=False), keep


def pca2(xs):
    xc = xs - xs.mean(0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    return xc @ vt[:2].T, ev


def main():
    from matplotlib import font_manager
    if "Times New Roman" not in {f.name for f in font_manager.fontManager.ttflist}:
        print("WARNING: Times New Roman is not installed, so the figures fall back to DejaVu Serif and will not\n"
              "match the other figures of the report. Run this script where Times New Roman exists (e.g. Windows).\n")
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    info = json.load(open(os.path.join(DATA, "summary.json")))
    D = {n: load(n) for n in DS}
    rng = np.random.default_rng(SEED)
    lines, S, C, P = [], {}, {}, {}

    def say(s=""):
        lines.append(s)
        print(s)

    say("Exploratory data analysis (training split only)")
    say("=" * 48)
    for n in DS:
        d, I = D[n], info[n]
        raw_sd = d["raw"].std(0)
        nonconst = raw_sd > 1e-8
        corr, keep = correlation(d["raw"])
        iu = np.triu_indices(corr.shape[0], 1)
        absr = np.abs(corr[iu])
        sp = batch_spectra(d["std"], rng)
        S[n] = spectrum_stats(sp)
        C[n] = (corr, keep)
        P[n] = pca2(d["std"])
        counts = np.array(I["class_counts_train"])
        share = counts / counts.sum()
        ev = P[n][1]
        k90 = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
        say(f"\n{DSN[n]}: d = {I['d']}, C = {I['C']}, train/val/test = {I['n_train']}/{I['n_val']}/{I['n_test']}")
        if "n_duplicates_removed" in I:
            say(f"  duplicates removed before splitting: {I['n_duplicates_removed']}")
        if "test_rows_identical_to_train" in I:
            say(f"  official test rows identical to a training row: {I['test_rows_identical_to_train']}")
        say(f"  missing values: {int(np.isnan(d['raw']).sum())}")
        say(f"  class shares (train): min {100 * share.min():.1f}% ({I['classes'][share.argmin()]}), "
            f"max {100 * share.max():.1f}% ({I['classes'][share.argmax()]}), imbalance ratio {share.max() / share.min():.1f}")
        say(f"  raw feature std: min {raw_sd[nonconst].min():.3g}, max {raw_sd.max():.3g} "
            f"(ratio {raw_sd.max() / raw_sd[nonconst].min():.3g}); constant columns: {int((~nonconst).sum())}")
        say(f"  feature pairs with |r| > 0.9: {100 * (absr > 0.9).mean():.1f}% of {len(absr)}; "
            f"median |r| {np.median(absr):.2f}")
        say(f"  first-layer spectrum (standardized batch of {BATCH}, bias row, median of {N_BATCHES}): "
            f"rank {S[n]['rank']}/{I['d'] + 1}, lambda_max {S[n]['lam_max']:.3g}, mu_+ {S[n]['mu_plus']:.3g}, "
            f"lambda_max/mu_+ {S[n]['cond']:.3g}")
        say(f"  PCA (standardized): PC1 {100 * ev[0]:.1f}%, PC2 {100 * ev[1]:.1f}%, "
            f"{k90} components for 90% of the variance")
        if n == "drybean":
            names = I["features"]
            kept = [names[i] for i in np.flatnonzero(keep)]
            top = sorted(zip(absr, iu[0], iu[1]), reverse=True)[:5]
            say("  most correlated pairs: " + "; ".join(f"{kept[i]}-{kept[j]} {corr[i, j]:+.3f}" for _, i, j in top))

    with open(os.path.join(OUT, "eda_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    make_paper_figure(D, S, C, P, info)
    make_supporting_figures(D, S, C, P, info)
    print("\nwrote paper/figures/eda.pdf and results/eda/")


def panel_spectrum(ax, S, info, show_lmax=False):
    for n in DS:
        lam = S[n]["lam"][:S[n]["rank"]]
        extra = f", $\\lambda_{{max}}$ {S[n]['lam_max']:.2g}" if show_lmax else ""
        color, ls = LINE[n]
        ax.plot(np.arange(1, len(lam) + 1), lam, ls=ls, color=color, lw=1.4,
                label=f"{DSN[n]} (rank {S[n]['rank']}/{info[n]['d'] + 1}{extra})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue index $i$")
    ax.set_ylabel("Eigenvalue $\\lambda_i$")
    top = max(S[n]["lam"][0] for n in DS)
    ax.set_ylim(top=top * 3e3)  # headroom so the legend sits above the curves
    ax.legend(loc="upper right", handlelength=2.4, borderaxespad=0.4)
    ax.grid(True, which="major")
    ax.grid(False, which="minor")


def panel_corr(ax, corr, labels, fig, cbar=True):
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.grid(False)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if cbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[-1, 0, 1])
        cb.ax.tick_params(labelsize=5.5, length=2)
        cb.outline.set_linewidth(0.4)
        cb.set_label("Pearson $r$", fontsize=6)
    return im


def panel_pca(ax, Z, ev, y, classes, rng, n_max=2000, legend=True, colors=CAT):
    idx = rng.permutation(len(y))[:n_max]
    for c in range(len(classes)):
        m = idx[y[idx] == c]
        ax.scatter(Z[m, 0], Z[m, 1], s=2.4, color=colors[c], alpha=0.6, lw=0, label=classes[c], rasterized=True)
    ax.set_xlabel(f"PC1 ({100 * ev[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({100 * ev[1]:.0f}%)")
    if legend:
        ax.legend(markerscale=3, handletextpad=0.2, borderaxespad=0.4, labelspacing=0.25, loc="best", fontsize=5.8)


def short_activity(name):
    return name.replace("WALKING_UPSTAIRS", "upstairs").replace("WALKING_DOWNSTAIRS", "downstairs") \
        .replace("WALKING", "walking").replace("SITTING", "sitting").replace("STANDING", "standing") \
        .replace("LAYING", "lying")


def make_paper_figure(D, S, C, P, info):
    fig = plt.figure(figsize=(7.16, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1], wspace=0.45, left=0.075, right=0.99, bottom=0.3, top=0.97)
    axes = [fig.add_subplot(gs[k]) for k in range(3)]
    panel_spectrum(axes[0], S, info)
    corr, keep = C["drybean"]
    names = [info["drybean"]["features"][i] for i in np.flatnonzero(keep)]
    panel_corr(axes[1], corr, [abbrev(s) for s in names], fig)
    Z, ev = P["har"]
    classes = [short_activity(c) for c in info["har"]["classes"]]
    panel_pca(axes[2], Z, ev, D["har"]["y"], classes, np.random.default_rng(SEED), colors=ACTIVITY)
    ps.subcaptions(fig, axes, ["(a) First-layer input spectrum", "(b) Dry Bean feature correlation",
                               "(c) HAR inputs, PCA"])
    fig.savefig(os.path.join(FIG, "eda.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "eda_paper_preview.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def abbrev(name):
    table = {"MajorAxisLength": "MajorAxis", "MinorAxisLength": "MinorAxis", "AspectRation": "AspectRatio",
             "EquivDiameter": "EquivDiam", "Compactness": "Compact.", "roundness": "Roundness"}
    return table.get(name, name).replace("ShapeFactor", "SF")


def make_supporting_figures(D, S, C, P, info):
    rng = np.random.default_rng(SEED)
    # class distribution per split
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6))
    for ax, n in zip(axes, DS):
        I = info[n]
        k = len(I["classes"])
        labels = [short_activity(str(c)) for c in I["classes"]]
        w = 0.27
        for j, (split, col) in enumerate([("train", INK), ("val", INK2), ("test", "#a9a7a1")]):
            cnt = np.array(I[f"class_counts_{split}"])
            ax.bar(np.arange(k) + (j - 1) * w, 100 * cnt / cnt.sum(), w * 0.92, color=col, label=split)
        ax.set_xticks(range(k))
        ax.set_xticklabels(labels, rotation=45 if n != "optdigits" else 0, ha="right" if n != "optdigits" else "center", fontsize=5.5)
        ax.set_title(DSN[n], fontsize=7.5, color=INK)
        ax.set_ylabel("share of split (%)")
        ax.grid(True, axis="y", color=GRID, lw=0.35)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "class_distribution.png"), dpi=200)
    plt.close(fig)

    # raw feature scale: per-feature standard deviation before z-scoring
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    for n in DS:
        sd = np.sort(D[n]["raw"].std(0))[::-1]
        sd = sd[sd > 1e-8]
        color, ls = LINE[n]
        ax.plot(np.linspace(0, 1, len(sd)), sd, ls=ls, color=color, lw=1.2, label=DSN[n])
    ax.set_yscale("log")
    ax.set_xlabel("features, sorted by standard deviation (fraction of d)")
    ax.set_ylabel("raw standard deviation")
    ax.legend(frameon=False)
    ax.grid(True, color=GRID, lw=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "feature_scale.png"), dpi=200)
    plt.close(fig)

    # spectrum with lambda_max annotated
    fig, ax = plt.subplots(figsize=(4.5, 3))
    panel_spectrum(ax, S, info, show_lmax=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "spectrum.png"), dpi=200)
    plt.close(fig)

    # correlation matrices
    for n in DS:
        corr, keep = C[n]
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        if n == "drybean":
            names = [info[n]["features"][i] for i in np.flatnonzero(keep)]
            panel_corr(ax, corr, [abbrev(s) for s in names], fig)
        else:
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[-1, 0, 1]).set_label("Pearson $r$")
            ax.set_xlabel("feature")
            ax.set_ylabel("feature")
        ax.set_title(f"{DSN[n]}: feature correlation (training split, {int(keep.sum())} non-constant features)", fontsize=6.5, color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"correlation_{n}.png"), dpi=200)
        plt.close(fig)

    # PCA for every dataset
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, n in zip(axes, DS):
        Z, ev = P[n]
        classes = [short_activity(str(c)) for c in info[n]["classes"]]
        panel_pca(ax, Z, ev, D[n]["y"], classes, rng)
        ax.set_title(DSN[n], fontsize=7.5, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pca.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
