"""Generate the dataset and update-rule tables: paper/tables/{data,methods}.tex.

Usage: python src/make_tables.py
"""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "paper", "tables")
os.makedirs(TAB, exist_ok=True)
DS = ["drybean", "optdigits", "har"]
DSN = {"drybean": "Dry Bean", "optdigits": "Optdigits", "har": "HAR"}


def sci(x, digits=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if e in (-1, 0, 1, 2):
        return f"{x:.{max(0, digits - e)}f}" if e < 1 else f"{x:.0f}"
    return f"{m:.{digits}f}\\mathrm{{e}}{{{e}}}"


def lr_tex(x):
    if x is None:
        return "--"
    e = math.log10(x)
    if abs(e - round(e)) < 1e-9:
        return f"$10^{{{int(round(e))}}}$"
    return f"$3{{\\cdot}}10^{{{int(math.floor(e))}}}$"


def data_table():
    s = json.load(open(os.path.join(ROOT, "data", "processed", "summary.json")))
    rec = json.load(open(os.path.join(ROOT, "results", "onestep", "records.json")))["records"]
    rows = []
    for ds in DS:
        info = s[ds]
        n = info["n_train"] + info["n_val"] + info["n_test"]
        cc = np.array(info["class_counts_train"]) / info["n_train"]
        # first-layer input spectrum at initialisation (independent of the network: it is the data batch)
        r0 = [r for r in rec if r["dataset"] == ds and r["layer"] == 0 and r["epoch"] == 0
              and r["arch"] == "shallow" and r["traj"] == "bp_sgd"]
        lam_max = np.median([r["lam_max"] for r in r0])
        mup = np.median([r["mu_plus"] for r in r0])
        rank = int(np.median([r["rank"] for r in r0]))
        dim = r0[0]["dim"]
        rows.append(f"{DSN[ds]} & {info['d']} & {info['C']} & {info['n_train']:,}/{info['n_val']:,}/{info['n_test']:,} & "
                    f"{rank}/{dim} & ${sci(lam_max)}$ & ${sci(mup)}$ \\\\")
    body = "\n".join(rows).replace(",", "{,}")
    tex = r"""\begin{table}[t]
\centering
\caption{Datasets and the spectrum of the first-layer input. Rank, $\lambda_{\max}$ and the
smallest positive eigenvalue $\mu_+$ of $\widehat A\widehat A^{\top}$ for a standardized mini-batch
($B=128$, bias row included; median over 10 batches).}
\label{tab:data}
\setlength{\tabcolsep}{3pt}
\small
\begin{tabular}{lrrcrrr}
\toprule
Dataset & $d$ & $C$ & train/val/test & rank & $\lambda_{\max}$ & $\mu_+$ \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(TAB, "data.tex"), "w").write(tex)


def methods_table():
    tex = r"""\begin{table}[t]
\centering
\caption{Update rules compared. $g$: BP gradient; $M=I+\widehat A\widehat A^{\top}$; $c_l$, $s_l(t)$:
ProxProp/BP step-norm ratios frozen at initialization (static) or recomputed every step (dynamic).}
\label{tab:methods}
\setlength{\tabcolsep}{3pt}
\small
\begin{tabular}{lll}
\toprule
Method & hidden layers & output layer \\
\midrule
BP+SGD & $-\eta g$ & $-\eta g$ \\
ProxProp & $-\eta gM^{-1}$ & $-\eta g$ \\
ProxProp-all & $-\eta gM^{-1}$ & $-\eta gM^{-1}$ \\
NM-static & $-\eta c_l\, g$ & $-\eta g$ \\
NM-dynamic & $-\eta s_l(t)\, g$ & $-\eta g$ \\
BP+Adam & Adam$(g)$ & Adam$(g)$ \\
\bottomrule
\end{tabular}
\end{table}
"""
    open(os.path.join(TAB, "methods.tex"), "w").write(tex)


if __name__ == "__main__":
    methods_table()
    if os.path.exists(os.path.join(ROOT, "results", "onestep", "records.json")):
        data_table()
    print("tables written to", TAB)
