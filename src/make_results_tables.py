"""LaTeX tables for the results section (run after analyze.py, final_test.py and logreg.py).

Writes paper/tables/sweep.tex (main window table), paper/tables/test.tex (test accuracy at the
validation-selected learning rate) and paper/tables/ablation.tex (momentum and tau_theta).
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "paper", "tables")
os.makedirs(TAB, exist_ok=True)
DS = ["drybean", "optdigits", "har"]
DSN = {"drybean": "Dry Bean", "optdigits": "Optdigits", "har": "HAR"}
ARCHS = ["shallow", "medium", "deep"]
ARCHN = {"shallow": "1", "medium": "2", "deep": "3"}
METHODS = ["bp_sgd", "prox_sgd", "prox_all", "nm_dynamic", "nm_static", "bp_adam"]
MN = {"bp_sgd": "BP+SGD", "prox_sgd": "ProxProp", "prox_all": "ProxProp-all", "nm_dynamic": "NM-dyn.",
      "nm_static": "NM-stat.", "bp_adam": "BP+Adam"}


def lr_short(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    e = math.floor(math.log10(x) + 1e-9)
    m = round(x / 10 ** e)
    if e >= 0:
        return f"{int(x)}" if x >= 1 else f"{x:g}"
    return f"{m}e{e}" if m != 1 else f"1e{e}"


def load(name):
    p = os.path.join(ROOT, "results", name)
    return json.load(open(p)) if os.path.exists(p) else None


def sweep_table(S):
    by = {(r["dataset"], r["arch"], r["method"]): r for r in S if r.get("scale", 1.0) == 1.0}
    lines = []
    for ds in DS:
        for i, arch in enumerate(ARCHS):
            cells = []
            for m in METHODS:
                r = by.get((ds, arch, m))
                if r is None:
                    cells.append("--")
                    continue
                w = r["win2_decades"] if r["win2_count"] else None
                wtxt = "--" if w is None else f"{w:.1f}"
                if r["win2_count"] and r["win2_censored"]:
                    wtxt += "$^\\dagger$"
                if r["win2_count"] and not r["win2_contig"]:
                    wtxt += "$^\\ddagger$"
                cells.append(f"{100 * r['peak_val_acc']:.1f} & {lr_short(r['best_lr'])} & {wtxt}")
            lead = f"\\multirow{{3}}{{*}}{{{DSN[ds]}}}" if i == 0 else ""
            lines.append(f"{lead} & {ARCHN[arch]} & " + " & ".join(cells) + " \\\\")
        if ds != DS[-1]:
            lines.append("\\midrule")
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{MN[m]}}}" for m in METHODS)
    cmid = " ".join(f"\\cmidrule(lr){{{3 + 3 * k}-{5 + 3 * k}}}" for k in range(len(METHODS)))
    sub = " & ".join(["acc & $\\eta^\\star$ & $W$"] * len(METHODS))
    tex = (r"""\begin{table*}[t]
\centering
\caption{Learning-rate sweeps (5 seeds, final-epoch validation accuracy). acc: best seed-mean accuracy (\%);
$\eta^\star$: learning rate attaining it; $W$: sampled span (decades) of the usable set, i.e., of the learning
rates within 2\,pp of the best SGD-family accuracy on that row ($0$: a single usable rate; ``--'': none).
$^\dagger$: the usable set reaches the grid boundary; $^\ddagger$: the usable set is not contiguous.}
\label{tab:sweep}
\setlength{\tabcolsep}{2.0pt}
\footnotesize
\begin{tabular}{ll""" + "@{\\hspace{7pt}}rrr" * len(METHODS) + r"""}
\toprule
 & & """ + head + r""" \\
""" + cmid + r"""
Data & $L$ & """ + sub + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
    open(os.path.join(TAB, "sweep.tex"), "w").write(tex)


def test_table(T, logreg):
    by = {(r["dataset"], r["arch"], r["method"]): r for r in T if r.get("scale", 1.0) == 1.0}
    lines = []
    for m in METHODS:
        cells = []
        for ds in DS:
            vals = [by.get((ds, a, m)) for a in ARCHS]
            for v in vals:
                cells.append("--" if v is None else f"{100 * v['test_acc']:.1f}")
        lines.append(f"{MN[m]} & " + " & ".join(cells) + " \\\\")
    lr = " & ".join(f"\\multicolumn{{3}}{{c}}{{{100 * logreg[ds]['test_acc']:.1f}}}" for ds in DS)
    lines.append("\\midrule")
    lines.append(f"Log. reg. & {lr} \\\\")
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{DSN[ds]}}}" for ds in DS)
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(3))
    tex = (r"""\begin{table}[t]
\centering
\caption{Test accuracy (\%, mean of 5 seeds) at each method's validation-selected learning rate, for 1, 2 and 3
hidden layers. Logistic regression: $C$ selected on validation.}
\label{tab:test}
\setlength{\tabcolsep}{2.4pt}
\footnotesize
\begin{tabular}{l""" + "rrr" * 3 + r"""}
\toprule
 & """ + head + r""" \\
""" + cmid + r"""
Hidden layers & 1 & 2 & 3 & 1 & 2 & 3 & 1 & 2 & 3 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    open(os.path.join(TAB, "test.tex"), "w").write(tex)


def ablation_table(S, M, Tau):
    """Medium MLP; windows measured against the main sweep's reference R (default settings)."""
    ref = {r["dataset"]: r["ref"] for r in S if r["arch"] == "medium"}
    rows = [("BP+SGD", "default", [r for r in S if r["arch"] == "medium" and r["method"] == "bp_sgd"]),
            ("ProxProp", "default", [r for r in S if r["arch"] == "medium" and r["method"] == "prox_sgd"]),
            ("BP+SGD", "momentum 0.9", [r for r in (M or []) if r["method"] == "bp_sgd"]),
            ("ProxProp", "momentum 0.9", [r for r in (M or []) if r["method"] == "prox_sgd"]),
            ("ProxProp", "$\\tau_\\theta=0.1$", [r for r in (Tau or []) if r["tau"] == 0.1]),
            ("ProxProp", "$\\tau_\\theta=10$", [r for r in (Tau or []) if r["tau"] == 10.0])]
    lines = []
    for name, setting, rs in rows:
        cells = []
        for ds in DS:
            r = next((x for x in rs if x["dataset"] == ds), None)
            if r is None:
                cells.append("-- & -- & --")
                continue
            ok = [float(k) for k, a in r["curve"].items() if a >= ref[ds] - 0.02]
            w = f"{math.log10(max(ok) / min(ok)):.1f}" if ok else "--"
            cells.append(f"{100 * r['peak_val_acc']:.1f} & {lr_short(r['best_lr'])} & {w}")
        lines.append(f"{name} & {setting} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{DSN[ds]}}}" for ds in DS)
    cmid = " ".join(f"\\cmidrule(lr){{{3 + 3 * k}-{5 + 3 * k}}}" for k in range(3))
    tex = (r"""\begin{table}[t]
\centering
\caption{Ablations on the 2-hidden-layer MLP (3 seeds for the ablations, 5 for the defaults). Columns as in
Table~\ref{tab:sweep}; windows use the reference $R$ of the default sweep.}
\label{tab:ablation}
\setlength{\tabcolsep}{2.0pt}
\footnotesize
\begin{tabular}{ll""" + "rrr" * 3 + r"""}
\toprule
 & & """ + head + r""" \\
""" + cmid + r"""
Method & setting & """ + " & ".join(["acc & $\\eta^\\star$ & $W$"] * 3) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    open(os.path.join(TAB, "ablation.tex"), "w").write(tex)


if __name__ == "__main__":
    S = load("summary_main.json")
    if S:
        sweep_table(S)
        ablation_table(S, load("summary_momentum.json"), load("summary_tau.json"))
    T = load("test_summary_main.json")
    if T:
        test_table(T, load("logreg.json"))
    print("results tables written")
