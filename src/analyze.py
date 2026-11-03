"""Aggregate sweep results (validation only) and test-stage results.

Usage: python src/analyze.py [--exp main]
Writes results/summary_<exp>.json, results/paired_<exp>.json and, if the test stage
has run, results/test_summary_<exp>.json.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SGD_FAMILY = ["bp_sgd", "prox_sgd", "prox_all", "nm_static", "nm_dynamic"]
GRID_MIN, GRID_MAX = 1e-4, 100.0


def load(exp):
    rows = []
    for f in glob.glob(os.path.join(ROOT, "results", exp, "*.json")):
        r = json.load(open(f))
        h = r.pop("history")
        r["damp_first"] = h["damp"][0] if h["damp"] else None
        r["damp_last"] = h["damp"][-1] if h["damp"] else None
        r["upd_first"] = h["upd_norm"][0] if h["upd_norm"] else None
        r["upd_last"] = h["upd_norm"][-1] if h["upd_norm"] else None
        r["val_acc_curve"] = h["val_acc"]
        rows.append(r)
    df = pd.DataFrame(rows)
    for c, v in (("scale", 1.0), ("momentum", 0.0), ("tau", 1.0)):
        if c not in df:
            df[c] = v
    return df


def window(acc_by_lr, ref, delta):
    ok = acc_by_lr[acc_by_lr >= ref - delta / 100.0]
    if len(ok) == 0:
        return dict(count=0, lo=None, hi=None, contig=True, decades=0.0, censored=False)
    lrs = sorted(acc_by_lr.index)
    pos = [lrs.index(x) for x in ok.index]
    lo, hi = float(min(ok.index)), float(max(ok.index))
    return dict(count=int(len(ok)), lo=lo, hi=hi, contig=max(pos) - min(pos) + 1 == len(pos),
                decades=math.log10(hi / lo), censored=lo <= GRID_MIN or hi >= GRID_MAX)


def summarize(df, keys=("dataset", "arch", "scale", "momentum", "tau"), deltas=(1.0, 2.0, 5.0)):
    out = []
    for kv, g in df.groupby(list(keys)):
        mean_acc = g.groupby(["method", "lr"])["final_val_acc"].mean()
        fam = [m for m in SGD_FAMILY if m in mean_acc.index.get_level_values(0)]
        ref = max(mean_acc[m].max() for m in fam) if fam else mean_acc.max()
        for m, gm in g.groupby("method"):
            acc = gm.groupby("lr")["final_val_acc"].mean()
            best_lr = float(acc[acc == acc.max()].index.min())
            at_best = gm[gm.lr == best_lr].sort_values("seed")
            rec = dict(zip(keys, kv), method=m, ref=float(ref), best_lr=best_lr,
                       best_lr_on_edge=best_lr in (GRID_MIN, GRID_MAX),
                       peak_val_acc=float(acc.max()), peak_val_acc_std=float(at_best.final_val_acc.std(ddof=1)),
                       peak_val_f1=float(at_best.final_val_f1.mean()),
                       n_runs=int(len(gm)), n_seeds=int(gm.seed.nunique()), n_nonfinite=int(gm.diverged.sum()),
                       nonfinite_by_lr={f"{k:g}": int(v) for k, v in gm.groupby("lr")["diverged"].sum().items()},
                       min_nonfinite_lr=float(gm[gm.diverged].lr.min()) if gm.diverged.any() else None,
                       time_sec_mean=float(gm.time_sec.mean()),
                       c_static=at_best.c_static.iloc[0],
                       damp_first=at_best.damp_first.iloc[0], damp_last=at_best.damp_last.iloc[0],
                       upd_first=at_best.upd_first.iloc[0], upd_last=at_best.upd_last.iloc[0],
                       curve={f"{k:g}": float(v) for k, v in acc.items()},
                       curve_std={f"{k:g}": float(v) for k, v in gm.groupby("lr")["final_val_acc"].std(ddof=1).items()},
                       curve_f1={f"{k:g}": float(v) for k, v in gm.groupby("lr")["final_val_f1"].mean().items()},
                       curve_loss={f"{k:g}": float(v) for k, v in gm.groupby("lr")["final_val_loss"].mean().items()})
            for dlt in deltas:
                w = window(acc, ref, dlt)
                for k, v in w.items():
                    rec[f"win{dlt:g}_{k}"] = v
            # self-referenced window (own peak), for the "shift vs widen" question
            w = window(acc, float(acc.max()), 2.0)
            for k, v in w.items():
                rec[f"selfwin2_{k}"] = v
            out.append(rec)
    return pd.DataFrame(out)


def paired_peak_test(df, a, b, metric="final_val_acc"):
    """Paired comparison (same seeds) of methods a vs b, each at its own val-selected lr."""
    res = []
    for (ds, arch), g in df.groupby(["dataset", "arch"]):
        sel = {}
        for m in (a, b):
            gm = g[g.method == m]
            if gm.empty:
                break
            acc = gm.groupby("lr")["final_val_acc"].mean()
            lr = acc[acc == acc.max()].index.min()
            sel[m] = gm[gm.lr == lr].set_index("seed")
        if len(sel) < 2:
            continue
        seeds = sorted(set(sel[a].index) & set(sel[b].index))
        da, db = sel[a].loc[seeds, metric].to_numpy(), sel[b].loc[seeds, metric].to_numpy()
        diff = da - db
        p = float(stats.ttest_rel(da, db).pvalue) if len(diff) > 1 and np.std(diff) > 0 else 1.0
        res.append(dict(dataset=ds, arch=arch, metric=metric, a=a, b=b, mean_diff=float(diff.mean()),
                        std_diff=float(diff.std(ddof=1)) if len(diff) > 1 else 0.0, p=p, n=len(seeds)))
    return pd.DataFrame(res)


def test_summary(exp):
    files = glob.glob(os.path.join(ROOT, "results", f"test_{exp}", "*.json"))
    if not files:
        return None
    rows = [json.load(open(f)) for f in files]
    for r in rows:
        r.pop("history", None)
    df = pd.DataFrame(rows)
    keys = [k for k in ("dataset", "arch", "method", "scale", "momentum", "tau") if k in df]
    T = df.groupby(keys).agg(lr=("lr", "first"), test_acc=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
                             test_f1=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
                             val_acc=("final_val_acc", "mean"), n=("seed", "size"),
                             reproduced=("reproduced_val_acc", "all"), nonfinite=("diverged", "sum")).reset_index()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main")
    args = ap.parse_args()
    df = load(args.exp)
    print(f"{len(df)} runs loaded from results/{args.exp}; code versions: {df.code_version.unique().tolist()}")
    S = summarize(df)
    S.to_json(os.path.join(ROOT, "results", f"summary_{args.exp}.json"), orient="records", indent=1)
    if args.exp == "main":
        T = pd.concat([paired_peak_test(df, "prox_sgd", "bp_sgd"), paired_peak_test(df, "prox_sgd", "nm_dynamic"),
                       paired_peak_test(df, "prox_sgd", "nm_static"), paired_peak_test(df, "prox_all", "prox_sgd"),
                       paired_peak_test(df, "bp_adam", "bp_sgd"), paired_peak_test(df, "prox_all", "bp_sgd")])
        T.to_json(os.path.join(ROOT, "results", f"paired_{args.exp}.json"), orient="records", indent=1)
    cols = ["dataset", "arch", "scale", "method", "best_lr", "best_lr_on_edge", "peak_val_acc", "peak_val_acc_std",
            "win2_count", "win2_lo", "win2_hi", "win2_decades", "win2_censored", "win1_count", "win5_count",
            "selfwin2_decades", "n_nonfinite", "min_nonfinite_lr", "n_seeds"]
    with pd.option_context("display.width", 260, "display.max_rows", 500, "display.max_columns", 40):
        print(S[cols].sort_values(["dataset", "arch", "scale", "method"]).to_string(index=False))
        if args.exp == "main":
            print(T.to_string(index=False))
    TS = test_summary(args.exp)
    if TS is not None:
        TS.to_json(os.path.join(ROOT, "results", f"test_summary_{args.exp}.json"), orient="records", indent=1)
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(TS.to_string(index=False))


if __name__ == "__main__":
    main()
