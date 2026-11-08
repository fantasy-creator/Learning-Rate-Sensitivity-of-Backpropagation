"""Data of the web demo's learning-rate explorer, from the main sweep (no training).

For every dataset, depth, learning rate and update rule (BP+SGD, ProxProp, ProxProp-all): the
validation accuracy after every epoch as the mean over the 5 seeds, with the minimum and maximum
as a band, and the seeds that failed numerically. After a failure a run is scored by its
constant predictor, as in the report. Accuracies are stored in tenths of a percent.
Run after src/analyze.py --exp main. Output: docs/data/sweep.js
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "data")
DATASETS = ["drybean", "optdigits", "har"]
ARCHS = ["shallow", "medium", "deep"]
METHODS = ["bp_sgd", "prox_sgd", "prox_all"]
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
EPOCHS, SEEDS, DELTA = 100, 5, 0.02


def per_epoch(run):
    """Validation accuracy after each epoch; after a numerical failure, the constant predictor's."""
    va = run["history"]["val_acc"]
    return [va[e] if e < len(va) and math.isfinite(va[e]) else run["const_val_acc"] for e in range(EPOCHS)]


def permille(x):
    return int(round(1000 * x))


def main():
    summary = {(r["dataset"], r["arch"], r["method"]): r
               for r in json.load(open(os.path.join(ROOT, "results", "summary_main.json")))}
    out = {"lrs": LRS, "epochs": EPOCHS, "seeds": SEEDS, "delta": DELTA, "methods": METHODS, "settings": {}}
    for ds in DATASETS:
        for arch in ARCHS:
            ref = summary[(ds, arch, METHODS[0])]["ref"]
            setting = {"ref": permille(ref), "methods": {}}
            for m in METHODS:
                s = summary[(ds, arch, m)]
                rec = {"best": s["best_lr"], "final": [], "runs": [],
                       # usable as in the report: exact seed-mean accuracy within DELTA of the reference
                       "usable": [s["curve"][f"{lr:g}"] >= ref - DELTA for lr in LRS]}
                for lr in LRS:
                    runs = [json.load(open(os.path.join(ROOT, "results", "main", f"{ds}_{arch}_{m}_lr{lr:g}_s{k}.json")))
                            for k in range(SEEDS)]
                    curves = [per_epoch(r) for r in runs]
                    mean = [sum(c[e] for c in curves) / SEEDS for e in range(EPOCHS)]
                    fails = [{"epoch": r["failure"]["epoch"] + 1, "reason": r["failure"]["reason"]}
                             for r in runs if r["diverged"]]
                    rec["runs"].append({"mean": [permille(v) for v in mean],
                                        "lo": [permille(min(c[e] for c in curves)) for e in range(EPOCHS)],
                                        "hi": [permille(max(c[e] for c in curves)) for e in range(EPOCHS)],
                                        "fails": fails})
                    final = sum(r["final_val_acc"] for r in runs) / SEEDS
                    assert abs(final - s["curve"][f"{lr:g}"]) < 1e-9, (ds, arch, m, lr)
                    rec["final"].append(permille(final))
                setting["methods"][m] = rec
            out["settings"][f"{ds}/{arch}"] = setting
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sweep.js"), "w") as f:
        f.write("window.SWEEP = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print("wrote", os.path.join(OUT, "sweep.js"), f"{os.path.getsize(os.path.join(OUT, 'sweep.js')) // 1024} KB")


if __name__ == "__main__":
    main()
