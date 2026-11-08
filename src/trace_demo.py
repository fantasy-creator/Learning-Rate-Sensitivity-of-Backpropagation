"""Per-epoch traces of a few runs for the experiment video (CPU, a few seconds with 12 workers).

Optdigits, two hidden layers, seed 0, BP+SGD / ProxProp / ProxProp-all at a few learning rates.
After every epoch: training loss, validation accuracy, weight norms, and the predictions on 60
validation digits (the first six of each class). These are seed-0 runs of the main sweep, and the
final validation accuracy of each must equal results/main exactly. After a numerical failure the
run predicts with its constant predictor, the scoring convention of the report. Prints the training
log of every run. Output: results/demo_trace.json
"""
import json
import multiprocessing as mp
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import core  # noqa: E402

DATASET, ARCH, SEED, EPOCHS = "optdigits", "medium", 0, 100
METHODS = ["bp_sgd", "prox_sgd", "prox_all"]
LRS = [0.01, 1.0, 10.0, 100.0]
PER_CLASS = 6


def digit_index(y):
    return [int(i) for c in range(10) for i in np.flatnonzero(y == c)[:PER_CLASS]]


def trace(job):
    import torch
    torch.set_num_threads(1)
    m, lr = job
    D = core.load_data(DATASET)
    idx = digit_index(D["y_val"].numpy())
    xv = D["x_val"][idx]
    res = core.train_run(DATASET, ARCH, m, lr, SEED, epochs=EPOCHS, data=D, snapshot_epochs=list(range(EPOCHS + 1)))
    const = int(np.argmax(np.bincount(D["y_train"].numpy(), minlength=D["C"])))
    preds = []
    for ep in range(EPOCHS + 1):  # predictions after ep epochs; ep = 0 is the initialization
        if ep in res["snapshots"]:
            with torch.no_grad():
                logits, _, _ = core.forward(res["snapshots"][ep], xv)
            preds.append([int(v) for v in logits.argmax(1)])
        else:
            preds.append([const] * len(idx))
    h = res["history"]
    ref = json.load(open(os.path.join(core.ROOT, "results", "main", f"{DATASET}_{ARCH}_{m}_lr{lr:g}_s{SEED}.json")))
    lines = []
    for k, e in enumerate(h["epoch"]):
        lines.append(f"[{m:<9s} lr={lr:<5g}] epoch {e:3d}/{EPOCHS}  loss {h['train_loss'][k]:9.3e}  "
                     f"val acc {100 * h['val_acc'][k]:5.1f}%  |W| {max(h['w_norm'][k]):8.2e}")
    if res["failure"]:
        f = res["failure"]
        lines.append(f"[{m:<9s} lr={lr:<5g}] numerical failure ({f['reason']}) in epoch {f['epoch'] + 1}: "
                     "constant predictor")
    return dict(method=m, lr=lr, train_loss=h["train_loss"], val_acc=h["val_acc"],
                w_norm=[max(w) for w in h["w_norm"]], upd_norm=h["upd_norm"], preds=preds,
                failure=res["failure"], final_val_acc=res["final_val_acc"],
                reproduced=abs(res["final_val_acc"] - ref["final_val_acc"]) < 1e-12, log=lines)


def main():
    z = np.load(os.path.join(core.DATA_DIR, f"{DATASET}.npz"), allow_pickle=False)
    idx = digit_index(z["y_val"])
    jobs = [(m, lr) for lr in LRS for m in METHODS]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        runs = pool.map(trace, jobs)
    for r in runs:
        print("\n".join(r["log"][-3:]), flush=True)
    bad = [(r["method"], r["lr"]) for r in runs if not r["reproduced"]]
    print(f"{len(runs)} runs, {len(runs) - len(bad)} reproduce the sweep exactly")
    if bad:
        raise SystemExit(f"not reproduced: {bad}")
    out = dict(dataset=DATASET, arch=ARCH, seed=SEED, epochs=EPOCHS, code_version=core.code_version(),
               digits=[dict(label=int(z["y_val"][i]), x=[int(v) for v in z["x_val"][i]]) for i in idx], runs=runs)
    path = os.path.join(core.ROOT, "results", "demo_trace.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print("wrote", path)


if __name__ == "__main__":
    main()
