"""Train and export the networks of the web demo's digit panel (CPU, about a minute with 13 workers).

Optdigits, two hidden layers (64-64-32-10), seed 0 and every learning rate of the grid, for BP+SGD,
ProxProp and ProxProp-all. These are the seed-0 runs of the main sweep: the final validation
accuracy of every run must equal results/main exactly. A run that failed numerically is exported
as its constant predictor (the training class frequencies), the scoring convention of the report.
Also exports the input standardization and three test digits per class, which the page only
uses as example inputs.
Usage: python src/export_demo.py [--workers 13]. Output: docs/data/models.js
"""
import argparse
import base64
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
OUT = os.path.join(core.ROOT, "docs", "data")


def b64(a):
    """float32 where it is exact enough; float64 for the huge weights of some large-rate runs."""
    with np.errstate(over="ignore"):
        f4 = np.ascontiguousarray(a, dtype="<f4")
    dtype = "f4" if np.isfinite(f4).all() else "f8"
    data = f4 if dtype == "f4" else np.ascontiguousarray(a, dtype="<f8")
    return dict(dtype=dtype, data=base64.b64encode(data.tobytes()).decode("ascii"))


def train(job):
    import torch
    torch.set_num_threads(1)
    m, lr = job
    D = core.load_data(DATASET)
    res = core.train_run(DATASET, ARCH, m, lr, SEED, epochs=EPOCHS, data=D, snapshot_epochs=[EPOCHS])
    ref = json.load(open(os.path.join(core.ROOT, "results", "main", f"{DATASET}_{ARCH}_{m}_lr{lr:g}_s{SEED}.json")))
    out = dict(method=m, lr=lr, val_acc=res["final_val_acc"],
               reproduced=abs(res["final_val_acc"] - ref["final_val_acc"]) < 1e-12)
    if res["diverged"]:
        freq = np.bincount(D["y_train"].numpy(), minlength=D["C"]) / len(D["y_train"])
        out.update(failed=True, reason=res["failure"]["reason"], epoch=res["failure"]["epoch"] + 1, const=freq.tolist())
    else:
        out.update(failed=False, max_abs_weight=max(float(W.abs().max()) for W, _ in res["snapshots"][EPOCHS]),
                   layers=[dict(shape=list(W.shape), W=b64(W.numpy()), b=b64(b.numpy()))
                           for W, b in res["snapshots"][EPOCHS]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=13)
    args = ap.parse_args()
    z = np.load(os.path.join(core.DATA_DIR, f"{DATASET}.npz"), allow_pickle=False)
    jobs = [(m, lr) for m in METHODS for lr in core.LR_GRID]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        models = pool.map(train, jobs)
    bad = [(r["method"], r["lr"]) for r in models if not r["reproduced"]]
    print(f"{len(models)} models, {sum(r['failed'] for r in models)} failed numerically, "
          f"{len(models) - len(bad)} reproduce the sweep exactly", flush=True)
    if bad:
        raise SystemExit(f"not reproduced: {bad}")
    examples = []
    for c in range(10):  # the first three test digits of each class, raw block counts 0..16
        for i in np.flatnonzero(z["y_test"] == c)[:3]:
            examples.append(dict(label=int(c), x=[int(v) for v in z["x_test"][i]]))
    out = dict(dataset=DATASET, arch=ARCH, seed=SEED, epochs=EPOCHS, lrs=core.LR_GRID, methods=METHODS,
               mean=z["mean"].tolist(), std=z["std"].tolist(), code_version=core.code_version(),
               models=models, examples=examples)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "models.js"), "w") as f:
        f.write("window.DEMO_MODELS = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print("wrote", os.path.join(OUT, "models.js"), f"{os.path.getsize(os.path.join(OUT, 'models.js')) // 1024} KB")


if __name__ == "__main__":
    main()
