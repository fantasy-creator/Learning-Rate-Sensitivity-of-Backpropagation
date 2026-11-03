"""Controlled per-step cost of each update rule (single process, 1 thread, warmed up).

Times only the update step (forward + backward + parameter update), excluding evaluation,
on the deep MLP with B=128. Run on an otherwise idle machine.
Output: results/bench_step.json
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import core  # noqa: E402

torch.set_num_threads(1)
METHODS = ["bp_sgd", "prox_sgd", "prox_all", "nm_static", "nm_dynamic", "bp_adam"]
out = {}
for ds in core.DATASETS:
    D = core.load_data(ds)
    for m in METHODS:
        lr = 1e-3 if m == "bp_adam" else 0.01
        core.train_run(ds, "deep", m, lr, seed=0, epochs=1, data=D)  # warm-up
        times = []
        for rep in range(5):  # T(6 epochs) - T(1 epoch) cancels initialisation and the final evaluation
            t0 = time.perf_counter()
            core.train_run(ds, "deep", m, lr, seed=rep, epochs=1, data=D, log_every=1000)
            t1 = time.perf_counter()
            r = core.train_run(ds, "deep", m, lr, seed=rep, epochs=6, data=D, log_every=1000)
            t2 = time.perf_counter()
            times.append(((t2 - t1) - (t1 - t0)) / (5 * r["steps_per_epoch"]))
        out[f"{ds}_{m}"] = dict(ms_per_step=1000 * min(times), d=D["d"])
        print(f"{ds:9s} {m:10s} {1000 * min(times):7.3f} ms/step", flush=True)
json.dump(out, open(os.path.join(core.ROOT, "results", "bench_step.json"), "w"), indent=1)
