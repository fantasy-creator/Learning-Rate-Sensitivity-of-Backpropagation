"""Run a named experiment grid in parallel on CPU; finished runs are skipped.

Usage: python src/sweep.py --exp {main,scale,momentum,tau,smoke} --workers 24 [--seeds 0 1 2 3 4]
Results: results/<exp>/<run_id>.json (one file per run).
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only, never touch a local GPU
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"  # one BLAS thread per worker; also avoids ~0.7 GB of per-process buffers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core  # noqa: E402

_DATA = {}
_VERSION = core.code_version()


def is_done(path):
    """A result counts as done only if it was produced by the current core.py."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return json.load(f).get("code_version") == _VERSION
    except (OSError, ValueError):
        return False


def grid(exp, seeds):
    lrs = core.LR_GRID
    if exp == "main":  # all 9 dataset x architecture combinations
        methods = ["bp_sgd", "prox_sgd", "prox_all", "nm_static", "nm_dynamic", "bp_adam"]
        for ds, arch, m, lr, s in itertools.product(core.DATASETS, core.ARCHS, methods, lrs, seeds):
            yield dict(dataset=ds, arch=arch, method=m, lr=lr, seed=s)
    elif exp == "scale":  # input-scale stress test on the medium MLP
        for ds, sc, m, lr, s in itertools.product(core.DATASETS, [0.1, 10.0],
                                                  ["bp_sgd", "prox_sgd", "prox_all"], lrs, seeds):
            yield dict(dataset=ds, arch="medium", method=m, lr=lr, seed=s, scale=sc)
    elif exp == "momentum":  # heavy-ball momentum 0.9 as in Rumelhart et al. (1986), Eq. 9
        for ds, m, lr, s in itertools.product(core.DATASETS, ["bp_sgd", "prox_sgd"], lrs, seeds):
            yield dict(dataset=ds, arch="medium", method=m, lr=lr, seed=s, momentum=0.9)
    elif exp == "tau":  # proximal parameter ablation
        for ds, t, lr, s in itertools.product(core.DATASETS, [0.1, 10.0], lrs, seeds):
            yield dict(dataset=ds, arch="medium", method="prox_sgd", lr=lr, seed=s, tau=t)
    elif exp == "smoke":
        for ds, m in itertools.product(core.DATASETS, ["bp_sgd", "prox_sgd", "prox_all", "nm_static",
                                                       "nm_dynamic", "bp_adam"]):
            yield dict(dataset=ds, arch="deep", method=m, lr=0.1 if m != "bp_adam" else 1e-3,
                       seed=0, epochs=2)
    else:
        raise ValueError(exp)


def work(args):
    cfg, out_dir = args
    import torch
    torch.set_num_threads(1)
    rid = core.run_id(cfg)
    path = os.path.join(out_dir, rid + ".json")
    if is_done(path):
        return rid, "skip", 0.0
    key = (cfg["dataset"], cfg.get("scale", 1.0), cfg.get("standardize", True))
    if key not in _DATA:
        _DATA[key] = core.load_data(cfg["dataset"], scale=key[1], standardize=key[2])
    try:
        res = core.train_run(data=_DATA[key], **cfg)
    except Exception:
        return rid, "ERROR\n" + traceback.format_exc(), 0.0
    core.save_json(res, path)
    return rid, "div" if res["diverged"] else "ok", res["time_sec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_dir = os.path.join(core.ROOT, "results", args.exp)
    os.makedirs(out_dir, exist_ok=True)
    cfgs = list(grid(args.exp, args.seeds))
    # longest jobs first (HAR, deep, prox-type) for better load balancing
    cost = lambda c: ({"har": 3, "drybean": 2, "optdigits": 1}[c["dataset"]]
                      * {"deep": 3, "medium": 2, "shallow": 1}[c["arch"]]
                      * (2 if c["method"] in ("prox_sgd", "prox_all", "nm_dynamic") else 1))
    cfgs.sort(key=cost, reverse=True)
    if args.limit:
        cfgs = cfgs[: args.limit]
    todo = [c for c in cfgs if not is_done(os.path.join(out_dir, core.run_id(c) + ".json"))]
    print(f"[{args.exp}] {len(cfgs)} runs, {len(todo)} to do, {args.workers} workers", flush=True)
    t0 = time.time()
    done = 0
    stats = {"ok": 0, "div": 0, "err": 0}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for rid, status, sec in pool.imap_unordered(work, [(c, out_dir) for c in todo], chunksize=1):
            done += 1
            if status.startswith("ERROR"):
                stats["err"] += 1
                print(f"!! {rid}: {status}", flush=True)
            elif status in stats:
                stats[status] += 1
            if done % 50 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"[{args.exp}] {done}/{len(todo)} done, {stats}, elapsed {el/60:.1f} min, "
                      f"eta {el/done*(len(todo)-done)/60:.1f} min", flush=True)
    print(f"[{args.exp}] finished in {(time.time()-t0)/60:.1f} min: {stats}", flush=True)


if __name__ == "__main__":
    main()
