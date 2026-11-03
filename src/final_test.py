"""Test stage: re-run each method's validation-selected configuration with test evaluation.

Selection (pre-specified, validation only): for every dataset x architecture x method,
the learning rate with the highest seed-mean final validation accuracy in results/<exp>
(ties -> smaller lr). The selected configurations are re-run deterministically with
the same seeds; the re-run must reproduce the sweep's validation accuracy exactly.
Output: results/test_<exp>/<run_id>.json and results/test_<exp>_selection.json
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402


def select(exp):
    acc = defaultdict(list)
    for f in glob.glob(os.path.join(core.ROOT, "results", exp, "*.json")):
        r = json.load(open(f))
        key = (r["dataset"], r["arch"], r["method"], r.get("scale", 1.0), r.get("momentum", 0.0), r.get("tau", 1.0))
        acc[key + (r["lr"],)].append((r["seed"], r["final_val_acc"]))
    best = {}
    for k, v in acc.items():
        group, lr = k[:-1], k[-1]
        m = sum(a for _, a in v) / len(v)
        if group not in best or m > best[group][1] + 1e-12 or (abs(m - best[group][1]) <= 1e-12 and lr < best[group][0]):
            best[group] = (lr, m, sorted(v))
    return best


def work(cfg):
    import torch
    torch.set_num_threads(1)
    out_dir, want = cfg.pop("_out"), cfg.pop("_val")
    res = core.train_run(eval_test=True, **cfg)
    res["reproduced_val_acc"] = abs(res["final_val_acc"] - want) < 1e-12
    core.save_json(res, os.path.join(out_dir, core.run_id(cfg) + ".json"))
    return core.run_id(cfg), res["reproduced_val_acc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    best = select(args.exp)
    out_dir = os.path.join(core.ROOT, "results", f"test_{args.exp}")
    os.makedirs(out_dir, exist_ok=True)
    jobs, sel = [], []
    for (ds, arch, m, sc, mom, tau), (lr, mean_acc, seeds) in sorted(best.items()):
        sel.append(dict(dataset=ds, arch=arch, method=m, scale=sc, momentum=mom, tau=tau, lr=lr,
                        mean_val_acc=mean_acc))
        for seed, va in seeds:
            jobs.append(dict(dataset=ds, arch=arch, method=m, lr=lr, seed=seed, scale=sc, momentum=mom,
                             tau=tau, _out=out_dir, _val=va))
    json.dump(sel, open(os.path.join(core.ROOT, "results", f"test_{args.exp}_selection.json"), "w"), indent=1)
    print(f"{len(sel)} selected configurations, {len(jobs)} test runs")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        ok = [r for _, r in pool.imap_unordered(work, jobs)]
    print(f"done: {sum(ok)}/{len(ok)} runs reproduced the sweep's validation accuracy exactly")


if __name__ == "__main__":
    main()
