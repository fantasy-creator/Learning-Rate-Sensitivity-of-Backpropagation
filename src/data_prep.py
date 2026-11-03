"""Build fixed train/val/test splits for the three UCI datasets.

Usage: python src/data_prep.py --raw data/raw --out data/processed

Reads the raw UCI files and writes <out>/<name>.npz with raw (unstandardized) float64 features,
integer labels, and the training-split mean/std used for z-scoring.

Split protocol (fixed, seed 0):
  drybean   : exact duplicate rows removed, then stratified 70/15/15.
  optdigits : official train/test kept; 20% of official train -> val (stratified).
  har       : official subject-wise train/test kept; 4 of the 21 training
              subjects -> val, so val subjects are disjoint from train subjects.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SEED = 0


def stratified_split(y, fracs, rng):
    """Return index arrays for each fraction, stratified by label."""
    parts = [[] for _ in fracs]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n = len(idx)
        cuts = np.round(np.cumsum(fracs)[:-1] * n).astype(int)
        for k, chunk in enumerate(np.split(idx, cuts)):
            parts[k].append(chunk)
    return [np.sort(np.concatenate(p)) for p in parts]


def zstats(x):
    mu = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-8] = 1.0  # constant columns (e.g. border pixels) stay at zero
    return mu, sd


def load_drybean(root):
    path = os.path.join(root, "drybean", "DryBeanDataset", "Dry_Bean_Dataset.arff")
    rows, names, in_data = [], [], False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if s.upper().startswith("@ATTRIBUTE"):
                names.append(s.split()[1])
            elif s.upper().startswith("@DATA"):
                in_data = True
            elif in_data:
                rows.append(s.split(","))
    df = pd.DataFrame(rows, columns=names)
    x = df.iloc[:, :-1].astype(np.float64).to_numpy()
    lab = df.iloc[:, -1].str.strip().to_numpy()
    classes = sorted(np.unique(lab).tolist())
    y = np.array([classes.index(v) for v in lab], dtype=np.int64)
    n_raw = len(y)
    key = np.concatenate([x, y[:, None]], 1)
    _, keep = np.unique(key, axis=0, return_index=True)
    keep = np.sort(keep)
    x, y = x[keep], y[keep]
    rng = np.random.default_rng(SEED)
    tr, va, te = stratified_split(y, [0.70, 0.15, 0.15], rng)
    info = dict(n_raw=n_raw, n_dedup=len(y), n_duplicates_removed=n_raw - len(y),
                classes=classes, features=names[:-1])
    return x, y, tr, va, te, info


def load_optdigits(root):
    tra = np.loadtxt(os.path.join(root, "optdigits", "optdigits.tra"), delimiter=",")
    tes = np.loadtxt(os.path.join(root, "optdigits", "optdigits.tes"), delimiter=",")
    x = np.concatenate([tra[:, :-1], tes[:, :-1]]).astype(np.float64)
    y = np.concatenate([tra[:, -1], tes[:, -1]]).astype(np.int64)
    n_tr = len(tra)
    rng = np.random.default_rng(SEED)
    sub_tr, sub_va = stratified_split(y[:n_tr], [0.80, 0.20], rng)
    te = np.arange(n_tr, len(y))
    # duplicate audit (reported, not removed: the official test set is kept intact)
    tr_keys = {r.tobytes() for r in tra[:, :-1].astype(np.int64)}
    cross = sum(r.tobytes() in tr_keys for r in tes[:, :-1].astype(np.int64))
    within = n_tr - len(np.unique(tra[:, :-1], axis=0))
    info = dict(n_raw=len(y), official_train=n_tr, official_test=len(tes),
                duplicates_within_train=int(within), test_rows_identical_to_train=int(cross),
                classes=[str(i) for i in range(10)])
    return x, y, sub_tr, sub_va, te, info


def load_har(root):
    base = os.path.join(root, "har", "UCI HAR Dataset")
    rd = lambda p: pd.read_csv(os.path.join(base, p), sep=r"\s+", header=None).to_numpy()
    xtr, xte = rd("train/X_train.txt").astype(np.float64), rd("test/X_test.txt").astype(np.float64)
    ytr, yte = rd("train/y_train.txt").ravel() - 1, rd("test/y_test.txt").ravel() - 1
    str_, ste = rd("train/subject_train.txt").ravel(), rd("test/subject_test.txt").ravel()
    x = np.concatenate([xtr, xte])
    y = np.concatenate([ytr, yte]).astype(np.int64)
    subj = np.concatenate([str_, ste])
    n_tr = len(ytr)
    rng = np.random.default_rng(SEED)
    train_subjects = np.unique(str_)
    val_subjects = np.sort(rng.choice(train_subjects, size=4, replace=False))
    is_val = np.isin(str_, val_subjects)
    tr = np.flatnonzero(~is_val)
    va = np.flatnonzero(is_val)
    te = np.arange(n_tr, len(y))
    labels = rd("activity_labels.txt")[:, 1].tolist()
    info = dict(n_raw=len(y), official_train=n_tr, official_test=len(yte),
                val_subjects=val_subjects.tolist(),
                train_subjects=sorted(set(train_subjects.tolist()) - set(val_subjects.tolist())),
                test_subjects=np.unique(ste).tolist(), classes=labels)
    return x, y, tr, va, te, info


LOADERS = dict(drybean=load_drybean, optdigits=load_optdigits, har=load_har)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for name, fn in LOADERS.items():
        x, y, tr, va, te, info = fn(args.raw)
        assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0
        mu, sd = zstats(x[tr])
        np.savez(os.path.join(args.out, f"{name}.npz"),
                 x_train=x[tr], y_train=y[tr], x_val=x[va], y_val=y[va],
                 x_test=x[te], y_test=y[te], mean=mu, std=sd)
        C = int(y.max()) + 1
        info.update(d=int(x.shape[1]), C=C, n_train=len(tr), n_val=len(va), n_test=len(te),
                    class_counts_train=np.bincount(y[tr], minlength=C).tolist(),
                    class_counts_val=np.bincount(y[va], minlength=C).tolist(),
                    class_counts_test=np.bincount(y[te], minlength=C).tolist(),
                    raw_feature_abs_max=float(np.abs(x[tr]).max()),
                    raw_feature_std_min=float(x[tr].std(0).min()),
                    raw_feature_std_max=float(x[tr].std(0).max()),
                    constant_train_columns=int((x[tr].std(0) < 1e-8).sum()))
        summary[name] = info
        print(name, {k: info[k] for k in ("d", "C", "n_train", "n_val", "n_test")})
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
