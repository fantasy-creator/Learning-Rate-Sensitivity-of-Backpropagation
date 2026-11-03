"""Shared pieces: data loading, MLPs, BP / ProxProp update rules, evaluation.

All parameter updates are written out by hand (no torch.optim except Adam) so
that each method is exactly the rule analysed in the paper:

  bp_sgd      W <- W - lr * g                                  (all layers)
  prox_sgd    W <- W - lr * g_hat (I/tau + A_hat A_hat^T)^-1   (hidden layers;
              output layer explicit, as in Frerix et al. 2018, Eq. 12)
  prox_all    proximal update on every layer, output layer included
  nm_static   SGD with per-layer lr * c_l, c_l = ||prox step|| / ||BP step||
              measured once at initialisation (output layer c = 1)
  nm_dynamic  BP direction, ProxProp length: lr * s_l(t) * g with
              s_l(t) = ||prox step|| / ||BP step|| recomputed every step
  bp_adam     Adam on the BP gradient

g_hat = [gW gb] and A_hat = [A; 1^T] fold the bias into the proximal problem.
Everything runs in float64 on CPU: in float32 the damped ProxProp steps at small
learning rates are partly rounded away, which would bias the low-lr edge.
"""
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
DTYPE = torch.float64

ARCHS = {"shallow": [32], "medium": [64, 32], "deep": [128, 64, 32]}
DATASETS = ["drybean", "optdigits", "har"]
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]


def code_version():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


class NumericalFailure(Exception):
    pass


# ----------------------------------------------------------------------------- data
def load_data(name, scale=1.0, standardize=True, with_test=False):
    z = np.load(os.path.join(DATA_DIR, f"{name}.npz"), allow_pickle=False)
    out = {}
    for split in ("train", "val", "test"):
        if split == "test" and not with_test:
            continue  # the test split is only loaded for the val-selected configurations
        x = z[f"x_{split}"]
        if standardize:
            x = (x - z["mean"]) / z["std"]
        out[f"x_{split}"] = torch.tensor(x * scale, dtype=DTYPE)
        out[f"y_{split}"] = torch.tensor(z[f"y_{split}"], dtype=torch.long)
    out["d"] = out["x_train"].shape[1]
    out["C"] = int(max(z["y_train"].max(), z["y_val"].max(), z["y_test"].max())) + 1
    return out


# ----------------------------------------------------------------------------- model
def init_mlp(d, hidden, C, seed):
    """Same initialisation as torch.nn.Linear (U(+-1/sqrt(fan_in)) for W and b)."""
    torch.manual_seed(seed)
    sizes = [d] + list(hidden) + [C]
    params = []
    for i in range(len(sizes) - 1):
        lin = torch.nn.Linear(sizes[i], sizes[i + 1])
        params.append([lin.weight.detach().to(DTYPE).clone().requires_grad_(True),
                       lin.bias.detach().to(DTYPE).clone().requires_grad_(True)])
    return params


def forward(params, x):
    """Return logits, inputs A_l (batch x d_l) and pre-activations Z_l of every linear layer."""
    acts, zs = [], []
    h = x
    L = len(params)
    for l, (W, b) in enumerate(params):
        acts.append(h)
        z = h @ W.t() + b
        zs.append(z)
        h = torch.relu(z) if l < L - 1 else z
    return h, acts, zs


def backprop(params, x, y):
    """One forward/backward pass. Returns loss, layer inputs A_l, errors G_l = dL/dZ_l (batch x m)
    and parameter gradients (gW_l = G_l^T A_l, gb_l = 1^T G_l), all detached."""
    logits, acts, zs = forward(params, x)
    loss = F.cross_entropy(logits, y)
    Gs = torch.autograd.grad(loss, zs)
    acts = [a.detach() for a in acts]
    grads = [(G.t() @ A, G.sum(0)) for G, A in zip(Gs, acts)]
    return loss.detach(), acts, Gs, grads


# ----------------------------------------------------------------------------- updates
def prox_direction(G, A, tau=1.0):
    """Exact ProxProp direction g_hat (I/tau + A_hat^T A_hat)^{-1} for one layer.

    G (batch x m) = dL/dZ, A (batch x d) = layer input; A_hat = [A 1] (batch x (d+1)),
    g_hat = G^T A_hat. If d+1 <= batch, solve the (d+1)x(d+1) system; otherwise use the
    push-through identity g_hat M^{-1} = (K^{-1} G)^T A_hat with K = I/tau + A_hat A_hat^T,
    which is exact and avoids the cancellation of the Woodbury form.
    Returns (dW, db).
    """
    Bn, d = A.shape
    Ah = torch.cat([A, torch.ones(Bn, 1, dtype=A.dtype)], 1)
    c = 1.0 / tau
    if not (torch.isfinite(Ah).all() and torch.isfinite(G).all()):
        raise NumericalFailure("nonfinite_solve_input")
    try:
        if d + 1 <= Bn:
            M = Ah.t() @ Ah
            M.diagonal().add_(c)
            gh = G.t() @ Ah
            out = torch.cholesky_solve(gh.t(), torch.linalg.cholesky(M)).t()
        else:
            K = Ah @ Ah.t()
            K.diagonal().add_(c)
            out = torch.cholesky_solve(G, torch.linalg.cholesky(K)).t() @ Ah
    except RuntimeError as e:  # Cholesky can fail only for astronomically large activations
        raise NumericalFailure("solve_failed") from e
    if not torch.isfinite(out).all():
        raise NumericalFailure("nonfinite_solve_output")
    return out[:, :d], out[:, d]


def frob(*ts):
    return math.sqrt(sum(float((t.detach().double() ** 2).sum()) for t in ts))


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(params, x, y, C):
    logits, _, _ = forward(params, x)
    if not torch.isfinite(logits).all():
        return float("nan"), float("nan"), float("nan")
    loss = float(F.cross_entropy(logits, y))
    pred = logits.argmax(1)
    acc = float((pred == y).double().mean())
    return loss, acc, macro_f1(pred, y, C)


def macro_f1(pred, y, C):
    cm = torch.zeros(C, C, dtype=torch.float64)
    cm.index_put_((y, pred), torch.ones_like(y, dtype=torch.float64), accumulate=True)
    tp = cm.diag()
    prec = tp / cm.sum(0).clamp_min(1e-12)
    rec = tp / cm.sum(1).clamp_min(1e-12)
    f1 = torch.where(prec + rec > 0, 2 * prec * rec / (prec + rec), torch.zeros_like(tp))
    return float(f1.mean())


def constant_predictor(y_train, y_eval, C):
    """Scores of the fallback model used for failed runs: it predicts the training class
    frequencies (so its cross-entropy is defined) and hence the training-majority class."""
    freq = torch.bincount(y_train, minlength=C).double() / len(y_train)
    loss = float(-torch.log(freq[y_eval].clamp_min(1e-300)).mean())
    pred = torch.full_like(y_eval, int(freq.argmax()))
    return loss, float((pred == y_eval).double().mean()), macro_f1(pred, y_eval, C)


# ----------------------------------------------------------------------------- training
def train_run(dataset, arch, method, lr, seed, epochs=100, batch=128, tau=1.0,
              momentum=0.0, scale=1.0, standardize=True, nm_batches=10, data=None,
              log_every=1, snapshot_epochs=None, eval_test=False):
    t0 = time.time()
    snaps = {}
    D = data if data is not None else load_data(dataset, scale=scale, standardize=standardize,
                                                 with_test=eval_test)
    d, C = D["d"], D["C"]
    xtr, ytr = D["x_train"], D["y_train"]
    n = xtr.shape[0]
    params = init_mlp(d, ARCHS[arch], C, seed)
    L = len(params)
    flat = [p for pair in params for p in pair]
    gen = torch.Generator().manual_seed(10_000 + seed)  # batch order shared across methods
    steps_per_epoch = n // batch  # drop_last: every step uses exactly `batch` samples

    opt = torch.optim.Adam(flat, lr=lr) if method == "bp_adam" else None
    vel = [[torch.zeros_like(W), torch.zeros_like(b)] for W, b in params] if momentum > 0 else None
    failure = None

    # norm-matched static scale: ratio of prox to BP step norm at initialisation,
    # averaged over the first `nm_batches` mini-batches of epoch 0 (no updates).
    c_static = [1.0] * L
    if method == "nm_static":
        perm0 = torch.randperm(n, generator=torch.Generator().manual_seed(10_000 + seed))
        ratios = [[] for _ in range(L)]
        for k in range(min(nm_batches, steps_per_epoch)):
            idx = perm0[k * batch:(k + 1) * batch]
            _, acts, Gs, grads = backprop(params, xtr[idx], ytr[idx])
            for l in range(L - 1):
                pW, pb = prox_direction(Gs[l], acts[l], tau)
                gn = frob(*grads[l])
                if gn > 0:
                    ratios[l].append(frob(pW, pb) / gn)
        c_static = [float(np.mean(r)) if r else 1.0 for r in ratios]
        c_static[-1] = 1.0

    hist = {k: [] for k in ("epoch", "train_loss", "full_train_loss", "val_loss", "val_acc", "val_f1",
                            "upd_norm", "w_norm", "damp")}
    snapshot_epochs = set(snapshot_epochs or [])
    for ep in range(epochs):
        if ep in snapshot_epochs:
            snaps[ep] = [[W.detach().clone(), b.detach().clone()] for W, b in params]
        perm = torch.randperm(n, generator=gen)
        ep_loss, upd_acc, damp_acc = 0.0, [0.0] * L, [0.0] * L
        try:
            for k in range(steps_per_epoch):
                idx = perm[k * batch:(k + 1) * batch]
                loss, acts, Gs, grads = backprop(params, xtr[idx], ytr[idx])
                if not torch.isfinite(loss):
                    raise NumericalFailure("nonfinite_loss")
                ep_loss += float(loss)
                with torch.no_grad():
                    before = [[W.clone(), b.clone()] for W, b in params]
                    if opt is not None:
                        for (W, b), (gW, gb) in zip(params, grads):
                            W.grad, b.grad = gW, gb
                        opt.step()
                    else:
                        for l in range(L):
                            W, b = params[l]
                            gW, gb = grads[l]
                            explicit = method == "bp_sgd" or (method != "prox_all" and l == L - 1)
                            if explicit:
                                dW, db = gW, gb
                            elif method in ("prox_sgd", "prox_all"):
                                dW, db = prox_direction(Gs[l], acts[l], tau)
                                gn = frob(gW, gb)
                                damp_acc[l] += frob(dW, db) / gn if gn > 0 else 0.0
                            elif method == "nm_static":
                                dW, db = c_static[l] * gW, c_static[l] * gb
                            elif method == "nm_dynamic":
                                pW, pb = prox_direction(Gs[l], acts[l], tau)
                                gn = frob(gW, gb)
                                s = frob(pW, pb) / gn if gn > 0 else 0.0
                                damp_acc[l] += s
                                dW, db = s * gW, s * gb
                            else:
                                raise ValueError(method)
                            if vel is not None:  # heavy ball, Rumelhart et al. (1986) Eq. 9
                                vel[l][0].mul_(momentum).add_(dW, alpha=-lr)
                                vel[l][1].mul_(momentum).add_(db, alpha=-lr)
                                W.add_(vel[l][0])
                                b.add_(vel[l][1])
                            else:
                                W.add_(dW, alpha=-lr)
                                b.add_(db, alpha=-lr)
                    for l in range(L):  # applied (not nominal) step length, same for all methods
                        upd_acc[l] += frob(params[l][0] - before[l][0], params[l][1] - before[l][1])
                if not all(torch.isfinite(p).all() for p in flat):
                    raise NumericalFailure("nonfinite_params")
        except NumericalFailure as e:
            failure = dict(reason=str(e), epoch=ep, step=k)
            break
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            ftl, _, _ = evaluate(params, xtr, ytr, C)
            vl, va, vf = evaluate(params, D["x_val"], D["y_val"], C)
            hist["epoch"].append(ep + 1)
            hist["train_loss"].append(ep_loss / steps_per_epoch)
            hist["full_train_loss"].append(ftl)
            hist["val_loss"].append(vl)
            hist["val_acc"].append(va)
            hist["val_f1"].append(vf)
            hist["upd_norm"].append([u / steps_per_epoch for u in upd_acc])
            hist["w_norm"].append([frob(W, b) for W, b in params])
            hist["damp"].append([s / steps_per_epoch for s in damp_acc])
            if not (math.isfinite(ftl) and math.isfinite(vl)):
                failure = dict(reason="nonfinite_eval", epoch=ep, step=steps_per_epoch)
                break

    cfg = dict(dataset=dataset, arch=arch, method=method, lr=lr, seed=seed, epochs=epochs,
               batch=batch, tau=tau, momentum=momentum, scale=scale, standardize=standardize,
               nm_batches=nm_batches)
    res = dict(cfg, code_version=code_version(), torch_version=torch.__version__,
               steps_per_epoch=steps_per_epoch, c_static=c_static,
               diverged=failure is not None, failure=failure, history=hist)
    cvl, cva, cvf = constant_predictor(ytr, D["y_val"], C)
    res.update(const_val_loss=cvl, const_val_acc=cva, const_val_f1=cvf)
    if failure is None:
        res.update(final_val_loss=hist["val_loss"][-1], final_val_acc=hist["val_acc"][-1],
                   final_val_f1=hist["val_f1"][-1], final_train_loss=hist["full_train_loss"][-1])
    else:  # scoring convention for failed runs: the constant predictor above
        res.update(final_val_loss=cvl, final_val_acc=cva, final_val_f1=cvf, final_train_loss=float("nan"))
    if eval_test:
        if failure is None:
            tl, ta, tf = evaluate(params, D["x_test"], D["y_test"], C)
            if not math.isfinite(tl):
                res["diverged"], res["failure"] = True, dict(reason="nonfinite_test_eval", epoch=epochs)
                tl, ta, tf = constant_predictor(ytr, D["y_test"], C)
        else:
            tl, ta, tf = constant_predictor(ytr, D["y_test"], C)
        res.update(test_loss=tl, test_acc=ta, test_f1=tf)
    # pre-divergence peak (diagnostic only; never used for scoring)
    fin = [(e, v) for e, v in zip(hist["epoch"], hist["val_acc"]) if v == v]
    res["peak_val_acc_diag"] = max(v for _, v in fin) if fin else None
    res["peak_epoch_diag"] = max(fin, key=lambda t: t[1])[0] if fin else None
    if snapshot_epochs:
        if epochs in snapshot_epochs and failure is None:
            snaps[epochs] = [[W.detach().clone(), b.detach().clone()] for W, b in params]
        res["snapshots"] = snaps  # in-memory only (tensors); never written to JSON
    res["time_sec"] = time.time() - t0
    return res


def run_id(cfg):
    s = f"{cfg['dataset']}_{cfg['arch']}_{cfg['method']}_lr{cfg['lr']:g}_s{cfg['seed']}"
    if cfg.get("momentum", 0):
        s += f"_mom{cfg['momentum']:g}"
    if cfg.get("tau", 1.0) != 1.0:
        s += f"_tau{cfg['tau']:g}"
    if cfg.get("scale", 1.0) != 1.0:
        s += f"_scale{cfg['scale']:g}"
    if not cfg.get("standardize", True):
        s += "_raw"
    if cfg.get("epochs", 100) != 100:
        s += f"_ep{cfg['epochs']}"
    if cfg.get("batch", 128) != 128:
        s += f"_b{cfg['batch']}"
    return s


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)  # NaN is written as the (Python-readable) token NaN
    os.replace(tmp, path)
