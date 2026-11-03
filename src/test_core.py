"""Correctness checks for the update rules (run: python src/test_core.py)."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import core  # noqa: E402

torch.manual_seed(0)
f64 = torch.float64


def prox_by_normal_equations(W, b, A, G, eta, tau):
    """Minimise 0.5||U_hat A_hat^T - (W_hat A_hat^T - eta G^T)||^2 + 1/(2 tau)||U_hat - W_hat||^2."""
    Ah = torch.cat([A, torch.ones(A.shape[0], 1, dtype=f64)], 1)  # B x (d+1)
    Wh = torch.cat([W, b[:, None]], 1)
    target = Wh @ Ah.t() - eta * G.t()  # m x B
    lhs = Ah.t() @ Ah + torch.eye(Ah.shape[1], dtype=f64) / tau
    rhs = target @ Ah + Wh / tau
    U = torch.linalg.solve(lhs, rhs.t()).t()
    return U[:, :-1], U[:, -1]


def prox_by_svd(G, A, tau):
    """Mode-wise reference: sum_i sigma_i/(1/tau + sigma_i^2) (G^T u_i) v_i^T (no subtraction)."""
    Ah = torch.cat([A, torch.ones(A.shape[0], 1, dtype=f64)], 1)
    U, S, Vh = torch.linalg.svd(Ah, full_matrices=False)
    out = (G.t() @ U) * (S / (1 / tau + S ** 2)) @ Vh
    return out[:, :-1], out[:, -1]


def check(d, m, B, tau, scale=1.0):
    A = torch.randn(B, d, dtype=f64) * scale
    W = torch.randn(m, d, dtype=f64)
    b = torch.randn(m, dtype=f64)
    G = torch.randn(B, m, dtype=f64) / B
    eta = 0.37
    dW, db = core.prox_direction(G, A, tau)
    U, c = prox_by_normal_equations(W, b, A, G, eta, tau)
    sW, sb = prox_by_svd(G, A, tau)
    nrm = core.frob(sW, sb)
    err_ne = max(float((W - eta * dW - U).abs().max()), float((b - eta * db - c).abs().max()))
    err_svd = core.frob(dW - sW, db - sb) / nrm
    # optimality: gradient of the proximal objective vanishes at the closed form
    Ut = (W - eta * dW).clone().requires_grad_(True)
    ct = (b - eta * db).clone().requires_grad_(True)
    Z_tgt = A @ W.t() + b - eta * G
    Q = (0.5 * ((A @ Ut.t() + ct - Z_tgt) ** 2).sum() + ((Ut - W) ** 2).sum() / (2 * tau)
         + ((ct - b) ** 2).sum() / (2 * tau))
    gU, gc = torch.autograd.grad(Q, [Ut, ct])
    opt = max(float(gU.abs().max()), float(gc.abs().max())) / max(1.0, float(Z_tgt.abs().max()))
    print(f"d={d:4d} m={m:3d} B={B:4d} tau={tau:5.2f} scale={scale:6.1f} | vs normal-eq {err_ne:.1e} | "
          f"rel. vs SVD reference {err_svd:.1e} | KKT residual {opt:.1e} | "
          f"path={'chol(d+1)' if d + 1 <= B else 'push-through(B)'}")
    assert err_svd < 1e-8, err_svd
    assert opt < 1e-7, opt


for d, m, B in [(16, 32, 128), (64, 32, 128), (561, 64, 128), (300, 10, 50)]:
    for tau in (1.0, 0.1, 10.0):
        check(d, m, B, tau)
for scale in (10.0, 100.0):  # scaled inputs (input-scale stress test) must stay accurate
    check(561, 64, 128, 1.0, scale)
    check(64, 32, 128, 1.0, scale)

# dead layer: zero error signal gives a zero step without numerical trouble
dW, db = core.prox_direction(torch.zeros(128, 8, dtype=f64), torch.randn(128, 20, dtype=f64))
assert core.frob(dW, db) == 0.0

# full network: manual gradients equal autograd's, and prox steps are damped
D = core.load_data("optdigits")
params = core.init_mlp(D["d"], core.ARCHS["deep"], D["C"], seed=0)
flat = [p for pair in params for p in pair]
xb, yb = D["x_train"][:128], D["y_train"][:128]
loss, acts, Gs, grads = core.backprop(params, xb, yb)
logits, _, _ = core.forward(params, xb)
auto = torch.autograd.grad(F.cross_entropy(logits, yb), flat)
for l in range(len(params)):
    e = core.frob(grads[l][0] - auto[2 * l], grads[l][1] - auto[2 * l + 1]) / core.frob(*grads[l])
    assert e < 1e-12, e
for l in range(len(params) - 1):
    pW, pb = core.prox_direction(Gs[l], acts[l], 1.0)
    r = core.frob(pW, pb) / core.frob(*grads[l])
    print(f"layer {l}: ||prox step|| / ||BP step|| = {r:.4f}")
    assert r <= 1.0 + 1e-12

# determinism: identical seeds give identical runs (needed for re-running selected configs)
r1 = core.train_run("optdigits", "shallow", "prox_sgd", 1.0, seed=3, epochs=2, data=D)
r2 = core.train_run("optdigits", "shallow", "prox_sgd", 1.0, seed=3, epochs=2, data=D)
assert r1["history"]["val_acc"] == r2["history"]["val_acc"]
# failure path: absurd lr must be caught and scored by the constant predictor
r3 = core.train_run("optdigits", "shallow", "bp_sgd", 1e6, seed=0, epochs=2, data=D)
assert r3["diverged"] and r3["final_val_acc"] == r3["const_val_acc"], r3["failure"]
print("failure path:", r3["failure"])
print("ALL CHECKS PASSED")
