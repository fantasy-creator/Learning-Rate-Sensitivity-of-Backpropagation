"""One-step learning-rate sensitivity at fixed checkpoints and batches.

For every proximal (hidden) layer we form the BP and ProxProp increments for a
learning rate eta and a perturbed eta + delta from the *same* parameters and
batch, and compare the parameter differences with the bounds of the paper:

  BP   : ||W_BP(eta+delta) - W_BP(eta)||_F = |delta| ||g||_F                  (identity)
  Prox : ... <= |delta| ||g||_F / (1 + mu)          mu  = lambda_min(A_hat^T A_hat)
         ... <= |delta| ||g||_F / (1 + mu_plus)     mu+ = smallest positive eigenvalue
         ... <= |delta| ||G||_F / 2                 input-scale-free bound (tau = 1)

Increments are obtained by independent solves for eta and eta + delta of the
centred systems D (I + A_hat^T A_hat) = -eta g (production solver, core.prox_direction,
scaled by eta), cross-checked against (i) the mode-wise SVD formula and (ii) a
from-scratch solve of the full proximal problem. Rank/mu_plus come from the SVD of
A_hat with an explicit tolerance; the energy of g on the discarded directions is
recorded and a conservative mu_plus bound includes it. Everything is float64.
Exit status is non-zero if any bound is violated or any value is non-finite.
Output: results/onestep/records.json
"""
import json
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import core  # noqa: E402

ETAS = [0.01, 0.1, 1.0]
KAPPAS = [-0.5, -0.1, 0.1, 0.5]  # delta = kappa * eta (so eta + delta > 0)
TRAJ = {"bp_sgd": 0.03, "prox_sgd": 1.0}  # trajectories that provide checkpoints
CKPTS = [0, 1, 10, 50]
N_BATCH = 10
B = 128
RTOL = 1e-9  # relative slack allowed for floating-point round-off in bound checks
f64 = torch.float64


def full_prox_solve(W, b, A, G, eta):
    """From-scratch solve of argmin_U 0.5||U A_hat^T - (W_hat A_hat^T - eta G^T)||^2 + 0.5||U - W_hat||^2."""
    Ah = torch.cat([A, torch.ones(A.shape[0], 1, dtype=f64)], 1)
    Wh = torch.cat([W, b[:, None]], 1)
    lhs = Ah.t() @ Ah + torch.eye(Ah.shape[1], dtype=f64)
    rhs = (Wh @ Ah.t() - eta * G.t()) @ Ah + Wh
    return torch.linalg.solve(lhs, rhs.t()).t() - Wh


def layer_records(params, xb, yb):
    _, acts, Gs, grads = core.backprop(params, xb, yb)
    out = []
    for l in range(len(params) - 1):
        A, G = acts[l], Gs[l]  # B x d, B x m
        W, b = params[l][0].detach(), params[l][1].detach()
        Ah = torch.cat([A, torch.ones(A.shape[0], 1, dtype=f64)], 1)  # B x (d+1)
        g = torch.cat([grads[l][0], grads[l][1][:, None]], 1)  # m x (d+1) = G^T A_hat
        dim = Ah.shape[1]
        U, S, Vh = torch.linalg.svd(Ah, full_matrices=False)  # S: min(B, d+1) values
        tol = float(S.max()) * max(Ah.shape) * torch.finfo(f64).eps if S.numel() else 0.0
        keep = S > tol
        rank = int(keep.sum())
        lam = S ** 2  # nonzero eigenvalues of A_hat^T A_hat (others are exactly 0)
        mu = float(lam.min()) if (rank == dim) else 0.0
        mu_plus = float(lam[keep].min()) if rank > 0 else float("nan")
        gn, Gn = core.frob(g), core.frob(G)
        rec = dict(layer=l, d=int(A.shape[1]), m=int(W.shape[0]), dim=dim, rank=rank, tol=tol,
                   mu=mu, mu_plus=mu_plus, lam_max=float(lam.max()), g_norm=gn, G_norm=Gn,
                   dead_layer=gn == 0.0)
        if gn == 0.0:  # dead layer: both responses must vanish exactly
            pW, pb = core.prox_direction(G, A)
            rec.update(zero_response=core.frob(pW, pb) == 0.0, ok=core.frob(pW, pb) == 0.0)
            out.append(rec)
            continue
        Vk = Vh[keep].t()  # (d+1) x rank, kept right singular vectors
        g_r = (g @ Vk) @ Vk.t()
        gd2 = max(gn ** 2 - core.frob(g_r) ** 2, 0.0)  # energy off the kept directions
        # mode-wise exact response per unit delta (no subtraction of large numbers)
        modewise = (G.t() @ U) * (S / (1.0 + S ** 2)) @ Vh
        mw = core.frob(modewise)
        pW, pb = core.prox_direction(G, A)
        prod = torch.cat([pW, pb[:, None]], 1)
        dirn = core.frob(prod)
        b1, b2 = 1.0 / (1.0 + mu), 1.0 / (1.0 + mu_plus)
        b2_cons = math.sqrt(core.frob(g_r) ** 2 / (1.0 + mu_plus) ** 2 + gd2) / gn
        b3 = Gn / (2.0 * gn)
        ratios, worst = [], {"b1": -1.0, "b2_conservative": -1.0, "b3": -1.0}
        bp_err, lin_err, full_err = 0.0, 0.0, 0.0
        for eta in ETAS:
            D0 = eta * prod  # independent production solves for eta and eta + delta
            full0 = full_prox_solve(W, b, A, G, eta)
            full_err = max(full_err, core.frob(full0 + D0) / max(core.frob(D0), 1e-300))
            for kap in KAPPAS:
                dlt = kap * eta
                qW, qb = core.prox_direction(G, A)
                D1 = (eta + dlt) * torch.cat([qW, qb[:, None]], 1)
                dprox = core.frob(D1 - D0)
                dbp = core.frob((eta + dlt) * g - eta * g)
                vals = [dprox, dbp]
                if not all(math.isfinite(v) for v in vals):
                    raise RuntimeError(f"non-finite response in layer {l}")
                bp_err = max(bp_err, abs(dbp - abs(dlt) * gn) / (abs(dlt) * gn))
                lin_err = max(lin_err, abs(dprox - abs(dlt) * mw) / (abs(dlt) * mw) if mw > 0 else 0.0)
                ratios.append(dprox / dbp)
                for key, fac in (("b1", b1), ("b2_conservative", b2_cons), ("b3", b3)):
                    worst[key] = max(worst[key], dprox / (abs(dlt) * gn * fac) - 1.0)
        ok = all(v <= RTOL for v in worst.values()) and bp_err <= 1e-12 and lin_err <= 1e-9
        g_top = float(((g @ Vh[keep][: max(1, math.ceil(0.1 * rank))].t()) ** 2).sum()) / gn ** 2
        rec.update(rho=float(np.mean(ratios)), rho_spread=float(np.ptp(ratios)), rho_modewise=mw / gn,
                   prod_vs_modewise=abs(dirn - mw) / mw, bound1=b1, bound2=b2,
                   bound2_conservative=b2_cons, bound3=b3, discarded_energy=gd2 / gn ** 2,
                   bp_identity_relerr=bp_err, linearity_relerr=lin_err, full_solve_relerr=full_err,
                   max_rel_slack=worst, energy_top10pct_modes=g_top, ok=ok)
        out.append(rec)
    return out


def main():
    out_dir = os.path.join(core.ROOT, "results", "onestep")
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(2)
    records, missing = [], []
    for ds in core.DATASETS:
        D = core.load_data(ds)
        for arch in core.ARCHS:
            for traj, lr in TRAJ.items():
                res = core.train_run(ds, arch, traj, lr, seed=0, epochs=max(CKPTS), data=D,
                                     snapshot_epochs=CKPTS)
                want = [e for e in CKPTS if not (traj != "bp_sgd" and e == 0)]
                missing += [dict(dataset=ds, arch=arch, traj=traj, epoch=e) for e in want
                            if e not in res["snapshots"]]
                for ep in want:
                    if ep not in res["snapshots"]:
                        continue
                    params = [[W.clone().requires_grad_(True), b.clone().requires_grad_(True)]
                              for W, b in res["snapshots"][ep]]
                    gen = torch.Generator().manual_seed(123)
                    for k in range(N_BATCH):
                        idx = torch.randperm(D["x_train"].shape[0], generator=gen)[:B]
                        for r in layer_records(params, D["x_train"][idx], D["y_train"][idx]):
                            r.update(dataset=ds, arch=arch, traj=traj, traj_lr=lr, epoch=ep, batch=k)
                            records.append(r)
                print(f"{ds:9s} {arch:7s} {traj:8s} done ({len(records)} records, "
                      f"diverged={res['diverged']})", flush=True)
    with open(os.path.join(out_dir, "records.json"), "w") as f:
        json.dump(dict(records=records, missing_checkpoints=missing, rtol=RTOL), f)
    live = [r for r in records if not r["dead_layer"]]
    bad = [r for r in records if not r["ok"]]
    print(f"{len(records)} layer records ({len(records) - len(live)} dead layers), "
          f"{len(missing)} missing checkpoints, {len(bad)} failed checks")
    for key in ("b1", "b2_conservative", "b3"):
        print(f"  max relative slack {key}: {max(r['max_rel_slack'][key] for r in live):.3e}")
    for key in ("bp_identity_relerr", "linearity_relerr", "prod_vs_modewise", "full_solve_relerr",
                "discarded_energy", "rho_spread"):
        print(f"  max {key}: {max(r[key] for r in live):.3e}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
