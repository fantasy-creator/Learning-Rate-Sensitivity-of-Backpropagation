"""Dump per-mode BP / ProxProp responses for the first hidden layer at initialisation.

For one batch per dataset (seed-0 deep MLP, B=128) we compute the SVD of A_hat and,
per singular mode i, the BP response sigma_i ||G u_i|| and the ProxProp response
sigma_i/(1+sigma_i^2) ||G u_i|| (Proposition 2). Output: results/onestep/modes.json
"""
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import core  # noqa: E402

out = {}
for ds in core.DATASETS:
    D = core.load_data(ds)
    params = core.init_mlp(D["d"], core.ARCHS["deep"], D["C"], seed=0)
    idx = torch.randperm(D["x_train"].shape[0], generator=torch.Generator().manual_seed(123))[:128]
    _, acts, Gs, grads = core.backprop(params, D["x_train"][idx], D["y_train"][idx])
    for l in (0, 1):
        A, G = acts[l], Gs[l]
        Ah = torch.cat([A, torch.ones(A.shape[0], 1, dtype=A.dtype)], 1)
        U, S, Vh = torch.linalg.svd(Ah, full_matrices=False)
        keep = S > S.max() * max(Ah.shape) * torch.finfo(S.dtype).eps
        gu = (G.t() @ U).norm(dim=0)  # ||G u_i||
        out[f"{ds}_L{l}"] = dict(lam=(S[keep] ** 2).tolist(), bp=(S * gu)[keep].tolist(),
                                 prox=(S / (1 + S ** 2) * gu)[keep].tolist(),
                                 G_norm=float(G.norm()))
json.dump(out, open(os.path.join(core.ROOT, "results", "onestep", "modes.json"), "w"))
print("wrote modes.json", {k: len(v["lam"]) for k, v in out.items()})
