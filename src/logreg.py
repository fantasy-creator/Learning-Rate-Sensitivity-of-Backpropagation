"""Classical reference: multinomial logistic regression on standardized features.

C is chosen on the validation split; test metrics are reported for that C only.
Output: results/logreg.json
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
for name in ("drybean", "optdigits", "har"):
    z = np.load(os.path.join(ROOT, "data", "processed", f"{name}.npz"), allow_pickle=False)
    s = lambda x: (x - z["mean"]) / z["std"]
    xtr, xva, xte = s(z["x_train"]), s(z["x_val"]), s(z["x_test"])
    best = None
    for C in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        clf = LogisticRegression(C=C, max_iter=10000).fit(xtr, z["y_train"])
        va = accuracy_score(z["y_val"], clf.predict(xva))
        if best is None or va > best[0]:
            best = (va, C, clf)
    va, C, clf = best
    pv, pt = clf.predict(xva), clf.predict(xte)
    out[name] = dict(C=C, val_acc=va, val_f1=f1_score(z["y_val"], pv, average="macro"),
                     test_acc=accuracy_score(z["y_test"], pt), test_f1=f1_score(z["y_test"], pt, average="macro"))
    print(name, out[name])
with open(os.path.join(ROOT, "results", "logreg.json"), "w") as f:
    json.dump(out, f, indent=1)
