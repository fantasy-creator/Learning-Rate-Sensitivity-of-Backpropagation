"""For each method and learning rate: in how many of the 9 dataset x architecture
combinations is that learning rate usable (seed-mean final val acc >= R - Delta)?
Output: results/universal_lr.json and a printed matrix.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(os.path.join(ROOT, "results", "summary_main.json")))
LRS = ["0.0001", "0.0003", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1", "3", "10", "30", "100"]
out = {}
for delta in (1.0, 2.0, 5.0):
    tab = {}
    for m in sorted({r["method"] for r in S}):
        rs = [r for r in S if r["method"] == m and r.get("scale", 1.0) == 1.0]
        tab[m] = {lr: sum(r["curve"].get(lr, 0.0) >= r["ref"] - delta / 100 for r in rs) for lr in LRS}
        tab[m]["n_combos"] = len(rs)
    out[f"delta{delta:g}"] = tab
    print(f"Delta = {delta:g} pp: number of combinations (of 9) in which each lr is usable")
    print("method        " + " ".join(f"{lr:>6s}" for lr in LRS))
    for m, row in tab.items():
        print(f"{m:13s} " + " ".join(f"{row[lr]:6d}" for lr in LRS))
json.dump(out, open(os.path.join(ROOT, "results", "universal_lr.json"), "w"), indent=1)
