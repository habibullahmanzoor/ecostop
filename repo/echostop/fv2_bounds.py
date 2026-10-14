"""Improvement 1: Hoeffding-Bentkus vs betting (WSR) certification, same rules, same folds.
Reports out-of-fold risk and savings, and validity over 1,000 random 70/30 splits."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.framework_v2 import crossfit, hb_p, load, select, wsr_pvalue

mats, names, dom = load()
strata = dom
RULES = ["EcoStop", "Patience (p=f*T, min-delta 0.1)", "Patience (fixed p)", "LC-PFN + EcoStop gates"]
P = {"Hoeffding-Bentkus (paper)": hb_p, "Betting (WSR)": wsr_pvalue}

rows = []
for rho in (0.10, 0.15, 0.20):
    for rule in RULES:
        loss = mats[rule][0]
        for pname, pf in P.items():
            ch = lambda tr, te, loss=loss, pf=pf: [(te, None if (i := select(loss[:, tr], rho, pf)) is None else (rule, i))]
            r = crossfit(mats, strata, rho, ch)
            rows.append({"rho": rho, "rule": rule, "bound": pname, "risk": r["risk"], "mean_saved": r["mean_saved"], "EW": r["EW"]})
df = pd.DataFrame(rows)
df.to_csv("figures/tierA/fv2_bounds.csv", index=False)
pd.set_option("display.width", 200, "display.max_colwidth", 34)
print(df.pivot_table(index=["rho", "rule"], columns="bound", values=["risk", "EW"], sort=False).round(3).to_string())

# validity: 1,000 random 70/30 splits, stratified by domain, certified on 70%, realised risk on 30%
rng = np.random.default_rng(1)
print("\nValidity over 1,000 random splits at rho=0.20 (fraction of splits whose held-out risk exceeds rho; must be <= 0.05):")
for rule in ("EcoStop", "Patience (p=f*T, min-delta 0.1)"):
    loss = mats[rule][0]
    for pname, pf in P.items():
        over, ew = 0, []
        for s in range(1000):
            te = np.zeros(len(dom), bool)
            for d in np.unique(dom):
                idx = np.where(dom == d)[0]
                te[rng.choice(idx, max(1, round(0.3 * len(idx))), replace=False)] = True
            i = select(loss[:, ~te], 0.20, pf)
            risk = 0.0 if i is None else loss[i, te].mean()
            over += risk > 0.20
        print(f"  {rule:34s} {pname:26s} exceed rate = {over / 1000:.3f}")
