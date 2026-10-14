"""A9: paired job-bootstrap CIs for matched-risk differences (out-of-fold, per-run loss)."""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import tierA

mats, names, dom = pickle.load(open("cache/tierA_mats_all.pkl", "rb"))
rng = np.random.default_rng(0)
A, B = "EcoStop", "Patience (p=f*T, min-delta 0.1)"
others = ["Patience (fixed p)", "LC-PFN + EcoStop gates", "Fixed budget fraction"]
rows = []
for rho in (0.15, 0.20):
    ra = tierA.crossfit(*mats[A], dom, rho, 0.05)
    for other in [B] + others:
        rb = tierA.crossfit(*mats[other], dom, rho, 0.05)
        for key, lab in (("job_saved", "mean_saved"), ("job_risk", "risk")):
            d = ra[key] - rb[key]
            idx = rng.integers(0, len(d), (5000, len(d)))
            bm = d[idx].mean(1)
            rows.append({"rho": rho, "EcoStop_minus": other, "metric": lab, "diff": d.mean(),
                         "ci_lo": np.percentile(bm, 2.5), "ci_hi": np.percentile(bm, 97.5)})
df = pd.DataFrame(rows); df.to_csv("figures/tierA/paired_ci.csv", index=False)
print(df.round(3).to_string(index=False))
