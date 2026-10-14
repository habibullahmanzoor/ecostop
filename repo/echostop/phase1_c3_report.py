"""Compare the nested learned score (C3) with calibrated patience and EcoStop k=1 at matched realised risk."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import phase1_gate as G
from echostop import phase1_prep as P
from echostop.phase1_c3 import RHOS
from echostop.phase1_eval import bootstrap_diff, run_method

d = pickle.load(open("cache/p1_c3.pkl", "rb"))
res, jobs, dom, fam, et = d["res"], d["jobs"], d["dom"], d["fam"], d["et"]
M = G.build_matrices()
assert list(M["names"]) == list(jobs), "job order mismatch"
strata = np.array([f"{a}|{b}" for a, b in zip(M["domain"], M["family"])])


def base_curve(fam_name, idx, key):
    idx = sorted(idx, key=key)
    return {rho: run_method(M, fam_name, idx, "seq1d", rho, strata) for rho in RHOS}


ec = lambda i: P.FAMILIES["eco"][i]["g"]
pt = lambda i: -P.FAMILIES["pat"][i]["f"]
curves = {
    "Patience p=fT, min-delta": base_curve("pat", G.idx_of("pat", h="sd0"), pt),
    "EcoStop k=1": base_curve("eco", G.idx_of("eco", k=1, h="sd0"), ec),
    "EcoStop k=1 + sudden gate h=4": base_curve("eco", G.idx_of("eco", k=1, h="sd4"), ec),
}
for (k, h), v in res.items():
    nm = f"EcoStop-L (learned) k={k}" + (" + sudden gate h=4" if h == "sd4" else "")
    curves[nm] = {rho: {"L": v["L"][i], "ES": v["ES"][i], "S": v["S"][i]} for i, rho in enumerate(RHOS)}

st, oc = fam == "step", fam == "onecycle"
rows = []
for name, c in curves.items():
    for rho in RHOS:
        r = c[rho]
        rows.append({"method": name, "rho": rho, "risk": r["L"].mean(), "risk_step": r["L"][st].mean(),
                     "risk_onecycle": r["L"][oc].mean(), "EW": r["ES"].sum() / et.sum() * 100})
df = pd.DataFrame(rows)
df.to_csv("figures/tierA/phase1_c3.csv", index=False)

pd.set_option("display.width", 220)
print("EW (energy-weighted %) at matched realised OOF risk (interpolated):")
out = []
for name, s in df.groupby("method", sort=False):
    s = s.sort_values("risk"); x, y = s.risk.values, s.EW.values
    out.append({"method": name, **{f"@{t:.2f}": (float(np.interp(t, x, y)) if x.min() <= t <= x.max() else np.nan) for t in (0.06, 0.08, 0.10, 0.12)}})
print(pd.DataFrame(out).round(1).to_string(index=False))

print("\nPaired job-bootstrap EW difference (pp) vs calibrated patience, at matched realised risk:")
for rho_p in (0.14, 0.18, 0.22):
    pa = curves["Patience p=fT, min-delta"][rho_p]
    for name in curves:
        if name.startswith("Patience"):
            continue
        best = min(RHOS, key=lambda r: abs(curves[name][r]["L"].mean() - pa["L"].mean()))
        dd, lo, hi = bootstrap_diff(curves[name][best], pa, et)
        print(f"patience risk {pa['L'].mean():.3f} | {name:34s} risk {curves[name][best]['L'].mean():.3f} | dEW {dd:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]")
print("\nStep / one-cycle risk at rho=0.20:")
print(df[df.rho == 0.20][["method", "risk", "risk_step", "risk_onecycle", "EW"]].round(3).to_string(index=False))
