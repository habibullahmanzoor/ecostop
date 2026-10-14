"""Scope-card abstention applied to the PAPER's frozen thresholds (b1_protocol.json), so the numbers are
directly comparable with Table b1_fresh. Scope flags come from tier1_scope.py (descriptors of the 142-job
calibration population)."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E
from echostop.calibration import precompute_score_trace

proto = json.load(open("b1_protocol.json"))
E.check_hashes(proto)
jobs, dom = E.load_b1()
names = list(jobs)
flags = pd.read_csv("figures/tierA/tier1_scope_flags.csv").set_index("job")
ok = np.array([bool(flags.loc["b1:" + n, "in_scope"]) for n in names])
traces = {k: [precompute_score_trace(df, 0.5) for df in dfs] for k, dfs in jobs.items()}
et = np.array([sum(df["cumulative_energy_joules"].iloc[-1] for df in jobs[k]) for k in names])
vis = np.array([dom[k] == "vision" for k in names])
rows = []
for tkey, fams in proto["targets"].items():
    rho = float(tkey.split(",")[0].split("=")[1])
    for fam in ("EcoStop", "Patience (p=f*T, min-delta 0.1)", "Patience (fixed p)"):
        lab = fams[fam]["pooled"]
        L, S, ES = [], [], []
        fn = E.stop_fn(fam, lab)
        for k in names:
            l, s, _ = E.score({k: jobs[k]}, {k: traces[k]}, fn)
            es = sum(df["cumulative_energy_joules"].iloc[-1] - df["cumulative_energy_joules"].iloc[fn(df, tr)]
                     for df, tr in zip(jobs[k], traces[k]))
            L.append(l[0]); S.append(s[0]); ES.append(es)
        L, S, ES = map(np.array, (L, S, ES))
        for policy, mask in (("certificate applied to every job", np.ones(len(names), bool)), ("scope card: abstain outside scope", ok)):
            rows.append({"rho": rho, "rule": fam, "param": lab, "policy": policy, "risk": (L * mask).mean(),
                         "risk_vision": (L * mask)[vis].mean(), "risk_in_scope": L[ok].mean(),
                         "mean_saved": (S * mask).mean(), "EW": 100 * (ES * mask).sum() / et.sum(), "abstained": int((~mask).sum())})
df = pd.DataFrame(rows)
df.to_csv("figures/tierA/tier1_scope_frozen.csv", index=False)
pd.set_option("display.width", 220)
print(f"{len(names)} fresh jobs; in scope: {int(ok.sum())}, abstain: {int((~ok).sum())}")
print(df.round(3).to_string(index=False))
