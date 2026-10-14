"""P2: is the cost of per-domain certification intrinsic or a sample-size effect? For n calibration jobs per domain, compare
(a) per-domain Hoeffding-Bentkus certification (n jobs each) with (b) pooled certification on the 3n jobs, scoring BOTH on the same
held-out jobs of each domain. 300 random splits per cell. Reports mean domain energy-weighted saving, share of splits certifying
nothing, and share of splits whose held-out domain risk exceeds rho."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.pool_loader import load_pool
from echostop.c1_framework import RULES, SHORT
from echostop.framework_v2 import hb_p, select

M, dom, grp, tag, et = load_pool()
DOMS = ["vision", "nlp", "tabular"]
NS = [8, 12, 16, 24, 28]
rng = np.random.default_rng(0)
rows = []
for rule in (RULES[1], RULES[0]):
    L, ES = M[rule]["L"], M[rule]["ES"]
    for rho in (0.15, 0.20):
        for n in NS:
            acc = {(d, m): {"ew": [], "none": 0, "exceed": 0, "n": 0} for d in DOMS for m in ("per-domain", "pooled")}
            for _ in range(300):
                cal, test = {}, {}
                for d in DOMS:
                    idx = rng.permutation(np.where(dom == d)[0])
                    cal[d], test[d] = idx[:n], idx[n:]
                pool_idx = np.concatenate([cal[d] for d in DOMS])
                ip = select(L[:, pool_idx], rho, hb_p)
                for d in DOMS:
                    if len(test[d]) < 5:
                        continue
                    for m, i in (("pooled", ip), ("per-domain", select(L[:, cal[d]], rho, hb_p))):
                        a = acc[(d, m)]; a["n"] += 1
                        if i is None:
                            a["none"] += 1; a["ew"].append(0.0)
                        else:
                            a["ew"].append(100 * ES[i, test[d]].sum() / et[test[d]].sum())
                            a["exceed"] += L[i, test[d]].mean() > rho
            for (d, m), a in acc.items():
                if a["n"]:
                    rows.append(dict(rule=SHORT[rule], rho=rho, n_per_domain=n, domain=d, method=m, EW=np.mean(a["ew"]),
                                     uncertified=a["none"] / a["n"], exceed=a["exceed"] / a["n"]))
df = pd.DataFrame(rows)
df.to_csv("figures/tierA/pool_vs_domain.csv", index=False)
pd.set_option("display.width", 200)
for rule in df.rule.unique():
    for rho in (0.15, 0.20):
        x = df[(df.rule == rule) & (df.rho == rho)]
        print(f"\n{rule}, rho={rho}: mean domain EW (%) [share of splits certifying nothing]")
        p = x.pivot_table(index=["domain", "method"], columns="n_per_domain", values="EW").round(1)
        u = x.pivot_table(index=["domain", "method"], columns="n_per_domain", values="uncertified").round(2)
        print(p.astype(str).add(" [" + u.astype(str) + "]").to_string())
