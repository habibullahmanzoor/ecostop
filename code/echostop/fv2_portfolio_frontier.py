"""Portfolio vs single rules at MATCHED realised risk (frontier over rho), which rule the portfolio picks per domain,
and validity of the portfolio in the bootstrap world where the true risk is known."""
import itertools
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.framework_v2 import BETA, crossfit, load, select, wsr_pvalue
from echostop.fv2_portfolio import DOMS, composite

mats, names, dom = load()
et = mats["EcoStop"][3]
RHOS = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]
SINGLE = ["EcoStop", "Patience (p=f*T, min-delta 0.1)", "LC-PFN + EcoStop gates"]
CANDS = SINGLE
assigns = [dict(zip(DOMS, c)) for c in itertools.product(CANDS, repeat=3)]
comp = {i: composite(a) for i, a in enumerate(assigns)}
B_EACH = BETA / len(assigns)


def portfolio_pick(tr, rho, pf=wsr_pvalue):
    best, best_ew = None, -1.0
    for i in comp:
        L, S, ES, _ = comp[i]
        k = select(L[:, tr], rho, pf, beta=B_EACH)
        if k is not None:
            ew = ES[k, tr].sum() / et[tr].sum()
            if ew > best_ew:
                best, best_ew = (i, k), ew
    return best


curves = {}
for rho in RHOS:
    for rule in SINGLE:
        loss = mats[rule][0]
        curves[(rule, rho)] = crossfit(mats, dom, rho, lambda tr, te, loss=loss, rule=rule:
                                       [(te, None if (i := select(loss[:, tr], rho, wsr_pvalue)) is None else (rule, i))])
    curves[("Portfolio", rho)] = crossfit(comp, dom, rho, lambda tr, te: [(te, portfolio_pick(tr, rho))])

rows = [{"method": m, "rho": r, "risk": v["risk"], "EW": v["EW"], "mean_saved": v["mean_saved"]} for (m, r), v in curves.items()]
df = pd.DataFrame(rows)
df.to_csv("figures/tierA/fv2_portfolio_frontier.csv", index=False)
pd.set_option("display.width", 200)
print("EW at matched realised risk (interpolated on each method's own frontier):")
out = []
for m, s in df.groupby("method", sort=False):
    s = s.sort_values("risk")
    out.append({"method": m, **{f"@{t:.2f}": float(np.interp(t, s.risk, s.EW)) if s.risk.min() <= t <= s.risk.max() else np.nan
                                for t in (0.06, 0.08, 0.10, 0.12)}})
print(pd.DataFrame(out).round(1).to_string(index=False))

rng = np.random.default_rng(0)
print("\nPortfolio minus best single rule at matched realised risk (paired job bootstrap, energy-weighted pp):")
for rule in SINGLE:
    for rho_s in (0.14, 0.18, 0.22 if (rule, 0.22) in curves else 0.20):
        if (rule, rho_s) not in curves:
            continue
        b = curves[(rule, rho_s)]
        best = min(RHOS, key=lambda r: abs(curves[("Portfolio", r)]["risk"] - b["risk"]))
        a = curves[("Portfolio", best)]
        d = a["ES"] - b["ES"]
        idx = rng.integers(0, len(d), (5000, len(d)))
        bs = d[idx].sum(1) / et[idx].sum(1) * 100
        print(f"  vs {rule:32s} risk {b['risk']:.3f} (EW {b['EW']:.1f}) | portfolio risk {a['risk']:.3f} (EW {a['EW']:.1f}) | "
              f"dEW {d.sum() / et.sum() * 100:+6.2f} [{np.percentile(bs, 2.5):+6.2f}, {np.percentile(bs, 97.5):+6.2f}]")

print("\nRule chosen per domain on the full development set:")
for rho in (0.15, 0.20):
    p = portfolio_pick(np.arange(len(dom)), rho)
    print(f"  rho={rho:.2f}:", None if p is None else assigns[p[0]])

print("\nValidity in the bootstrap world (true risk known): P(true risk of the chosen composite > rho), 1,000 draws of n=142")
for rho in (0.15, 0.20):
    over = 0
    for s in range(1000):
        cal = rng.integers(0, len(dom), len(dom))
        p = portfolio_pick(cal, rho)
        if p is not None:
            over += comp[p[0]][0][p[1]].mean() > rho
    print(f"  rho={rho:.2f}: {over / 1000:.3f}  (must be <= 0.05)")
