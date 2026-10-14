"""Out-of-sample test of the certified portfolio: calibrate on the 142 development jobs ONLY (assignment + aggressiveness
for the portfolio, threshold for each single rule; betting bound, beta = 0.05), then apply everything unchanged to the
35 fresh jobs, which played no part in designing or calibrating the portfolio."""
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E
from echostop import tierA
from echostop.ablation import sim
from echostop.calibration import precompute_score_trace
from echostop.framework_v2 import BETA, load, select, wsr_pvalue
from echostop.fv2_portfolio import DOMS, NK
from echostop.lcpfn_real import PFN_GRID

SINGLE = ["EcoStop", "Patience (p=f*T, min-delta 0.1)", "LC-PFN + EcoStop gates"]

# ---------------- development matrices (cached) and fresh matrices with the SAME parameter grids
dev_mats, dev_names, dev_dom = load()
fresh, fdom = E.load_b1()
fnames = list(fresh)
fdom_arr = np.array([fdom[k] for k in fnames])
tr_f = {k: [precompute_score_trace(df, 0.5) for df in dfs] for k, dfs in fresh.items()}
pf_tr = pickle.load(open("cache/b1_lcpfn_traces.pkl", "rb"))
pol = tierA.build_policies(None)
pfn_pl = [(f"g={g}", (lambda df, t, g=g: sim(t, g, len(df), max(2, round(0.125 * len(df))), 2))) for g in PFN_GRID]
fresh_mats = {"EcoStop": tierA.evaluate_family(fresh, tr_f, pol["EcoStop"]),
              "Patience (p=f*T, min-delta 0.1)": tierA.evaluate_family(fresh, tr_f, pol["Patience (p=f*T, min-delta 0.1)"]),
              "LC-PFN + EcoStop gates": tierA.evaluate_family(fresh, pf_tr, pfn_pl)}
for r in SINGLE:
    assert dev_mats[r][0].shape[0] == fresh_mats[r][0].shape[0], r


def composite(mats, dom, assign):
    J = len(dom)
    L, S, ES = np.zeros((NK, J)), np.zeros((NK, J)), np.zeros((NK, J))
    for d in DOMS:
        loss, saved, es, _ = mats[assign[d]]
        idx = np.round(np.linspace(0, loss.shape[0] - 1, NK)).astype(int)
        m = dom == d
        L[:, m], S[:, m], ES[:, m] = loss[idx][:, m], saved[idx][:, m], es[idx][:, m]
    return L, S, ES


assigns = [dict(zip(DOMS, c)) for c in itertools.product(SINGLE, repeat=3)]
dev_comp = [composite(dev_mats, dev_dom, a) for a in assigns]
fr_comp = [composite(fresh_mats, fdom_arr, a) for a in assigns]
et_dev, et_fr = dev_mats["EcoStop"][3], fresh_mats["EcoStop"][3]

rows, keep = [], {}
for rho in (0.15, 0.20):
    # portfolio: choose on dev only
    best, best_ew = None, -1
    for i, (L, S, ES) in enumerate(dev_comp):
        k = select(L, rho, wsr_pvalue, beta=BETA / len(assigns))
        if k is not None and ES[k].sum() / et_dev.sum() > best_ew:
            best, best_ew = (i, k), ES[k].sum() / et_dev.sum()
    i, k = best
    L, S, ES = fr_comp[i]
    keep[("Portfolio", rho)] = ES[k]
    rows.append({"rho": rho, "method": "Portfolio " + str({d: assigns[i][d].split(" (")[0] for d in DOMS}), "fresh_risk": L[k].mean(),
                 "fresh_vision_risk": L[k][fdom_arr == "vision"].mean(), "fresh_mean_saved": S[k].mean(), "fresh_EW": 100 * ES[k].sum() / et_fr.sum()})
    for r in SINGLE:
        j = select(dev_mats[r][0], rho, wsr_pvalue)
        Lf, Sf, ESf, _ = fresh_mats[r]
        keep[(r, rho)] = ESf[j] if j is not None else np.zeros(len(fnames))
        rows.append({"rho": rho, "method": r, "fresh_risk": Lf[j].mean() if j is not None else 0.0,
                     "fresh_vision_risk": Lf[j][fdom_arr == "vision"].mean() if j is not None else 0.0,
                     "fresh_mean_saved": Sf[j].mean() if j is not None else 0.0,
                     "fresh_EW": 100 * ESf[j].sum() / et_fr.sum() if j is not None else 0.0})
df = pd.DataFrame(rows)
df.to_csv("figures/tierA/fv2_portfolio_fresh.csv", index=False)
pd.set_option("display.width", 220, "display.max_colwidth", 90)
print(df.round(3).to_string(index=False))
rng = np.random.default_rng(0)
print("\nFresh jobs: portfolio minus single rule, energy-weighted pp at the same frozen target (paired job bootstrap):")
for rho in (0.15, 0.20):
    for r in SINGLE:
        d = keep[("Portfolio", rho)] - keep[(r, rho)]
        idx = rng.integers(0, len(d), (5000, len(d)))
        bs = d[idx].sum(1) / et_fr[idx].sum(1) * 100
        print(f"  rho={rho:.2f} vs {r:32s}: {d.sum() / et_fr.sum() * 100:+6.2f} [{np.percentile(bs, 2.5):+6.2f}, {np.percentile(bs, 97.5):+6.2f}]")
