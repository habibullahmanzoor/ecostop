"""Framework fix: CONSENSUS stopping. Stop only when independent stopping evidence agrees.

Members (each with its own ordered grid; a shared aggressiveness index k maps to each grid by rank):
  E = EcoStop (curve forecast + gates, paper rule)
  P = calibrated-patience family (p proportional to T, min-delta 0.1)
  L = LC-PFN with EcoStop gates (pretrained learned extrapolator)
Combinations, fixed in advance (no data-driven choice):
  E&P  : stop at max(tau_E, tau_P)          (both must have fired)
  E&L  : stop at max(tau_E, tau_L)
  2of3 : stop at the median of the three   (any two agree)
  all3 : stop at max of the three
The shared index k is certified with the same fixed-sequence test (betting bound) as every single rule.
Development: out-of-fold. Fresh: calibrated on the 142 development jobs, applied unchanged to the 35 fresh jobs."""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E
from echostop import tierA
from echostop.ablation import sim
from echostop.calibration import forward_fill_audit, precompute_score_trace, simulate_from_trace
from echostop.framework_v2 import crossfit, select, wsr_pvalue
from echostop.lcpfn_real import PFN_GRID

EPS, NK = 0.5, 25
GAM = list(np.round(np.linspace(0.02, 0.98, 25), 2))
FR = [0.5, 0.4, 0.33, 0.3, 0.25, 0.2, 0.175, 0.15, 0.125, 0.1, 0.08, 0.06]
RHOS = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]


def stops_for(jobs, eco_tr, pfn_tr):
    """per job: list over seeds of dict member -> np.array of stop epochs on that member's NK-rank grid"""
    out = {}
    iE = np.round(np.linspace(0, len(GAM) - 1, NK)).astype(int)
    iP = np.round(np.linspace(0, len(FR) - 1, NK)).astype(int)
    iL = np.round(np.linspace(0, len(PFN_GRID) - 1, NK)).astype(int)
    for k, dfs in jobs.items():
        rs = []
        for s, df in enumerate(dfs):
            T = len(df)
            e = np.array([simulate_from_trace(eco_tr[k][s], GAM[i], T) for i in iE])
            p = np.array([tierA.patience_stop_md(df, max(1, round(FR[i] * T)), 0.1) for i in iP])
            l = np.array([sim(pfn_tr[k][s], PFN_GRID[i], T, max(2, round(0.125 * T)), 2) for i in iL])
            rs.append({"E": e, "P": p, "L": l, "a": forward_fill_audit(df), "e": df["cumulative_energy_joules"].values})
        out[k] = rs
    return out


COMBOS = {
    "EcoStop": lambda s: s["E"], "Patience p=fT": lambda s: s["P"], "LC-PFN": lambda s: s["L"],
    "Consensus E&P": lambda s: np.maximum(s["E"], s["P"]),
    "Consensus E&L": lambda s: np.maximum(s["E"], s["L"]),
    "Consensus 2-of-3": lambda s: np.median(np.vstack([s["E"], s["P"], s["L"]]), axis=0).astype(int),
    "Consensus all-3": lambda s: np.maximum(np.maximum(s["E"], s["P"]), s["L"]),
}


def matrices(stops, names):
    mats = {}
    for c, f in COMBOS.items():
        J = len(names)
        L, S, ES, ET = np.zeros((NK, J)), np.zeros((NK, J)), np.zeros((NK, J)), np.zeros(J)
        for j, k in enumerate(names):
            for s in stops[k]:
                t = f(s); a, e = s["a"], s["e"]
                L[:, j] += (a[-1] - a[t] > EPS)
                S[:, j] += 100 * (1 - e[t] / e[-1])
                ES[:, j] += e[-1] - e[t]
                ET[j] += e[-1]
            n = len(stops[k]); L[:, j] /= n; S[:, j] /= n
        mats[c] = (L, S, ES, ET)
    return mats


def main():
    cache = "cache/consensus_stops.pkl"
    if os.path.exists(cache):
        dev_st, dev_names, dev_dom, fr_st, fr_names, fr_dom = pickle.load(open(cache, "rb"))
    else:
        dev, dev_tr, dom = tierA.load()
        fresh, fdom = E.load_b1()
        fr_tr = {k: [precompute_score_trace(df, EPS) for df in dfs] for k, dfs in fresh.items()}
        dev_st = stops_for(dev, dev_tr, pickle.load(open("cache/lcpfn_traces.pkl", "rb")))
        fr_st = stops_for(fresh, fr_tr, pickle.load(open("cache/b1_lcpfn_traces.pkl", "rb")))
        dev_names, fr_names = list(dev), list(fresh)
        dev_dom = np.array([dom[k] for k in dev_names]); fr_dom = np.array([fdom[k] for k in fr_names])
        pickle.dump((dev_st, dev_names, dev_dom, fr_st, fr_names, fr_dom), open(cache, "wb"))
    D, F = matrices(dev_st, dev_names), matrices(fr_st, fr_names)
    etd, etf = D["EcoStop"][3], F["EcoStop"][3]

    dev_pts, fr_pts, dev_ES, fr_ES = {}, {}, {}, {}
    for c in COMBOS:
        L = D[c][0]
        for rho in RHOS:
            r = crossfit(D, dev_dom, rho, lambda tr, te, L=L, c=c, rho=rho: [(te, None if (i := select(L[:, tr], rho, wsr_pvalue)) is None else (c, i))])
            dev_pts.setdefault(c, []).append((r["risk"], r["EW"])); dev_ES[(c, rho)] = (r["risk"], r["ES"])
            i = select(L, rho, wsr_pvalue)
            Lf, Sf, ESf, _ = F[c]
            if i is None:
                fr_pts.setdefault(c, []).append((0.0, 0.0)); fr_ES[(c, rho)] = (0.0, np.zeros(len(fr_names)))
            else:
                fr_pts.setdefault(c, []).append((Lf[i].mean(), 100 * ESf[i].sum() / etf.sum())); fr_ES[(c, rho)] = (Lf[i].mean(), ESf[i])

    def interp(pts, t):
        p = sorted(pts); x = np.array([q[0] for q in p]); y = np.array([q[1] for q in p])
        return float(np.interp(t, x, y)) if x.min() <= t <= x.max() else np.nan

    rows = []
    for c in COMBOS:
        rows.append({"rule": c, **{f"dev@{t:.2f}": interp(dev_pts[c], t) for t in (0.06, 0.08, 0.10, 0.12)},
                     **{f"fresh@{t:.2f}": interp(fr_pts[c], t) for t in (0.06, 0.08, 0.10, 0.12, 0.15)}})
    df = pd.DataFrame(rows)
    df.to_csv("figures/tierA/fv2_consensus.csv", index=False)
    pd.set_option("display.width", 220)
    print("Energy-weighted saving (%) at matched realised risk. dev = out-of-fold; fresh = calibrated on dev, applied unchanged.")
    print(df.round(1).to_string(index=False))

    rng = np.random.default_rng(0)
    print("\nPaired job bootstrap at matched realised risk vs the strongest single rule (calibrated patience):")
    for setname, store, et in (("dev", dev_ES, etd), ("fresh", fr_ES, etf)):
        for c in ("Consensus E&P", "Consensus E&L", "Consensus 2-of-3", "Consensus all-3"):
            for rho_b in (0.12, 0.16, 0.20):
                rb, eb = store[("Patience p=fT", rho_b)]
                best = min(RHOS, key=lambda q: abs(store[(c, q)][0] - rb))
                ra, ea = store[(c, best)]
                d = ea - eb
                idx = rng.integers(0, len(d), (5000, len(d)))
                bs = d[idx].sum(1) / et[idx].sum(1) * 100
                print(f"  {setname:5s} {c:18s} risk {ra:.3f} vs patience {rb:.3f}: dEW {d.sum() / et.sum() * 100:+6.2f} "
                      f"[{np.percentile(bs, 2.5):+6.2f}, {np.percentile(bs, 97.5):+6.2f}]")


if __name__ == "__main__":
    main()
