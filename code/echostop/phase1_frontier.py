"""Phase 1, step 3: matched-realised-risk frontier for single-parameter (seq1d) rules, and the schedule
gate forced on (not optimised). Answers: at equal realised risk, which rule saves most energy, and does
a fixed schedule gate repair the step/one-cycle failures?"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.phase1_eval import eco_idx, pat_idx, run_method
from echostop.phase1_prep import FAMILIES, load_mats

RHOS = (0.10, 0.15, 0.20, 0.25, 0.30)


def variants():
    v = {}
    for h in (0, 2, 4, 6):
        sfx = "" if h == 0 else f" +gate h={h}"
        v[f"Patience p=fT, min-delta{sfx}"] = ("pat", sorted(pat_idx(md=0.1, h=h), key=lambda i: -FAMILIES["pat"][i]["f"]))
        v[f"EcoStop paper (gate, k=2){sfx}"] = ("eco", sorted(eco_idx(nf=0.125, gf=0.125, k=2, h=h), key=lambda i: FAMILIES["eco"][i]["g"]))
        v[f"EcoStop k=1{sfx}"] = ("eco", sorted(eco_idx(nf=0.125, gf=0.125, k=1, h=h), key=lambda i: FAMILIES["eco"][i]["g"]))
        v[f"EcoStop no flat-gate{sfx}"] = ("eco", sorted(eco_idx(nf=0.125, gf=0.0, k=2, h=h), key=lambda i: FAMILIES["eco"][i]["g"]))
        v[f"EcoStop no flat-gate, k=1{sfx}"] = ("eco", sorted(eco_idx(nf=0.125, gf=0.0, k=1, h=h), key=lambda i: FAMILIES["eco"][i]["g"]))
    return v


def main():
    M = load_mats()
    strata = np.array([f"{d}|{f}" for d, f in zip(M["domain"], M["family"])])
    et = M["et"]
    rows = []
    for name, (fam, idx) in variants().items():
        for rho in RHOS:
            r = run_method(M, fam, idx, "seq1d", rho, strata)
            g = lambda m: r["L"][m].mean() if m.any() else np.nan
            rows.append({"method": name, "rho": rho, "risk": r["L"].mean(), "risk_step": g(M["family"] == "step"),
                         "risk_onecycle": g(M["family"] == "onecycle"), "risk_vision": g(M["domain"] == "vision"),
                         "mean_saved": r["S"].mean(), "EW": r["ES"].sum() / et.sum() * 100})
    d = pd.DataFrame(rows)
    d.to_csv("figures/tierA/phase1_frontier.csv", index=False)

    # EW at matched realised risk (linear interpolation along each rule's own frontier)
    out = []
    for name, sub in d.groupby("method", sort=False):
        s = sub.sort_values("risk")
        x, y = s.risk.values, s.EW.values
        row = {"method": name}
        for tgt in (0.06, 0.08, 0.10, 0.12):
            row[f"EW@risk{tgt:.2f}"] = float(np.interp(tgt, x, y)) if x.min() <= tgt <= x.max() else np.nan
        out.append(row)
    m = pd.DataFrame(out)
    pd.set_option("display.width", 250, "display.max_colwidth", 48)
    print("EW (energy-weighted %) at matched realised OOF risk:")
    print(m.round(1).to_string(index=False))
    print()
    print("Schedule gate effect on step / one-cycle vision risk at rho=0.20:")
    sub = d[d.rho == 0.20][["method", "risk", "risk_step", "risk_onecycle", "risk_vision", "EW"]]
    print(sub.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
