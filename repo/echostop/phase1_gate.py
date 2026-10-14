"""Phase 1, step 4: (a) paired CI for the EcoStop k=1 gain at matched realised risk;
(b) a narrower schedule gate that blocks stopping only before a SUDDEN learning-rate drop
(>=50% epoch-to-epoch), leaving smooth cosine/one-cycle decay unaffected."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import phase1_prep as P
from echostop.phase1_eval import bootstrap_diff, run_method

SUDDEN = 0.5


def sudden_blocks(run):
    lr, T = run["lr"], run["T"]
    drop = np.zeros(T, bool)
    drop[1:] = lr[1:] <= SUDDEN * np.maximum(lr[:-1], 1e-12)
    for h in (2, 4, 6):
        b = np.zeros(T, bool)
        for t in range(T):
            b[t] = drop[t + 1: min(t + h, T - 1) + 1].any()
        run["block"][f"sd{h}"] = b
    run["block"]["sd0"] = np.zeros(T, bool)


def build_matrices():
    runs = P.load_runs()
    for r in runs:
        sudden_blocks(r)
    hs = ["sd0", "sd2", "sd4", "sd6"]
    P.FAMILIES.clear()
    P.FAMILIES["eco"] = [dict(g=g, nf=0.125, gf=0.125, k=k, h=h) for g in P.GAMMAS for k in (1, 2) for h in hs]
    P.FAMILIES["pat"] = [dict(f=f, md=0.1, h=h) for f in P.FRACS for h in hs]
    by_job = {}
    for r in runs:
        by_job.setdefault(r["job"], []).append(r)
    names = list(by_job)
    res = [P.sim_job(by_job[n]) for n in names]
    M = {"names": names, "domain": np.array([by_job[n][0]["domain"] for n in names]),
         "family": np.array([by_job[n][0]["family"] for n in names]), "et": np.array([r["et"] for r in res])}
    for fam in ("eco", "pat"):
        M[fam] = tuple(np.stack([r[fam][i] for r in res], axis=1) for i in range(4))
    return M


def idx_of(fam, **kw):
    return [i for i, c in enumerate(P.FAMILIES[fam]) if all(c[k] == v for k, v in kw.items())]


def main():
    M = build_matrices()
    strata = np.array([f"{d}|{f}" for d, f in zip(M["domain"], M["family"])])
    et = M["et"]
    rhos = [round(x, 2) for x in np.arange(0.08, 0.31, 0.02)]

    def curve(fam, idx, order_key):
        idx = sorted(idx, key=order_key)
        return {rho: run_method(M, fam, idx, "seq1d", rho, strata) for rho in rhos}

    ec = lambda i: P.FAMILIES["eco"][i]["g"]
    pt = lambda i: -P.FAMILIES["pat"][i]["f"]
    curves = {
        "Patience p=fT, min-delta": curve("pat", idx_of("pat", h="sd0"), pt),
        "EcoStop paper (k=2)": curve("eco", idx_of("eco", k=2, h="sd0"), ec),
        "EcoStop k=1": curve("eco", idx_of("eco", k=1, h="sd0"), ec),
        "EcoStop k=1 + sudden-drop gate h=2": curve("eco", idx_of("eco", k=1, h="sd2"), ec),
        "EcoStop k=1 + sudden-drop gate h=4": curve("eco", idx_of("eco", k=1, h="sd4"), ec),
        "Patience + sudden-drop gate h=4": curve("pat", idx_of("pat", h="sd4"), pt),
    }
    st, oc = M["family"] == "step", M["family"] == "onecycle"
    summ = {k: pd.DataFrame([{"rho": r, "risk": v["L"].mean(), "risk_step": v["L"][st].mean(), "risk_onecycle": v["L"][oc].mean(),
                              "EW": v["ES"].sum() / et.sum() * 100} for r, v in c.items()]) for k, c in curves.items()}
    # (a) matched realised risk: for each patience point, the EcoStop k=1 point with closest realised risk
    print("== (a) EcoStop k=1 vs calibrated patience at matched realised risk (paired job bootstrap, EW pp) ==")
    for rho_p in (0.14, 0.18, 0.22):
        pa = curves["Patience p=fT, min-delta"][rho_p]
        for name in ("EcoStop k=1", "EcoStop k=1 + sudden-drop gate h=4"):
            best = min(rhos, key=lambda r: abs(curves[name][r]["L"].mean() - pa["L"].mean()))
            d, lo, hi = bootstrap_diff(curves[name][best], pa, et)
            print(f"patience rho={rho_p} (risk {pa['L'].mean():.3f}, EW {pa['ES'].sum()/et.sum()*100:.1f}) | {name} rho={best} "
                  f"(risk {curves[name][best]['L'].mean():.3f}, EW {curves[name][best]['ES'].sum()/et.sum()*100:.1f}) | "
                  f"dEW {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    print()
    print("== (b) gate effect (rho target 0.20 / 0.14) ==")
    for name, s in summ.items():
        for rho in (0.14, 0.20):
            r = s[s.rho == rho].iloc[0]
            print(f"{name:40s} rho={rho:.2f} risk={r.risk:.3f} step={r.risk_step:.3f} onecycle={r.risk_onecycle:.3f} EW={r.EW:.1f}")
    pd.concat({k: v for k, v in summ.items()}).to_csv("figures/tierA/phase1_gate.csv")


if __name__ == "__main__":
    main()
