"""The constructive counterpart of the theory: certify per declared schedule family.

Compares three ways of choosing the stopping threshold on the same 259-job corpus:

  pooled        one threshold for everything (current practice, and the paper's baseline)
  per-domain    one per domain (vision / tabular / nlp) -- the standard Mondrian remedy
  per-family    one per declared schedule family (cosine / linear / step / one-cycle)

Calibration is always the 142 development jobs; evaluation is the 35 fresh jobs (which contain two
families with no calibration support) and the 82 confirmatory jobs. A stratum with no calibration
support abstains: no savings, no risk. Everything is a replay of cached matrices, so this is CPU-only
and reproducible with `python echostop/family_cert.py`.
"""
from __future__ import annotations

import collections
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BETA = 0.05
RHOS = (0.10, 0.15, 0.20)
RULES = ("EcoStop", "Patience (p=f*T, min-delta 0.1)")
ROOT = Path(__file__).resolve().parents[1]


def h1(a: float, b: float) -> float:
    a = min(max(a, 0.0), 1.0)
    if b <= 0.0 or b >= 1.0:
        return 0.0
    out = 0.0
    if a > 0:
        out += a * math.log(a / b)
    if a < 1:
        out += (1 - a) * math.log((1 - a) / (1 - b))
    return max(out, 0.0)


def pvalues(L: np.ndarray, rho: float) -> np.ndarray:
    """Hoeffding p-values for H0: mean(loss) >= rho, one per threshold row."""
    mean = L.mean(1)
    n = L.shape[1]
    return np.minimum(np.array([math.exp(-n * h1(min(m, rho), rho)) for m in mean]), 1.0)


def select(L: np.ndarray, rho: float):
    last = None
    for i, p in enumerate(pvalues(L, rho)):
        if p <= BETA:
            last = i
        else:
            break
    return last


def main() -> None:
    M, dom, grp, tag, et = pickle.load(open(ROOT / "cache/pool_mats.pkl", "rb"))
    tag = np.array([str(t) for t in tag])
    dom = np.array(dom)
    names = [str(g) for g in grp]

    def family(name: str, t: str, d: str) -> str:
        base = name.split(":", 1)[1]
        if d == "nlp":
            return "linear"
        if t == "b1" and d == "vision":
            if base.endswith("_step"):
                return "step"
            if "vgg_small" in base:
                return "onecycle"
        return "cosine"

    fam = np.array([family(g, t, d) for g, t, d in zip(names, tag, dom)])
    cal = np.where(tag == "dev")[0]
    strata = {"pooled": np.array(["all"] * len(names)), "per-domain": dom, "per-family": fam}

    rows = []
    for rule in RULES:
        L, ES = M[rule]["L"], M[rule]["ES"]
        for rho in RHOS:
            for strat_name, strat in strata.items():
                for ev_name in ("b1", "c1"):
                    ev = np.where(tag == ev_name)[0]
                    chosen, admitted = {}, np.zeros(len(ev), bool)
                    for s in sorted(set(strat[cal])):
                        cal_s = cal[strat[cal] == s]
                        i = select(L[:, cal_s], rho)
                        chosen[s] = i
                        if i is not None:
                            admitted |= (strat[ev] == s)
                    if not admitted.any():
                        rows.append(dict(rule=rule, rho=rho, strat=strat_name, eval=ev_name,
                                         risk=np.nan, coverage=0.0, EW_adm=np.nan, EW_pol=0.0,
                                         n_adm=0, strata_certified=sum(v is not None for v in chosen.values())))
                        continue
                    # per-job stop index where admitted; run-to-completion (last row) otherwise
                    stops = np.full(len(ev), L.shape[0] - 1, int)
                    for j_local, j_global in enumerate(ev):
                        s = strat[j_global]
                        if s in chosen and chosen[s] is not None:
                            stops[j_local] = chosen[s]
                    loss = L[stops, ev]
                    saved = ES[stops, ev].copy()
                    # an abstained job runs to completion: it saves nothing. Without this the
                    # whole-set EW credits abstained jobs with the most aggressive threshold.
                    saved[~admitted] = 0.0
                    rows.append(dict(
                        rule=rule, rho=rho, strat=strat_name, eval=ev_name,
                        risk=float(loss[admitted].mean()),
                        coverage=float(et[ev][admitted].sum() / et[ev].sum()),
                        EW_adm=float(100 * saved[admitted].sum() / et[ev][admitted].sum()),
                        EW_pol=float(100 * saved.sum() / et[ev].sum()),
                        n_adm=int(admitted.sum()),
                        strata_certified=sum(v is not None for v in chosen.values())))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "figures/tierA/family_cert.csv", index=False)

    pd.set_option("display.width", 200)
    for rule in RULES:
        print(f"\n=== {rule} ===")
        sub = df[df.rule == rule]
        piv = sub.pivot_table(index=["eval", "strat"], columns="rho",
                              values=["risk", "coverage", "EW_pol"], sort=False)
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
