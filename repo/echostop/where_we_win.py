"""Where does the curve rule beat calibrated patience, and where does it lose?

For each rule, pick the most aggressive threshold whose mean job-level loss on the development corpus
is at most rho, then compare the two rules on the fresh jobs, broken down by domain and by schedule
family. This is the per-stratum version of the matched-risk comparison in Q2; the paper reports only
the pooled version, and the regime table that answered this question was removed with appendix A.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RHO = 0.20
CURVE = "EcoStop"
PATIENCE = "Patience (p=f*T, min-delta 0.1)"


def pick(loss_cal: np.ndarray, rho: float) -> int:
    """Most aggressive threshold whose calibration mean loss is at most rho."""
    best = 0
    for i in range(loss_cal.shape[0]):
        if loss_cal[i].mean() <= rho:
            best = i
    return best


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
        return "continuous"

    fam = np.array([family(g, t, d) for g, t, d in zip(names, tag, dom)])
    cal = np.where(tag == "dev")[0]
    fresh = np.where(tag == "b1")[0]

    idx = {r: pick(M[r]["L"][:, cal], RHO) for r in (CURVE, PATIENCE)}
    print(f"rho={RHO}: threshold index curve={idx[CURVE]} patience={idx[PATIENCE]}\n")

    rows = []
    for label, mask in (("domain: vision", dom[fresh] == "vision"),
                        ("domain: tabular", dom[fresh] == "tabular"),
                        ("domain: nlp", dom[fresh] == "nlp"),
                        ("family: continuous", fam[fresh] == "continuous"),
                        ("family: step", fam[fresh] == "step"),
                        ("family: onecycle", fam[fresh] == "onecycle")):
        js = fresh[mask]
        if len(js) == 0:
            continue
        rec = {"stratum": label, "jobs": len(js)}
        for r, key in ((CURVE, "curve"), (PATIENCE, "pat")):
            i = idx[r]
            rec[f"{key}_risk"] = float(M[r]["L"][i, js].mean())
            rec[f"{key}_EW"] = float(100 * M[r]["ES"][i, js].sum() / et[js].sum())
        rec["dEW"] = rec["curve_EW"] - rec["pat_EW"]
        rec["drisk"] = rec["curve_risk"] - rec["pat_risk"]
        rows.append(rec)

    df = pd.DataFrame(rows)[["stratum", "jobs", "curve_risk", "pat_risk", "drisk",
                             "curve_EW", "pat_EW", "dEW"]]
    df.to_csv(ROOT / "figures/tierA/where_we_win.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))

    wins = df[df.dEW > 0]
    print(f"\nstrata where the curve rule saves more: {len(wins)} of {len(df)}")
    if len(wins):
        print("  " + ", ".join(f"{r.stratum} (+{r.dEW:.1f} pp)" for r in wins.itertuples()))


if __name__ == "__main__":
    main()
