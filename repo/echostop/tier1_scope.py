"""Tier 1(a): certificate scope card + abstention.

Scope = the support of the CALIBRATION population in four recipe descriptors, fixed a priori:
  (1) sudden LR drop (>= 50% epoch-to-epoch, before 90% of the horizon),
  (2) warm-up fraction (epochs until the LR first reaches its maximum / T),
  (3) horizon T (within a factor 2 of the calibration range of the job's domain),
  (4) domain.
A new job outside the scope abstains: it runs to completion (zero savings, zero risk).
Test: calibrate on the 142 development jobs only; apply to the 35 fresh B1 jobs with and without the check."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import phase1_prep as P
from echostop.tierA import select

SUDDEN = 0.5
T_FACTOR = 2.0


def descriptors(run):
    lr, T = run["lr"], run["T"]
    ratio = lr[1:] / np.maximum(lr[:-1], 1e-12)
    drops = (1 - ratio)[: int(0.9 * T) - 1]
    max_drop = float(drops.max()) if len(drops) else 0.0
    warm = int(np.argmax(lr >= 0.999 * lr.max())) / T
    return {"max_drop": max_drop, "warm": warm, "T": T, "domain": run["domain"]}


def main():
    runs = P.load_runs()
    by = {}
    for r in runs:
        by.setdefault(r["job"], []).append(r)
    names = list(by)
    desc = {n: descriptors(by[n][0]) for n in names}
    st = {n: by[n][0]["set"] for n in names}
    fam = {n: by[n][0]["family"] for n in names}
    cal = [n for n in names if st[n] == "dev"]
    tst = [n for n in names if st[n] == "b1"]

    # ---- scope from the calibration population
    sc = {"max_drop": max(desc[n]["max_drop"] for n in cal), "warm": max(desc[n]["warm"] for n in cal)}
    trange = {d: (min(desc[n]["T"] for n in cal if desc[n]["domain"] == d), max(desc[n]["T"] for n in cal if desc[n]["domain"] == d))
              for d in {desc[n]["domain"] for n in cal}}
    print("calibration support:", {k: round(v, 3) for k, v in sc.items()}, "| T ranges:", trange)

    def in_scope(n):
        d = desc[n]
        if d["domain"] not in trange:
            return False, "domain"
        if d["max_drop"] >= max(SUDDEN, sc["max_drop"] + 1e-9):
            return False, "sudden LR drop"
        if d["warm"] > sc["warm"] + 1e-9:
            return False, "warm-up fraction"
        lo, hi = trange[d["domain"]]
        if not (lo / T_FACTOR <= d["T"] <= hi * T_FACTOR):
            return False, "horizon"
        return True, ""

    flags = {n: in_scope(n) for n in tst}
    tab = pd.DataFrame([{"job": n, "family": fam[n], "domain": desc[n]["domain"], "T": desc[n]["T"],
                         "max_drop": round(desc[n]["max_drop"], 2), "warm": round(desc[n]["warm"], 2),
                         "in_scope": flags[n][0], "reason": flags[n][1]} for n in tst])
    pd.set_option("display.width", 200)
    print(tab.sort_values(["in_scope", "family"]).to_string(index=False))
    tab.to_csv("figures/tierA/tier1_scope_flags.csv", index=False)

    # ---- calibrate on dev only, apply to B1 with / without abstention
    M = P.load_mats()
    col = {n: i for i, n in enumerate(M["names"])}
    cal_i = np.array([col[n] for n in cal]); tst_i = np.array([col[n] for n in tst])
    eco, pat = P.FAMILIES["eco"], P.FAMILIES["pat"]
    sel = {
        "EcoStop paper (k=2)": ("eco", sorted([i for i, c in enumerate(eco) if c["nf"] == .125 and c["gf"] == .125 and c["k"] == 2 and c["h"] == 0], key=lambda i: eco[i]["g"])),
        "EcoStop k=1": ("eco", sorted([i for i, c in enumerate(eco) if c["nf"] == .125 and c["gf"] == .125 and c["k"] == 1 and c["h"] == 0], key=lambda i: eco[i]["g"])),
        "Patience p=fT, min-delta": ("pat", sorted([i for i, c in enumerate(pat) if c["md"] == .1 and c["h"] == 0], key=lambda i: -pat[i]["f"])),
    }
    ok = np.array([flags[n][0] for n in tst])
    rows = []
    for rho in (0.15, 0.20):
        for name, (fm, idx) in sel.items():
            loss, saved, es, gap = M[fm]
            i = select(loss[idx][:, cal_i], rho, 0.05)
            if i is None:
                continue
            c = idx[i]
            L, S, ES = loss[c, tst_i], saved[c, tst_i], es[c, tst_i]
            et = M["et"][tst_i]
            vis = M["domain"][tst_i] == "vision"
            for policy, mask in (("no scope check", np.ones_like(ok)), ("with scope check (abstain)", ok)):
                Lm, ESm, Sm = L * mask, ES * mask, S * mask
                rows.append({"rho": rho, "rule": name, "policy": policy, "risk": Lm.mean(), "risk_vision": Lm[vis].mean(),
                             "mean_saved": Sm.mean(), "EW": ESm.sum() / et.sum() * 100, "abstained_jobs": int((~mask.astype(bool)).sum())})
    df = pd.DataFrame(rows)
    df.to_csv("figures/tierA/tier1_scope_eval.csv", index=False)
    print(); print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
