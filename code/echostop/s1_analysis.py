"""S1 analysis (certified early cooldown). Hash-locked by s1_freeze.py before test jobs are trained.

COMMON REFERENCE for every method: the standard full run, i.e. the same seed's recipe-matched 40-epoch cosine run
(A_ref = test accuracy of its validation-best checkpoint). Loss of a run = 1[A_ref - A_method > eps], eps = 0.5 pp;
loss of a job = fraction of violating seeds. Cost = epochs trained (energy readings are contaminated whenever other
GPU jobs run; they are reported separately, never used for the claim).

Methods (each has ONE ordered parameter, conservative -> aggressive, certified with the same fixed-sequence
Hoeffding-Bentkus test):
  adaptive cooldown   : cooldown (8 epochs) at the first checkpoint t whose EcoStop score s_t < gamma, else the reference branch (t=32)
  fixed-trigger cooldown (budgeted-training analogue): cooldown at the same checkpoint t for every job
  stop-only EcoStop   : stop the cosine run when the paper's EcoStop rule fires (gamma)
  stop-only patience  : stop the cosine run with patience p = round(f*T), min-delta 0.1
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ecostop_bench"))
import bench
from echostop import s1_registry as R

EPS = 0.5
T_FULL = R.T_COSINE


def _val_best_test(val, test):
    a = np.empty(len(val)); best, cur = -1.0, np.nan
    for i in range(len(val)):
        if val[i] >= best:
            best, cur = val[i], test[i]
        a[i] = cur
    return a


def load_cool(path):
    df = pd.read_csv(path)
    st = df[df.phase == "stable"].sort_values("epoch")
    br = {}
    for t, g in df[df.phase == "branch"].groupby("branch_t"):
        g = g.sort_values("epoch")
        v = g.val_acc.values
        k = len(v) - 1 - int(np.argmax(v[::-1]))
        br[int(t)] = {"acc": float(g.test_acc.values[k]), "e": float(g.energy_joules.sum())}
    return {"val": st.val_acc.values.astype(float), "e": st.energy_joules.values.astype(float), "branch": br}


def load_cosine(path):
    c = pd.read_csv(path).sort_values("epoch")
    q, tst = c.val_acc.values.astype(float), c.test_acc.values.astype(float)
    return {"T": len(q), "q": q, "b": np.maximum.accumulate(q), "a": _val_best_test(q, tst),
            "e": np.cumsum(c.energy_joules.values.astype(float))}


def _paths():
    """job_id -> {seed: (cool_csv, cosine_csv)} from the pilot directories and logs/s1/{cal,test}."""
    out = {"cal": {}, "test": {}}
    for p in sorted(glob.glob("logs/pilot_cool/*_seed*.csv")):
        if not os.path.exists(p.replace(".csv", ".meta.json")):
            continue
        ds, sd = os.path.basename(p)[:-4].rsplit("_seed", 1)
        pc = f"logs/pilot_cosine/{ds}_seed{sd}.csv"
        if os.path.exists(pc.replace(".csv", ".meta.json")):
            out["cal"].setdefault(f"{ds}_resnet9_lr0.05", {})[int(sd)] = (p, pc)
    for part in ("cal", "test"):
        for p in sorted(glob.glob(f"logs/s1/{part}/*_cool.csv")):
            if not os.path.exists(p.replace(".csv", ".meta.json")):
                continue
            base = os.path.basename(p)[: -len("_cool.csv")]
            job, sd = base.rsplit("_seed", 1)
            pc = p.replace("_cool.csv", "_cosine.csv")
            if os.path.exists(pc.replace(".csv", ".meta.json")):
                out[part].setdefault(job, {})[int(sd)] = (p, pc)
    return out


def collect(part):
    jobs = {}
    for job, seeds in _paths()[part].items():
        jobs[job] = []
        for sd, (pcool, pcos) in sorted(seeds.items()):
            cool, cos = load_cool(pcool), load_cosine(pcos)
            run = {"T": 32, "q": cool["val"][:32], "b": np.maximum.accumulate(cool["val"][:32])}
            cool["s"] = bench.ecostop_scores(run)
            jobs[job].append({"seed": sd, "cool": cool, "cos": cos})
    return jobs


# ---------------------------------------------------------------- per-seed outcomes -> per-job loss / saving
def cooldown_outcome(rec, t):
    acc = rec["cool"]["branch"][t]["acc"]
    return acc, (t + R.C)


def adaptive_trigger(rec, gamma):
    s = rec["cool"]["s"]
    for t in R.CKPTS[:-1]:
        if s[t - 1] == s[t - 1] and s[t - 1] < gamma:
            return t
    return 32


METHODS = {   # name -> list of thetas, conservative -> aggressive
    "adaptive cooldown": [round(x, 2) for x in np.linspace(0.02, 0.98, 25)],
    "fixed-trigger cooldown": sorted(R.CKPTS, reverse=True),
    "stop-only EcoStop": bench.RULES["ecostop"][1],
    "stop-only patience": bench.RULES["patience_fraction"][1],
}


def seed_result(method, theta, rec):
    ref = rec["cos"]["a"][-1]
    if method == "adaptive cooldown":
        t = adaptive_trigger(rec, theta); acc, cost = cooldown_outcome(rec, t)
    elif method == "fixed-trigger cooldown":
        acc, cost = cooldown_outcome(rec, int(theta))
    else:
        run = {"T": rec["cos"]["T"], "q": rec["cos"]["q"], "b": rec["cos"]["b"], "a": rec["cos"]["a"], "e": rec["cos"]["e"]}
        t = bench.ecostop(run, theta, k=2) if method == "stop-only EcoStop" else bench.patience_fraction(run, theta)
        acc, cost = rec["cos"]["a"][t], t + 1
    return float(ref - acc > EPS), 100 * (1 - cost / T_FULL), ref - acc


def method_tables(jobs):
    names = sorted(jobs)
    tab = {}
    for m, thetas in METHODS.items():
        L = np.zeros((len(thetas), len(names))); S = np.zeros_like(L); G = np.zeros_like(L)
        for j, name in enumerate(names):
            for i, th in enumerate(thetas):
                r = [seed_result(m, th, rec) for rec in jobs[name]]
                L[i, j] = np.mean([x[0] for x in r]); S[i, j] = np.mean([x[1] for x in r]); G[i, j] = np.mean([x[2] for x in r])
        tab[m] = {"thetas": thetas, "L": L, "S": S, "G": G}
    return names, tab


def certify(tab, rho, beta=0.05):
    out = {}
    for m, t in tab.items():
        i = bench.select(t["L"], rho, beta)
        out[m] = None if i is None else t["thetas"][i]
    return out


def evaluate(tab_test, thresholds):
    rows = []
    for m, th in thresholds.items():
        t = tab_test[m]
        if th is None:
            rows.append({"method": m, "theta": None, "risk": 0.0, "saved": 0.0, "gap_pp": 0.0}); continue
        i = t["thetas"].index(th)
        rows.append({"method": m, "theta": th, "risk": t["L"][i].mean(), "saved": t["S"][i].mean(), "gap_pp": t["G"][i].mean()})
    return pd.DataFrame(rows)
