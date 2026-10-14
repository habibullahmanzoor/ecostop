"""Phase 1 (ECOSTOP_BEST_PLAN.md), step 1: enlarged development set (142 dev + 35 B1 = 177 jobs)
and per-run caches: EcoStop curve-fit score traces, learning-rate schedule, config-simulation matrices.
CPU only. Nothing here touches the GPU."""
from __future__ import annotations

import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E
from echostop import tierA
from echostop.calibration import forward_fill_audit
from echostop.methods import fit_exp_saturation

EPS = 0.5
RUNS_CACHE = "cache/p1_runs.pkl"
MATS_CACHE = "cache/p1_mats.pkl"
N0_FRAC = 0.05            # earliest start of score traces
LR_THETA = 0.15           # schedule gate: |lr change within h epochs| >= theta * lr_max


# ------------------------------------------------------------------ learning-rate schedule (per epoch)
def lr_curve(family: str, T: int, warmup: int = 0) -> np.ndarray:
    t = np.arange(T, dtype=float)
    if family in ("cosine", "tab_cosine"):
        lr = 0.5 * (1 + np.cos(np.pi * t / T))
    elif family == "cosine_warmup":
        lr = np.where(t < warmup, (t + 1) / max(1, warmup),
                      0.5 * (1 + np.cos(np.pi * (t - warmup) / max(1, T - warmup))))
    elif family == "step":
        lr = np.where(t < int(0.5 * T), 1.0, np.where(t < int(0.75 * T), 0.1, 0.01))
    elif family == "onecycle":
        x = (t + 0.5) / T
        init, fin = 1 / 25, 1 / (25 * 1e4)
        up = init + (1 - init) * (1 - np.cos(np.pi * x / 0.3)) / 2
        down = fin + (1 - fin) * (1 + np.cos(np.pi * (x - 0.3) / 0.7)) / 2
        lr = np.where(x < 0.3, up, down)
    elif family == "nlp_linear":
        x = (t + 0.5) / T
        lr = np.where(x < 0.1, x / 0.1, (1 - x) / 0.9)
    else:
        raise ValueError(family)
    return lr / lr.max()


def block_arrays(lr: np.ndarray) -> dict:
    """block[h][t] True => a scheduled LR change of >= LR_THETA*lr_max lies within the next h epochs."""
    T = len(lr)
    out = {0: np.zeros(T, bool)}
    for h in (2, 4, 6):
        idx = np.minimum(np.arange(T) + h, T - 1)
        out[h] = np.abs(lr[idx] - lr) >= LR_THETA
    return out


# ------------------------------------------------------------------ job registry (dev + B1)
def classify(tag: str, key: str, dom: str):
    if tag == "dev":
        return {"vision": "cosine", "tabular": "tab_cosine", "nlp": "nlp_linear"}[dom], 0
    if key.endswith("mobilenetv3"):
        return "cosine_warmup", 3
    if key.endswith("vit_tiny"):
        return "cosine_warmup", 5
    if key.endswith("resnet18_step"):
        return "step", 0
    if key.endswith("vgg_small"):
        return "onecycle", 0
    return {"vision": "cosine", "tabular": "tab_cosine", "nlp": "nlp_linear"}[dom], 0


def build_runs():
    dev, _, dom_dev = tierA.load()
    b1, dom_b1 = E.load_b1()
    runs = []
    for tag, jobs, dom in (("dev", dev, dom_dev), ("b1", b1, dom_b1)):
        for key, dfs in jobs.items():
            fam, wu = classify(tag, key, dom[key])
            for si, df in enumerate(dfs):
                runs.append({"job": f"{tag}:{key}", "set": tag, "domain": dom[key], "family": fam, "warmup": wu,
                             "seed": si, "T": len(df),
                             "q": df["val_acc"].values.astype(float), "b": df["B_t"].values.astype(float),
                             "a": forward_fill_audit(df), "e": df["cumulative_energy_joules"].values.astype(float),
                             "loss_tr": df["train_loss"].values.astype(float)})
    return runs


def _trace(run):
    T, q, b = run["T"], run["q"], run["b"]
    tt = np.arange(T, dtype=float)
    s = np.full(T, np.nan); gh = np.full(T, np.nan); sg = np.full(T, np.nan)
    n0 = max(2, round(N0_FRAC * T))
    for t in range(n0, T):
        if t + 1 < 4:
            continue
        fit = fit_exp_saturation(tt[: t + 1], q[: t + 1])
        if fit is None:
            continue
        a, _, _, sigma = fit
        s[t] = 1.0 - norm.cdf(EPS, loc=a - b[t], scale=sigma)
        gh[t], sg[t] = a - b[t], sigma
    flat = np.zeros(T, int)
    for t in range(T):
        c = 0
        for u in range(t, 0, -1):
            if b[u] - b[u - 1] > 0.10:
                break
            c += 1
        flat[t] = c
    run.update({"s": s, "ghat": gh, "sigma": sg, "flat": flat,
                "lr": lr_curve(run["family"], T, run["warmup"])})
    run["block"] = block_arrays(run["lr"])
    return run


def load_runs():
    if os.path.exists(RUNS_CACHE):
        return pickle.load(open(RUNS_CACHE, "rb"))
    runs = build_runs()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        runs = list(ex.map(_trace, runs, chunksize=4))
    # real LC-PFN scores (already computed for dev and B1) -> attach
    pf_dev = pickle.load(open("cache/lcpfn_traces.pkl", "rb"))
    pf_b1 = pickle.load(open("cache/b1_lcpfn_traces.pkl", "rb"))
    for r in runs:
        src = pf_dev if r["set"] == "dev" else pf_b1
        tr = src[r["job"].split(":", 1)[1]][r["seed"]]
        s = np.full(r["T"], np.nan)
        for row in tr:
            if row["s_t"] is not None:
                s[row["t"]] = row["s_t"]
        r["s_pfn"] = s
    pickle.dump(runs, open(RUNS_CACHE, "wb"))
    return runs


# ------------------------------------------------------------------ configuration families
GAMMAS = np.round(np.linspace(0.02, 0.98, 25), 2)
PFN_GAMMAS = np.array([0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
FRACS = [0.5, 0.4, 0.33, 0.3, 0.25, 0.2, 0.175, 0.15, 0.125, 0.1, 0.08, 0.06]
NF, GF, KS, HS = (0.05, 0.125, 0.2), (0.0, 0.05, 0.125, 0.2), (1, 2, 3), (0, 2, 4, 6)


def score_configs(gammas, nfs):
    return [dict(g=g, nf=nf, gf=gf, k=k, h=h) for g, nf, gf, k, h in product(gammas, nfs, GF, KS, HS)]


FAMILIES = {
    "eco": score_configs(GAMMAS, NF),
    "pfn": score_configs(PFN_GAMMAS, (0.125, 0.2)),
    "pat": [dict(f=f, md=md, h=h) for f, md, h in product(FRACS, (0.0, 0.05, 0.1, 0.2), HS)],
    "patfix": [dict(p=p, md=md) for p, md in product(range(20, 0, -1), (0.0, 0.1))],
}


def stop_score(run, c, key):
    T, s, flat, blk = run["T"], run[key], run["flat"], run["block"][c["h"]]
    N = max(2, round(c["nf"] * T))
    glen = 0 if c["gf"] == 0 else max(2, round(c["gf"] * T))
    consec = 0
    for t in range(N, T):
        st = s[t]
        if st == st and st < c["g"] and flat[t] >= glen and not blk[t]:
            consec += 1
            if consec >= c["k"]:
                return t
        else:
            consec = 0
    return T - 1


def stop_patience(run, c):
    T, q, blk = run["T"], run["q"], run["block"][c.get("h", 0)]
    p = c["p"] if "p" in c else max(1, round(c["f"] * T))
    best, since = -np.inf, 0
    for t in range(T):
        if q[t] > best + c["md"]:
            best, since = q[t], 0
        else:
            since += 1
        if since >= p and not blk[t]:
            return t
    return T - 1


def sim_job(args):
    """All configs of all families on all seeds of one job -> job-level arrays."""
    jruns = args
    out = {}
    for fam, cfgs in FAMILIES.items():
        C = len(cfgs)
        loss, saved, es, gap = np.zeros(C), np.zeros(C), np.zeros(C), np.zeros(C)
        for r in jruns:
            e, a = r["e"], r["a"]
            for i, c in enumerate(cfgs):
                if fam == "eco":
                    t = stop_score(r, c, "s")
                elif fam == "pfn":
                    t = stop_score(r, c, "s_pfn")
                else:
                    t = stop_patience(r, c)
                loss[i] += float(a[-1] - a[t] > EPS)
                saved[i] += 100 * (1 - e[t] / e[-1]) if e[-1] > 0 else 0.0
                es[i] += e[-1] - e[t]
                gap[i] += a[-1] - a[t]
        n = len(jruns)
        out[fam] = (loss / n, saved / n, es, gap / n)
    out["et"] = sum(r["e"][-1] for r in jruns)
    return out


def load_mats():
    if os.path.exists(MATS_CACHE):
        return pickle.load(open(MATS_CACHE, "rb"))
    runs = load_runs()
    by_job = {}
    for r in runs:
        by_job.setdefault(r["job"], []).append(r)
    names = list(by_job)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as ex:
        res = list(ex.map(sim_job, [by_job[n] for n in names], chunksize=2))
    mats = {"names": names,
            "domain": np.array([by_job[n][0]["domain"] for n in names]),
            "family": np.array([by_job[n][0]["family"] for n in names]),
            "set": np.array([by_job[n][0]["set"] for n in names]),
            "et": np.array([r["et"] for r in res])}
    for fam in FAMILIES:
        mats[fam] = tuple(np.stack([r[fam][i] for r in res], axis=1) for i in range(4))  # loss, saved, es, gap : [C,J]
    pickle.dump(mats, open(MATS_CACHE, "wb"))
    return mats


if __name__ == "__main__":
    m = load_mats()
    print("jobs:", len(m["names"]), {d: int((m["domain"] == d).sum()) for d in ("vision", "nlp", "tabular")})
    print("families:", {f: int((m["family"] == f).sum()) for f in sorted(set(m["family"]))})
    print("configs:", {f: len(c) for f, c in FAMILIES.items()})
