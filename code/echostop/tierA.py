"""Tier A re-analysis (IMPROVEMENT_PLAN.md A1-A4, A7-A8). CPU only.

One loss definition everywhere: a job's loss is the FRACTION of its seeds whose
audited accuracy gap exceeds eps (in [0,1]); this is the probability that a
single deployed run violates. Certification uses fixed-sequence testing over a
conservative->aggressive parameter ordering with a Hoeffding-Bentkus p-value
(Angelopoulos et al., LTT), so there is no Bonferroni penalty. Every method is
calibrated with the same wrapper, and evaluated out-of-fold (K-fold, repeated).
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.analyze_pilot import discover_runs
from echostop.calibration import audited_loss, forward_fill_audit, precompute_score_trace, simulate_from_trace
from echostop.run_calibration_demo2 import discover_tabular_runs
from echostop.run_calibration_final_split import discover_nlp_runs

EPS = 0.5
OUT = "figures/tierA"
CACHE = "cache/traces.pkl"
GAMMAS = np.round(np.linspace(0.02, 0.98, 25), 2)


# ---------------------------------------------------------------- policies
def patience_stop_md(df: pd.DataFrame, p: int, delta: float = 0.0) -> int:
    best, since = -np.inf, 0
    for t, v in enumerate(df["val_acc"].values):
        if v > best + delta:
            best, since = v, 0
        else:
            since += 1
        if since >= p:
            return t
    return len(df) - 1


def build_policies(traces_for):
    """Each family: ordered list (conservative -> aggressive) of (label, fn(df, trace)->stop epoch)."""
    fam = {}
    fam["EcoStop"] = [(f"g={g}", lambda df, tr, g=g: simulate_from_trace(tr, g, len(df))) for g in GAMMAS]
    fam["Patience (fixed p)"] = [(f"p={p}", lambda df, tr, p=p: patience_stop_md(df, p)) for p in range(20, 0, -1)]
    fracs = [0.5, 0.4, 0.33, 0.3, 0.25, 0.2, 0.175, 0.15, 0.125, 0.1, 0.08, 0.06]
    fam["Patience (p=f*T)"] = [(f"f={f}", lambda df, tr, f=f: patience_stop_md(df, max(1, round(f * len(df))))) for f in fracs]
    fam["Patience (p=f*T, min-delta 0.1)"] = [
        (f"f={f}", lambda df, tr, f=f: patience_stop_md(df, max(1, round(f * len(df))), 0.1)) for f in fracs]
    fam["Fixed budget fraction"] = [
        (f"f={f}", lambda df, tr, f=f: int(round(f * (len(df) - 1)))) for f in np.round(np.arange(0.95, 0.05, -0.05), 2)]
    return fam


# ---------------------------------------------------------------- stats
def hb_pvalue(rhat: float, n: int, rho: float) -> float:
    """Hoeffding-Bentkus p-value for H0: E[loss] > rho, loss in [0,1]."""
    if rhat >= rho:
        return 1.0
    a, b = max(rhat, 1e-12), rho
    h1 = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    p_bennett = np.exp(-n * h1)
    p_bentkus = np.e * binom.cdf(np.ceil(n * rhat - 1e-9), n, rho)
    return float(min(1.0, p_bennett, p_bentkus))


def select(loss_mat: np.ndarray, rho: float, beta: float) -> int | None:
    """loss_mat [n_params, n_jobs], params ordered conservative->aggressive.
    Fixed-sequence testing: certify while p <= beta; return last certified index."""
    n = loss_mat.shape[1]
    last = None
    for i in range(loss_mat.shape[0]):
        if hb_pvalue(float(loss_mat[i].mean()), n, rho) <= beta:
            last = i
        else:
            break
    return last


# ---------------------------------------------------------------- data
def load():
    vision = {f"{d}_{m}": dfs for (d, m), dfs in discover_runs().items()}
    tab = discover_tabular_runs()
    nlp = discover_nlp_runs()
    jobs = {**vision, **tab, **nlp}
    dom = {k: ("nlp" if k in nlp else "vision" if k in vision else "tabular") for k in jobs}
    if os.path.exists(CACHE):
        traces = pickle.load(open(CACHE, "rb"))
    else:
        traces = {k: [precompute_score_trace(df, EPS) for df in dfs] for k, dfs in jobs.items()}
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        pickle.dump(traces, open(CACHE, "wb"))
    return jobs, traces, dom


def evaluate_family(jobs, traces, plist):
    """Returns loss[P,J], saved[P,J] (job-mean %), eSaved[P,J], eTotal[J] (joules, summed over seeds)."""
    names = list(jobs)
    P, J = len(plist), len(names)
    loss, saved, e_s = np.zeros((P, J)), np.zeros((P, J)), np.zeros((P, J))
    e_t = np.zeros(J)
    for j, k in enumerate(names):
        for df, tr in zip(jobs[k], traces[k]):
            e = df["cumulative_energy_joules"].values
            a = forward_fill_audit(df)
            e_t[j] += e[-1]
            for i, (_, fn) in enumerate(plist):
                t = fn(df, tr)
                loss[i, j] += float(a[-1] - a[t] > EPS)
                saved[i, j] += 100 * (1 - e[t] / e[-1])
                e_s[i, j] += e[-1] - e[t]
        ns = len(jobs[k])
        loss[:, j] /= ns
        saved[:, j] /= ns
    return loss, saved, e_s, e_t


# ---------------------------------------------------------------- cross-fit
def evaluate_gap(jobs, traces, plist):
    """gap[P,J]: job-mean accuracy gap A_T - A_tau in percentage points (can be negative)."""
    names = list(jobs)
    gap = np.zeros((len(plist), len(names)))
    for j, k in enumerate(names):
        for df, tr in zip(jobs[k], traces[k]):
            a = forward_fill_audit(df)
            for i, (_, fn) in enumerate(plist):
                gap[i, j] += a[-1] - a[fn(df, tr)]
        gap[:, j] /= len(jobs[k])
    return gap


def crossfit(loss, saved, e_s, e_t, dom_arr, rho, beta, K=5, R=20, per_domain=False, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    J = loss.shape[1]
    res = []
    EX = []
    for r in range(R):
        fold = np.zeros(J, int)
        for d in np.unique(dom_arr):
            idx = np.where(dom_arr == d)[0]
            rng.shuffle(idx)
            fold[idx] = np.arange(len(idx)) % K
        tl, ts, tes, tet, ncert = np.zeros(J), np.zeros(J), np.zeros(J), np.zeros(J), 0
        tx = np.zeros(J)
        for f in range(K):
            te = fold == f
            groups = [(np.where(dom_arr == d)[0]) for d in np.unique(dom_arr)] if per_domain else [np.arange(J)]
            for g in groups:
                tr_idx = g[~te[g]]
                te_idx = g[te[g]]
                i = select(loss[:, tr_idx], rho, beta)
                if i is None:  # nothing certified -> run to completion (zero savings, zero risk)
                    ts[te_idx] = 0; tl[te_idx] = 0; tes[te_idx] = 0; tet[te_idx] = e_t[te_idx]; ncert += 1
                else:
                    tl[te_idx] = loss[i, te_idx]; ts[te_idx] = saved[i, te_idx]
                    tes[te_idx] = e_s[i, te_idx]; tet[te_idx] = e_t[te_idx]
                    if extra is not None:
                        tx[te_idx] = extra[i, te_idx]
        res.append((tl, ts, tes.sum() / tet.sum() * 100, ncert))
        EX.append(tx)
    L = np.array([x[0] for x in res]); S = np.array([x[1] for x in res])
    return {"risk": L.mean(), "risk_by_dom": {d: L[:, dom_arr == d].mean() for d in np.unique(dom_arr)},
            "saved": S.mean(), "ew": np.mean([x[2] for x in res]),
            "sd_risk": L.mean(1).std(), "n_uncert": np.mean([x[3] for x in res]),
            "job_risk": L.mean(0), "job_saved": S.mean(0),
            "job_extra": np.array(EX).mean(0) if extra is not None else None}


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs, traces, dom = load()
    names = list(jobs)
    dom_arr = np.array([dom[k] for k in names])
    fam = build_policies(None)
    mats = {}
    for fname, plist in fam.items():
        print(f"evaluating {fname} ({len(plist)} params)...", flush=True)
        mats[fname] = evaluate_family(jobs, traces, plist)
    pickle.dump({"mats": mats, "names": names, "dom": dom_arr}, open("cache/tierA_mats.pkl", "wb"))

    rows = []
    for beta in (0.05,):
        for rho in (0.10, 0.15, 0.20):
            for fname in fam:
                for pd_ in (False, True):
                    r = crossfit(*mats[fname], dom_arr, rho, beta, per_domain=pd_)
                    rows.append({"method": fname, "per_domain": pd_, "rho": rho, "beta": beta,
                                 "oof_risk": r["risk"], "risk_vision": r["risk_by_dom"]["vision"],
                                 "risk_tabular": r["risk_by_dom"]["tabular"], "risk_nlp": r["risk_by_dom"]["nlp"],
                                 "mean_saved": r["saved"], "energy_weighted_saved": r["ew"],
                                 "n_uncertified_folds": r["n_uncert"]})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/crossfit_frontier.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(df.round(3).to_string(index=False))

    # full-data certificate for EcoStop at several (rho, beta)
    print("\nFull-data (in-sample) certified EcoStop gamma, per-run loss, fixed-sequence HB:")
    loss = mats["EcoStop"][0]
    for beta in (0.05, 0.10, 0.20):
        for rho in (0.10, 0.15, 0.20, 0.25):
            i = select(loss, rho, beta)
            print(f"  beta={beta} rho={rho}: gamma_hat={None if i is None else GAMMAS[i]}")
    print("per-run risk curve (all 142 jobs):", dict(zip(GAMMAS.tolist(), loss.mean(1).round(3).tolist())))


if __name__ == "__main__":
    main()
