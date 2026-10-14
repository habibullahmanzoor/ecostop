"""EcoStop-Bench: matched-risk, out-of-fold evaluation of early-stopping rules on energy-metered runs.

Self-contained (numpy, pandas, scipy). Usage:
    from bench import load, evaluate, RULES
    runs = load("runs.csv.gz")                              # development corpus, 142 jobs / 535 runs
    evaluate(runs, "patience_fraction", rho=0.20)           # -> dict(risk, mean_saved, energy_weighted_saved, ...)
    # a rule is  stop_epoch = rule(run, theta);  thetas are ordered conservative -> aggressive.

Protocol (fixed):
  * loss of a run: 1[ A_T - A_tau > eps ] with A_t = test accuracy of the checkpoint with the best validation accuracy up
    to epoch t (never re-maximised on test data), eps = 0.5 pp. Loss of a JOB = fraction of its seeds that violate.
  * calibration: fixed-sequence testing with Hoeffding-Bentkus p-values, beta = 0.05; the last certified theta is deployed;
    if nothing is certified the run goes to completion.
  * evaluation: K=5 folds stratified by (domain, schedule family), repeated R times; every job is scored by a theta
    calibrated on the other folds. Report risk, mean saving, energy-weighted saving (total J saved / total J).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import binom, norm

EPS = 0.5


# ---------------------------------------------------------------- data
def load(path="runs.csv.gz", sets=("dev",)):
    df = pd.read_csv(path)
    df = df[df["set"].isin(sets)]
    runs = []
    for (job, seed), g in df.groupby(["job", "seed"], sort=False):
        g = g.sort_values("epoch")
        q = g["val_acc"].values.astype(float)
        runs.append({"job": job, "seed": int(seed), "domain": g["domain"].iloc[0], "family": g["schedule_family"].iloc[0],
                     "T": len(g), "q": q, "b": np.maximum.accumulate(q), "a": g["test_acc_at_best_val"].values.astype(float),
                     "e": g["cumulative_energy_J"].values.astype(float), "lr": g["lr_multiplier"].values.astype(float)})
    return runs


# ---------------------------------------------------------------- certification
def hb_pvalue(rhat, n, rho):
    if rhat >= rho:
        return 1.0
    a, b = max(rhat, 1e-12), rho
    h1 = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    return float(min(1.0, np.exp(-n * h1), np.e * binom.cdf(np.ceil(n * rhat - 1e-9), n, rho)))


def select(loss, rho, beta=0.05):
    """loss[theta, job], thetas ordered conservative -> aggressive; fixed-sequence testing."""
    last = None
    for i in range(loss.shape[0]):
        if hb_pvalue(float(loss[i].mean()), loss.shape[1], rho) <= beta:
            last = i
        else:
            break
    return last


# ---------------------------------------------------------------- rules (stop epoch, 0-based) and parameter grids
def patience(run, p, md=0.0):
    best, since = -np.inf, 0
    for t, v in enumerate(run["q"]):
        if v > best + md:
            best, since = v, 0
        else:
            since += 1
        if since >= p:
            return t
    return run["T"] - 1


def patience_fraction(run, f, md=0.1):
    return patience(run, max(1, round(f * run["T"])), md)


def fixed_budget(run, f):
    return int(round(f * (run["T"] - 1)))


def _fit(t, q, window=15):
    if len(t) < 4:
        return None
    if len(t) > window:
        t, q = t[-window:], q[-window:]
    qm = q.max()
    a0 = min(qm + 1.0, 100.0)
    try:
        popt, pcov = curve_fit(lambda x, a, b, c: a - b * np.exp(-c * x), t, q, p0=[a0, max(a0 - q[0], 1.0), 0.1],
                               bounds=([0, 0, 1e-4], [100.0, 200.0, 5.0]), maxfev=5000)
    except RuntimeError:
        return None
    a = popt[0]
    if a >= 99.5:                                    # asymptote pinned at the ceiling: read as converged
        return qm, max(float(np.std(q - qm)), 0.05)
    s = np.sqrt(pcov[0, 0]) if np.isfinite(pcov[0, 0]) else 100.0
    return max(a, qm), max(float(s), 0.05)


def ecostop_scores(run):
    """EcoStop continuation score s_t = P(still-attainable gain > eps) (cached per run)."""
    if "_s" in run:
        return run["_s"]
    T, q, b = run["T"], run["q"], run["b"]
    s = np.full(T, np.nan)
    tt = np.arange(T, dtype=float)
    for t in range(max(2, round(0.125 * T)), T):
        f = _fit(tt[: t + 1], q[: t + 1])
        if f is not None:
            s[t] = 1.0 - norm.cdf(EPS, loc=f[0] - b[t], scale=f[1])
    run["_s"] = s
    return s


def ecostop(run, gamma, k=2):
    T, b, s = run["T"], run["b"], ecostop_scores(run)
    N = max(2, round(0.125 * T))
    consec = 0
    for t in range(N, T):
        flat = 0
        for u in range(t, 0, -1):
            if b[u] - b[u - 1] > 0.10:
                break
            flat += 1
        fire = s[t] == s[t] and s[t] < gamma and flat >= N
        consec = consec + 1 if fire else 0
        if consec >= k:
            return t
    return T - 1


RULES = {   # name -> (rule, thetas ordered conservative -> aggressive)
    "patience": (patience, list(range(20, 0, -1))),
    "patience_fraction": (patience_fraction, [0.5, 0.4, 0.33, 0.3, 0.25, 0.2, 0.175, 0.15, 0.125, 0.1, 0.08, 0.06]),
    "fixed_budget": (fixed_budget, [round(x, 2) for x in np.arange(0.95, 0.05, -0.05)]),
    "ecostop": (ecostop, [round(x, 2) for x in np.linspace(0.02, 0.98, 25)]),
}


# ---------------------------------------------------------------- outcomes and out-of-fold evaluation
def job_table(runs, rule, thetas):
    """Per theta and job: loss (fraction of violating seeds), mean saving %, joules saved, joules total, accuracy gap."""
    jobs = sorted({r["job"] for r in runs})
    by = {j: [r for r in runs if r["job"] == j] for j in jobs}
    P, J = len(thetas), len(jobs)
    L, S, ES, G = np.zeros((P, J)), np.zeros((P, J)), np.zeros((P, J)), np.zeros((P, J))
    ET = np.zeros(J)
    for j, name in enumerate(jobs):
        for r in by[name]:
            ET[j] += r["e"][-1]
            for i, th in enumerate(thetas):
                t = rule(r, th)
                L[i, j] += float(r["a"][-1] - r["a"][t] > EPS)
                S[i, j] += 100 * (1 - r["e"][t] / r["e"][-1])
                ES[i, j] += r["e"][-1] - r["e"][t]
                G[i, j] += r["a"][-1] - r["a"][t]
        n = len(by[name])
        L[:, j] /= n; S[:, j] /= n; G[:, j] /= n
    strata = np.array([f"{by[j][0]['domain']}|{by[j][0]['family']}" for j in jobs])
    return {"jobs": jobs, "L": L, "S": S, "ES": ES, "G": G, "ET": ET, "strata": strata,
            "domain": np.array([by[j][0]["domain"] for j in jobs])}


def evaluate(runs, rule_name, rho=0.20, beta=0.05, K=5, R=20, seed=0, table=None):
    rule, thetas = RULES[rule_name]
    T = table or job_table(runs, rule, thetas)
    J = len(T["jobs"])
    rng = np.random.default_rng(seed)
    L, S, ES = np.zeros((R, J)), np.zeros((R, J)), np.zeros((R, J))
    for r in range(R):
        fold = np.zeros(J, int)
        for s in np.unique(T["strata"]):
            idx = np.where(T["strata"] == s)[0]
            rng.shuffle(idx)
            fold[idx] = (np.arange(len(idx)) + rng.integers(K)) % K
        for f in range(K):
            te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
            i = select(T["L"][:, tr], rho, beta)
            if i is not None:
                L[r, te], S[r, te], ES[r, te] = T["L"][i, te], T["S"][i, te], T["ES"][i, te]
    return {"rule": rule_name, "rho": rho, "risk": L.mean(), "mean_saved": S.mean(),
            "energy_weighted_saved": 100 * ES.mean(0).sum() / T["ET"].sum(),
            "risk_by_domain": {d: L.mean(0)[T["domain"] == d].mean() for d in np.unique(T["domain"])}}
