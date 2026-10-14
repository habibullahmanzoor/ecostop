"""Phase 1, step 5: C3 learned continuation score (EcoStop-L), strictly nested.

p_t = P(stopping at epoch t loses > eps test accuracy), predicted from information available at epoch t
(curve shape, EcoStop's own score, patience's signals, noise, training loss, learning-rate phase, domain).
Inside each outer fold, the calibration jobs get OUT-OF-FOLD predictions (inner 5-fold by job); the final
model (trained on all calibration jobs) scores the held-out fold. gamma is then certified exactly as for
every other rule (fixed-sequence Hoeffding-Bentkus). Same outer folds as the other methods."""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import phase1_prep as P
from echostop.phase1_eval import K, R, bootstrap_diff, folds_for
from echostop.phase1_gate import sudden_blocks
from echostop.tierA import select

EPS = 0.5
GAM = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
DOM = {"vision": 0, "nlp": 1, "tabular": 2}
RHOS = [round(x, 2) for x in np.arange(0.08, 0.31, 0.02)]


def features(r):
    T = r["T"]; q, b, s, gh, sg, fl, lr, lt = r["q"], r["b"], r["s"], r["ghat"], r["sigma"], r["flat"], r["lr"], r["loss_tr"]
    t = np.arange(T)
    def lag(x, k): return x - x[np.maximum(t - k, 0)]
    since = np.zeros(T)
    best, c = -np.inf, 0
    for i in range(T):
        if q[i] > best: best, c = q[i], 0
        else: c += 1
        since[i] = c
    noise = np.array([np.std(np.diff(q[max(0, i - 5): i + 1])) if i >= 2 else 0.0 for i in range(T)])
    nxt = np.minimum(t + 3, T - 1); nxt6 = np.minimum(t + 6, T - 1)
    drop = np.zeros(T, bool); drop[1:] = lr[1:] <= 0.5 * np.maximum(lr[:-1], 1e-12)
    until = np.full(T, float(T))
    nxt_drop = T
    for i in range(T - 1, -1, -1):
        if i + 1 < T and drop[i + 1]: nxt_drop = i + 1
        until[i] = nxt_drop - i if nxt_drop < T else T
    head = (b - q[0]) / (100 - q[0] + 1e-6)
    l0 = max(lt[0], 1e-9)
    F = np.column_stack([t / T, np.full(T, T), b, q - b, lag(b, 1), lag(b, 3) / 3, lag(b, 5) / 5, lag(b, 10) / 10,
                         fl / T, since / T, noise, s, gh, sg, lt / l0, lag(lt, 3) / l0, lr, lr[nxt] - lr, lr[nxt6] - lr,
                         until / T, head, np.full(T, DOM[r["domain"]])])
    y = (r["a"][-1] - r["a"] > EPS).astype(int)
    return F[2:], y[2:]


def stops_from_p(r, p, g, k, hkey):
    T, blk = r["T"], r["block"][hkey]
    N = max(2, round(0.125 * T))
    consec = 0
    for t in range(N, T):
        if p[t] == p[t] and p[t] < g and not blk[t]:
            consec += 1
            if consec >= k:
                return t
        else:
            consec = 0
    return T - 1


def make_model():
    return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1, max_depth=4, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)


def main():
    runs = P.load_runs()
    for r in runs:
        sudden_blocks(r)
    jobs = list(P.load_mats()["names"])   # same job order as the baselines => identical outer folds
    jidx = {j: i for i, j in enumerate(jobs)}
    J = len(jobs)
    dom = np.array([next(r["domain"] for r in runs if r["job"] == j) for j in jobs])
    fam = np.array([next(r["family"] for r in runs if r["job"] == j) for j in jobs])
    strata = np.array([f"{d}|{f}" for d, f in zip(dom, fam)])
    et = np.zeros(J)
    for r in runs: et[jidx[r["job"]]] += r["e"][-1]
    # features once
    Xs, ys, ws, rj = [], [], [], []
    offs = []
    pos = 0
    for r in runs:
        X, y = features(r)
        Xs.append(X); ys.append(y); ws.append(np.full(len(y), 1.0 / r["T"])); rj.append(np.full(len(y), jidx[r["job"]]))
        offs.append((pos, pos + len(y))); pos += len(y)
    X, y, w, rjob = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(rj)
    print("rows:", X.shape, "positive rate:", y.mean().round(3), flush=True)
    run_job = np.array([jidx[r["job"]] for r in runs])

    configs = [(g, k, h) for g in GAM for k in (1, 2) for h in ("sd0", "sd4")]
    CFG = {(k, h): [i for i, c in enumerate(configs) if c[1] == k and c[2] == h] for k in (1, 2) for h in ("sd0", "sd4")}
    out = {key: {"L": np.zeros((R, len(RHOS), J)), "ES": np.zeros((R, len(RHOS), J)), "S": np.zeros((R, len(RHOS), J))}
           for key in CFG}
    for rep in range(R):
        fold = folds_for(strata, rep)
        for f in range(K):
            te = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
            # inner OOF predictions for calibration jobs
            inner = folds_for(strata[tr], 100 + rep * 10 + f)
            p_rows = np.full(len(y), np.nan)
            for ik in range(K):
                fit_jobs = tr[inner != ik]; hold_jobs = tr[inner == ik]
                m = np.isin(rjob, fit_jobs)
                mdl = make_model().fit(X[m], y[m], sample_weight=w[m])
                h = np.isin(rjob, hold_jobs)
                p_rows[h] = mdl.predict_proba(X[h])[:, 1]
            m = np.isin(rjob, tr)
            final = make_model().fit(X[m], y[m], sample_weight=w[m])
            h = np.isin(rjob, te)
            p_rows[h] = final.predict_proba(X[h])[:, 1]
            # simulate every config on every run using the appropriate predictions
            C = len(configs)
            loss, es, sv = np.zeros((C, J)), np.zeros((C, J)), np.zeros((C, J))
            cnt = np.zeros(J)
            for ri, r in enumerate(runs):
                a, e, T = r["a"], r["e"], r["T"]
                pr = np.full(T, np.nan); s0, s1 = offs[ri]; pr[2:] = p_rows[s0:s1]
                j = run_job[ri]; cnt[j] += 1
                for ci, (g, k, hk) in enumerate(configs):
                    t = stops_from_p(r, pr, g, k, hk)
                    loss[ci, j] += float(a[-1] - a[t] > EPS)
                    es[ci, j] += e[-1] - e[t]
                    sv[ci, j] += 100 * (1 - e[t] / e[-1]) if e[-1] > 0 else 0.0
            loss /= cnt; sv /= cnt
            for key, idx in CFG.items():
                idx = sorted(idx, key=lambda i: configs[i][0])
                for ri_, rho in enumerate(RHOS):
                    i = select(loss[idx][:, tr], rho, 0.05)
                    if i is not None:
                        c = idx[i]
                        out[key]["L"][rep, ri_, te] = loss[c, te]; out[key]["ES"][rep, ri_, te] = es[c, te]; out[key]["S"][rep, ri_, te] = sv[c, te]
        print(f"repeat {rep + 1}/{R} done", flush=True)
    res = {key: {m: v.mean(0) for m, v in d.items()} for key, d in out.items()}
    pickle.dump({"res": res, "jobs": jobs, "dom": dom, "fam": fam, "et": et}, open("cache/p1_c3.pkl", "wb"))
    return res, dom, fam, et


if __name__ == "__main__":
    main()
