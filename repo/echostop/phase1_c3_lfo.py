"""Leave-one-schedule-family-out test of the learned score (C3): the held-out family contributes NOTHING to
model training, to gamma calibration, or to the baselines' calibration. Does the learned score still avoid the
plateau-then-jump failure on a schedule family it has never seen?"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import phase1_prep as P
from echostop.phase1_c3 import EPS, GAM, features, make_model, stops_from_p
from echostop.phase1_eval import folds_for
from echostop.phase1_gate import sudden_blocks
from echostop.tierA import select


def main():
    runs = P.load_runs()
    for r in runs:
        sudden_blocks(r)
    M = P.load_mats()
    jobs = list(M["names"]); jidx = {j: i for i, j in enumerate(jobs)}
    dom, fam = M["domain"], M["family"]
    strata = np.array([f"{d}|{f}" for d, f in zip(dom, fam)])
    J = len(jobs)
    Xs, ys, ws, rj, offs, pos = [], [], [], [], [], 0
    for r in runs:
        X, y = features(r)
        Xs.append(X); ys.append(y); ws.append(np.full(len(y), 1.0 / r["T"])); rj.append(np.full(len(y), jidx[r["job"]]))
        offs.append((pos, pos + len(y))); pos += len(y)
    X, y, w, rjob = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(rj)
    run_job = np.array([jidx[r["job"]] for r in runs])
    et = M["et"]
    eco, pat = P.FAMILIES["eco"], P.FAMILIES["pat"]
    pat_idx = sorted([i for i, c in enumerate(pat) if c["md"] == 0.1 and c["h"] == 0], key=lambda i: -pat[i]["f"])
    eco1_idx = sorted([i for i, c in enumerate(eco) if c["nf"] == .125 and c["gf"] == .125 and c["k"] == 1 and c["h"] == 0], key=lambda i: eco[i]["g"])
    rows = []
    for held in ("step", "onecycle", "cosine_warmup"):
        te = np.where(fam == held)[0]; tr = np.where(fam != held)[0]
        inner = folds_for(strata[tr], 7)
        p_rows = np.full(len(y), np.nan)
        for ik in range(5):
            m = np.isin(rjob, tr[inner != ik]); h = np.isin(rjob, tr[inner == ik])
            p_rows[h] = make_model().fit(X[m], y[m], sample_weight=w[m]).predict_proba(X[h])[:, 1]
        m = np.isin(rjob, tr); h = np.isin(rjob, te)
        p_rows[h] = make_model().fit(X[m], y[m], sample_weight=w[m]).predict_proba(X[h])[:, 1]
        loss, es, sv, cnt = np.zeros((len(GAM), J)), np.zeros((len(GAM), J)), np.zeros((len(GAM), J)), np.zeros(J)
        for ri, r in enumerate(runs):
            a, e, T = r["a"], r["e"], r["T"]
            pr = np.full(T, np.nan); s0, s1 = offs[ri]; pr[2:] = p_rows[s0:s1]
            j = run_job[ri]; cnt[j] += 1
            for ci, g in enumerate(GAM):
                t = stops_from_p(r, pr, g, 1, "sd0")
                loss[ci, j] += float(a[-1] - a[t] > EPS); es[ci, j] += e[-1] - e[t]
                sv[ci, j] += 100 * (1 - e[t] / e[-1]) if e[-1] > 0 else 0.0
        loss /= cnt; sv /= cnt
        for rho in (0.15, 0.20):
            out = {}
            i = select(loss[:, tr], rho, 0.05)
            out["EcoStop-L (learned) k=1"] = None if i is None else (loss[i, te].mean(), es[i, te].sum() / et[te].sum() * 100, sv[i, te].mean())
            for name, fm, idx in (("Patience p=fT, min-delta", "pat", pat_idx), ("EcoStop k=1", "eco", eco1_idx)):
                L, S, ES, G = M[fm]
                i = select(L[idx][:, tr], rho, 0.05)
                out[name] = None if i is None else (L[idx[i], te].mean(), ES[idx[i], te].sum() / et[te].sum() * 100, S[idx[i], te].mean())
            for name, v in out.items():
                rows.append({"held_out_family": held, "jobs": len(te), "rho": rho, "rule": name,
                             "risk": np.nan if v is None else v[0], "EW": np.nan if v is None else v[1], "mean_saved": np.nan if v is None else v[2]})
        print(f"done {held}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("figures/tierA/phase1_c3_lfo.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
