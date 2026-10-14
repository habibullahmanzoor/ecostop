"""Leave-one-schedule-family-out, stress test of the learned score:
 - feature-group ablation (which inputs carry the robustness?),
 - rule-based schedule-aware baselines (sudden-drop gate) so the learned score is not compared with a strawman,
 - pooled over the 16 held-out jobs of the three held-out families."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import phase1_gate as G
from echostop import phase1_prep as P
from echostop.phase1_c3 import EPS, GAM, features, make_model, stops_from_p
from echostop.phase1_eval import folds_for
from echostop.tierA import select

GROUPS = {"all features": [], "no LR-phase features": [16, 17, 18, 19]}
HELD = ("step", "onecycle", "cosine_warmup")
RHOS = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]


def main():
    runs = P.load_runs()
    for r in runs:
        G.sudden_blocks(r)
    Mg = G.build_matrices()                                  # eco / pat with sudden-drop gates (corrected gate length)
    names = list(Mg["names"]); jidx = {j: i for i, j in enumerate(names)}
    dom, fam = Mg["domain"], Mg["family"]
    strata = np.array([f"{d}|{f}" for d, f in zip(dom, fam)])
    J = len(names); et = Mg["et"]
    Xs, ys, ws, rj, offs, pos = [], [], [], [], [], 0
    for r in runs:
        X, y = features(r)
        Xs.append(X); ys.append(y); ws.append(np.full(len(y), 1.0 / r["T"])); rj.append(np.full(len(y), jidx[r["job"]]))
        offs.append((pos, pos + len(y))); pos += len(y)
    X, y, w, rjob = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(rj)
    run_job = np.array([jidx[r["job"]] for r in runs])
    eco, pat = P.FAMILIES["eco"], P.FAMILIES["pat"]
    idx = {
        "Patience p=fT, min-delta": ("pat", sorted([i for i, c in enumerate(pat) if c["md"] == 0.1 and c["h"] == "sd0"], key=lambda i: -pat[i]["f"])),
        "Patience + sudden-drop gate h=4": ("pat", sorted([i for i, c in enumerate(pat) if c["md"] == 0.1 and c["h"] == "sd4"], key=lambda i: -pat[i]["f"])),
        "EcoStop k=1": ("eco", sorted([i for i, c in enumerate(eco) if c["k"] == 1 and c["h"] == "sd0"], key=lambda i: eco[i]["g"])),
        "EcoStop k=1 + sudden-drop gate h=4": ("eco", sorted([i for i, c in enumerate(eco) if c["k"] == 1 and c["h"] == "sd4"], key=lambda i: eco[i]["g"])),
    }
    res = {}      # (method, family) -> (risk-per-job array, es-per-job array, te idx)
    for held in HELD:
        te = np.where(fam == held)[0]; tr = np.where(fam != held)[0]
        for name, (fm, ids) in idx.items():
            L, S, ES, Gp = Mg[fm]
            for rho in RHOS:
                i = select(L[ids][:, tr], rho, 0.05)
                res[(name, held, rho)] = (np.zeros(len(te)) if i is None else L[ids[i], te], np.zeros(len(te)) if i is None else ES[ids[i], te], te)
        inner = folds_for(strata[tr], 7)
        for gname, drop in GROUPS.items():
            keep = [c for c in range(X.shape[1]) if c not in drop]
            p_rows = np.full(len(y), np.nan)
            for ik in range(5):
                m = np.isin(rjob, tr[inner != ik]); h = np.isin(rjob, tr[inner == ik])
                p_rows[h] = make_model().fit(X[m][:, keep], y[m], sample_weight=w[m]).predict_proba(X[h][:, keep])[:, 1]
            m = np.isin(rjob, tr); h = np.isin(rjob, te)
            p_rows[h] = make_model().fit(X[m][:, keep], y[m], sample_weight=w[m]).predict_proba(X[h][:, keep])[:, 1]
            loss, es, cnt = np.zeros((len(GAM), J)), np.zeros((len(GAM), J)), np.zeros(J)
            for ri, r in enumerate(runs):
                a, e, T = r["a"], r["e"], r["T"]
                pr = np.full(T, np.nan); s0, s1 = offs[ri]; pr[2:] = p_rows[s0:s1]
                j = run_job[ri]
                if not (np.isin(j, tr) or np.isin(j, te)):
                    continue
                cnt[j] += 1
                for ci, g in enumerate(GAM):
                    t = stops_from_p(r, pr, g, 1, "sd0")
                    loss[ci, j] += float(a[-1] - a[t] > EPS); es[ci, j] += e[-1] - e[t]
            loss /= np.maximum(cnt, 1)
            for rho in RHOS:
                i = select(loss[:, tr], rho, 0.05)
                res[("EcoStop-L: " + gname, held, rho)] = (np.zeros(len(te)) if i is None else loss[i, te], np.zeros(len(te)) if i is None else es[i, te], te)
        print("done", held, flush=True)

    rows = []
    methods = list(dict.fromkeys(k[0] for k in res))
    for m in methods:
        for rho in RHOS:
            Ls, ESs, ETs = [], 0.0, 0.0
            for held in HELD:
                L, ES, te = res[(m, held, rho)]
                Ls.append(L); ESs += ES.sum(); ETs += et[te].sum()
            rows.append({"method": m, "rho": rho, "risk_pooled16": np.concatenate(Ls).mean(), "EW_pooled16": 100 * ESs / ETs})
    df = pd.DataFrame(rows)
    df.to_csv("figures/tierA/phase1_c3_lfo3.csv", index=False)
    pd.set_option("display.width", 250)
    out = []
    for m, sub in df.groupby("method", sort=False):
        sub = sub.sort_values("risk_pooled16"); x, yv = sub.risk_pooled16.values, sub.EW_pooled16.values
        out.append({"method": m, **{f"EW@risk{t:.2f}": (float(np.interp(t, x, yv)) if x.min() <= t <= x.max() else np.nan) for t in (0.08, 0.10, 0.15, 0.20, 0.25)},
                    "min_risk": x.min(), "max_risk": x.max()})
    print("Pooled over the 16 held-out jobs (3 never-seen schedule families): EW at matched realised risk")
    print(pd.DataFrame(out).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
