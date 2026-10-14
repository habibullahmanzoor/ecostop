"""Phase 1, step 2: out-of-fold comparison of EcoStop variants (C1 multi-parameter calibration,
C2 schedule gate) against equally treated patience baselines, on the enlarged development set.

Selection rules (all certified with fixed-sequence Hoeffding-Bentkus testing):
  seq1d   : one ordered parameter, calibrated on the whole training fold (as in the paper).
  pareto  : Pareto Testing. Training jobs are split 50/50: the optimisation half orders the
            configurations along its (risk, energy) Pareto frontier; the calibration half certifies
            along that path. Scored once on the held-out fold. Primary metric: energy-weighted saving.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.phase1_prep import FAMILIES, load_mats
from echostop.tierA import hb_pvalue, select

BETA, K, R = 0.05, 5, 5


def folds_for(strata, seed):
    rng = np.random.default_rng(seed)
    J = len(strata)
    fold = np.zeros(J, int)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        fold[idx] = (np.arange(len(idx)) + rng.integers(K)) % K
    return fold


def pareto_path(loss_o, es_o, et_o, rho):
    risk = loss_o.mean(1)
    ew = es_o.sum(1) / et_o.sum()
    order = np.lexsort((-ew, risk))
    path, best = [], -1.0
    for c in order:
        if risk[c] > rho + 0.15:
            break
        if ew[c] > best + 1e-12:
            path.append(c)
            best = ew[c]
    return path


def choose(M, fam, idx, mode, tr, strata, rho, rng):
    loss, saved, es, gap = M[fam]
    et = M["et"]
    idx = np.asarray(idx)
    if mode == "seq1d":
        i = select(loss[idx][:, tr], rho, BETA)
        return None if i is None else idx[i]
    # pareto: stratified 50/50 split of the training jobs
    opt = np.zeros(len(tr), bool)
    for s in np.unique(strata[tr]):
        pos = np.where(strata[tr] == s)[0]
        rng.shuffle(pos)
        opt[pos[: len(pos) // 2]] = True
    o, c = tr[opt], tr[~opt]
    path = pareto_path(loss[idx][:, o], es[idx][:, o], et[o], rho)
    last = None
    for p in path:
        if hb_pvalue(float(loss[idx[p], c].mean()), len(c), rho) <= BETA:
            last = idx[p]
        else:
            break
    return last


def run_method(M, fam, idx, mode, rho, strata, seed0=0):
    loss, saved, es, gap = M[fam]
    J = loss.shape[1]
    L, S, ES, G = (np.zeros((R, J)) for _ in range(4))
    for r in range(R):
        fold = folds_for(strata, seed0 + r)
        rng = np.random.default_rng(1000 + r)
        for f in range(K):
            te = np.where(fold == f)[0]
            tr = np.where(fold != f)[0]
            c = choose(M, fam, idx, mode, tr, strata, rho, rng)
            if c is not None:
                L[r, te], S[r, te], ES[r, te], G[r, te] = loss[c, te], saved[c, te], es[c, te], gap[c, te]
    return {"L": L.mean(0), "S": S.mean(0), "ES": ES.mean(0), "G": G.mean(0)}


def bootstrap_diff(a, b, et, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = a["ES"] - b["ES"]
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = d[idx].sum(1) / et[idx].sum(1) * 100
    return d.sum() / et.sum() * 100, *np.percentile(bs, [2.5, 97.5])


def eco_idx(**kw):
    return [i for i, c in enumerate(FAMILIES["eco"]) if all(c[k] == v for k, v in kw.items())]


def pat_idx(**kw):
    return [i for i, c in enumerate(FAMILIES["pat"]) if all(c[k] == v for k, v in kw.items())]


def methods(M):
    eco_paper = sorted(eco_idx(nf=0.125, gf=0.125, k=2, h=0), key=lambda i: FAMILIES["eco"][i]["g"])
    pat_paper = sorted(pat_idx(md=0.1, h=0), key=lambda i: -FAMILIES["pat"][i]["f"])
    fixp = sorted(range(len(FAMILIES["patfix"])), key=lambda i: -FAMILIES["patfix"][i]["p"])
    fixp = [i for i in fixp if FAMILIES["patfix"][i]["md"] == 0.0]
    n = lambda f: range(len(FAMILIES[f]))
    return {
        "Patience, fixed p (paper baseline)": ("patfix", fixp, "seq1d"),
        "Patience p=fT, min-delta (paper-fair)": ("pat", pat_paper, "seq1d"),
        "EcoStop (paper)": ("eco", eco_paper, "seq1d"),
        "Patience, Pareto (f, min-delta)": ("pat", pat_idx(h=0), "pareto"),
        "Patience, Pareto + schedule gate": ("pat", list(n("pat")), "pareto"),
        "EcoStop C1 (Pareto, no schedule gate)": ("eco", eco_idx(h=0), "pareto"),
        "EcoStop C1+C2 (Pareto + schedule gate)": ("eco", list(n("eco")), "pareto"),
        "LC-PFN C1+C2 (Pareto + schedule gate)": ("pfn", list(n("pfn")), "pareto"),
    }


def main():
    M = load_mats()
    strata = np.array([f"{d}|{f}" for d, f in zip(M["domain"], M["family"])])
    et = M["et"]
    rows, res = [], {}
    for rho in (0.15, 0.20):
        for name, (fam, idx, mode) in methods(M).items():
            r = run_method(M, fam, idx, mode, rho, strata)
            res[(name, rho)] = r
            grp = lambda m: r["L"][m].mean() if m.any() else np.nan
            rows.append({"rho": rho, "method": name, "risk": r["L"].mean(),
                         "risk_vision": grp(M["domain"] == "vision"), "risk_step": grp(M["family"] == "step"),
                         "risk_onecycle": grp(M["family"] == "onecycle"), "risk_cosw": grp(M["family"] == "cosine_warmup"),
                         "mean_saved": r["S"].mean(), "EW": r["ES"].sum() / et.sum() * 100,
                         "gap_pp": r["G"].mean()})
    df = pd.DataFrame(rows)
    # paired EW differences vs the strongest fair baseline
    diffs = []
    for rho in (0.15, 0.20):
        base = res[("Patience, Pareto + schedule gate", rho)]
        for name in methods(M):
            if name.startswith("EcoStop") or name.startswith("LC-PFN"):
                d, lo, hi = bootstrap_diff(res[(name, rho)], base, et)
                diffs.append({"rho": rho, "method": name, "EW_minus_PatienceParetoSched": d, "ci_lo": lo, "ci_hi": hi})
    dd = pd.DataFrame(diffs)
    os.makedirs("figures/tierA", exist_ok=True)
    df.to_csv("figures/tierA/phase1_oof.csv", index=False)
    dd.to_csv("figures/tierA/phase1_paired.csv", index=False)
    pickle.dump(res, open("cache/p1_res.pkl", "wb"))
    pd.set_option("display.width", 250, "display.max_colwidth", 44)
    print(df.round(3).to_string(index=False))
    print()
    print(dd.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
