"""Per-domain risk-saving frontiers, EcoStop-SR vs calibrated patience, with job-bootstrap intervals.

Frontier F_m(t) = best energy-weighted saving of rule m over its parameter grid among settings whose realised risk
in that domain is <= t. Calibration-free: it compares the detectors, as ROC curves compare scores. Each method's
best setting is chosen on the same data, so a larger grid gives more chances; the robustness row thins EcoStop's
25-value grid to 12 values (every other one), the size of patience's grid.
Bootstrap: resample jobs within the domain with replacement (B = 2000), recompute both frontiers, take differences.
Combined row: energy-weighted over NLP + vision + tabular, each domain at its own risk <= t (what per-domain
calibration would realise)."""
import pickle

import numpy as np

D, F, dom, fdom, ok = pickle.load(open("cache/srgates_mats.pkl", "rb"))
PAT = "Patience (p=f*T, min-delta 0.1)"
TS = (0.05, 0.10, 0.15, 0.20)
B = 2000


def frontier(L, ES, et, t):
    R = L.mean(1)
    W = 100 * ES.sum(1) / et.sum()
    c = R <= t + 1e-12
    return W[c].max() if c.any() else 0.0


def front_es(L, ES, t):
    """energy saved (J) of the frontier setting, for combining domains"""
    R = L.mean(1)
    c = np.where(R <= t + 1e-12)[0]
    return ES[c].sum(1).max() if len(c) else 0.0


def diffs(M, mask, rule, thin=False, rng=None, combined=False, domarr=None):
    L1, _, E1, et = M[rule]
    L0, _, E0, _ = M[PAT]
    if thin:
        L1, E1 = L1[::2], E1[::2]
    idx_all = np.where(mask)[0]
    out = np.zeros((B + 1, len(TS)))
    for b in range(B + 1):
        if combined:
            pick = np.concatenate([rng.choice(np.where(mask & (domarr == d))[0], (mask & (domarr == d)).sum())
                                   if b else np.where(mask & (domarr == d))[0] for d in ("nlp", "tabular", "vision")])
            for k, t in enumerate(TS):
                s1 = s0 = 0.0
                for d in ("nlp", "tabular", "vision"):
                    p = pick[domarr[pick] == d]
                    s1 += front_es(L1[:, p], E1[:, p], t); s0 += front_es(L0[:, p], E0[:, p], t)
                out[b, k] = 100 * (s1 - s0) / et[pick].sum()
        else:
            p = rng.choice(idx_all, len(idx_all)) if b else idx_all
            for k, t in enumerate(TS):
                out[b, k] = frontier(L1[:, p], E1[:, p], et[p], t) - frontier(L0[:, p], E0[:, p], et[p], t)
    return out


rng = np.random.default_rng(0)
print("EcoStop-SR minus calibrated patience: frontier difference in energy-weighted pp at domain risk <= t, [95% CI]")
print("t =", "      ".join(f"{t:.2f}" for t in TS))
for setname, M, darr, extra in (("DEV", D, dom, np.ones(len(dom), bool)), ("FRESH", F, fdom, ok)):
    for d in ("nlp", "vision"):
        for thin in (False, True):
            o = diffs(M, (darr == d) & extra, "EcoStop-SR", thin, rng)
            lab = f"{setname:5s} {d:6s}{' (EcoStop grid thinned to 12)' if thin else ''}"
            print(f"  {lab:46s}", "  ".join(f"{o[0, k]:+5.1f} [{np.percentile(o[1:, k], 2.5):+5.1f},{np.percentile(o[1:, k], 97.5):+5.1f}]" for k in range(len(TS))))
    for thin in (False, True):
        o = diffs(M, extra, "EcoStop-SR", thin, rng, combined=True, domarr=darr)
        lab = f"{setname:5s} ALL, per-domain risk{' (thinned)' if thin else ''}"
        print(f"  {lab:46s}", "  ".join(f"{o[0, k]:+5.1f} [{np.percentile(o[1:, k], 2.5):+5.1f},{np.percentile(o[1:, k], 97.5):+5.1f}]" for k in range(len(TS))))
