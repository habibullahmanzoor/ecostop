"""Pooled out-of-fold evaluation of the framework on ALL 259 jobs (142 dev + 35 B1 + 82 C1), folds grouped by DATASET
(both arms of a C1 dataset stay together), stratified by domain. DEVELOPMENT-style evidence (all data was seen by the
analyst); it is not a held-out confirmation and is reported as such.
Methods, each calibrated on the training folds at the same rho and beta = 0.05:
  LTT-HB (one rule)      Hoeffding-Bentkus, fixed-sequence, rules: EcoStop / calibrated patience / LC-PFN + gates
  betting only (one rule) the betting bound, same rules
  EcoStop framework      betting + certified per-domain selection (27 assignments, beta/27)
Validity: 1,000 random 70/30 group splits; share of splits whose held-out risk exceeds rho (guarantee: <= 5%)."""
from __future__ import annotations

import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E
from echostop import c1_eval as CE
from echostop import tierA
from echostop.c1_framework import DOMS, NK, RULES, SHORT, mats
from echostop.calibration import precompute_score_trace
from echostop.framework_v2 import BETA, hb_p, select, wsr_pvalue

RHOS = (0.10, 0.15, 0.20)
R, K = 20, 5


def main():
    dev, eco_d, dom_d = tierA.load()
    b1, dom_b = E.load_b1()
    eco_b = {k: [precompute_score_trace(df, 0.5) for df in v] for k, v in b1.items()}
    pfn_d = pickle.load(open("cache/lcpfn_traces.pkl", "rb"))
    pfn_b = pickle.load(open("cache/b1_lcpfn_traces.pkl", "rb"))
    c1, _, _ = CE.load_logs("logs/c1/nlp")
    pfn_c = pickle.load(open("cache/c1_lcpfn_traces.pkl", "rb"))
    eco_c = {k: [precompute_score_trace(df, 0.5) for df in v] for k, v in c1.items()}
    jobs, eco, pfn, dom, grp = {}, {}, {}, [], []
    for k, v in dev.items():
        n = f"dev:{k}"; jobs[n], eco[n], pfn[n] = v, eco_d[k], pfn_d[k]; dom.append(dom_d[k]); grp.append(n)
    for k, v in b1.items():
        n = f"b1:{k}"; jobs[n], eco[n], pfn[n] = v, eco_b[k], pfn_b[k]; dom.append(dom_b[k]); grp.append(n)
    for k in sorted(c1):
        n = f"c1:{k}"; jobs[n], eco[n], pfn[n] = c1[k], eco_c[k], pfn_c[k]; dom.append("nlp"); grp.append("c1:" + k.split("|")[0])
    dom, grp = np.array(dom), np.array(grp)
    M = mats(jobs, eco, pfn)
    J = len(dom)
    et = M[RULES[0]]["et"]
    print(f"{J} jobs; by domain {dict(zip(*np.unique(dom, return_counts=True)))}; {len(np.unique(grp))} dataset groups")
    idxmap = {r: np.round(np.linspace(0, M[r]["P"] - 1, NK)).astype(int) for r in RULES}
    assigns = [dict(zip(DOMS, c)) for c in itertools.product(RULES, repeat=3)]
    comp = []
    for a in assigns:
        L, ES = np.zeros((NK, J)), np.zeros((NK, J))
        for d in DOMS:
            m = dom == d
            r = a[d]
            L[:, m], ES[:, m] = M[r]["L"][idxmap[r]][:, m], M[r]["ES"][idxmap[r]][:, m]
        comp.append((L, ES))

    def folds(rng, K):
        fold = np.zeros(J, int)
        for d in DOMS:
            gs = np.unique(grp[dom == d]); rng.shuffle(gs)
            gf = {g: i % K for i, g in enumerate(gs)}
            for j in np.where(dom == d)[0]:
                fold[j] = gf[grp[j]]
        return fold

    def apply_methods(tr, rho):
        """returns {method: (loss_row[J], es_row[J])} chosen using only training jobs tr"""
        out = {}
        best, best_ew = None, -1.0
        for i, (L, ES) in enumerate(comp):
            k = select(L[:, tr], rho, wsr_pvalue, beta=BETA / len(assigns))
            if k is not None:
                ew = ES[k, tr].sum() / et[tr].sum()
                if ew > best_ew:
                    best, best_ew = (i, k), ew
        z = np.zeros(J)
        out["EcoStop framework"] = (comp[best[0]][0][best[1]], comp[best[0]][1][best[1]]) if best else (z, z)
        for r in RULES:
            for tag, pf in (("LTT-HB", hb_p), ("betting only", wsr_pvalue)):
                i = select(M[r]["L"][:, tr], rho, pf)
                out[f"{tag}, {SHORT[r]}"] = (M[r]["L"][i], M[r]["ES"][i]) if i is not None else (z, z)
        return out

    rng = np.random.default_rng(0)
    res = {}
    for rho in RHOS:
        acc = {}
        for rep in range(R):
            fold = folds(rng, K)
            for f in range(K):
                te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
                for m, (L, ES) in apply_methods(tr, rho).items():
                    a = acc.setdefault(m, (np.zeros((R, J)), np.zeros((R, J))))
                    a[0][rep, te], a[1][rep, te] = L[te], ES[te]
        res[rho] = {m: (a[0].mean(0), a[1].mean(0)) for m, a in acc.items()}
        print("done", rho, flush=True)

    uniq = np.unique(grp)
    gi = np.array([np.searchsorted(uniq, g) for g in grp])
    rows = []
    for rho in RHOS:
        for m, (L, ES) in res[rho].items():
            rows.append(dict(rho=rho, method=m, risk=L.mean(), EW=100 * ES.sum() / et.sum(),
                             **{f"EW_{d}": 100 * ES[dom == d].sum() / et[dom == d].sum() for d in DOMS},
                             **{f"risk_{d}": L[dom == d].mean() for d in DOMS}))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240, "display.max_colwidth", 38)
    print(df.round(3).to_string(index=False))
    df.to_csv("figures/tierA/c1_framework_oof.csv", index=False)

    diffs = []
    print("\nFramework minus single-rule baselines, EW pp (bootstrap over dataset groups, 5,000):")
    wr =np.random.default_rng(1).multinomial(len(uniq), np.ones(len(uniq)) / len(uniq), size=5000)[:, gi].astype(float)
    for rho in RHOS:
        fL, fE = res[rho]["EcoStop framework"]
        for m in [x for x in res[rho] if x != "EcoStop framework"]:
            d = fE - res[rho][m][1]
            bs = (wr @ d) / (wr @ et) * 100
            print(f"  rho={rho:.2f} vs {m:34s}: {100 * d.sum() / et.sum():+6.2f} [{np.percentile(bs, 2.5):+6.2f}, {np.percentile(bs, 97.5):+6.2f}]")
            diffs.append(dict(rho=rho, baseline=m, diff=100 * d.sum() / et.sum(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    pd.DataFrame(diffs).to_csv("figures/tierA/c1_framework_oof_diffs.csv", index=False)

    print("\nValidity: share of 1,000 random 70/30 group splits whose held-out risk exceeds rho (must be <= 0.05):")
    rng = np.random.default_rng(2)
    splits = []
    for _ in range(1000):
        te = np.zeros(J, bool)
        for d in DOMS:
            gs = np.unique(grp[dom == d])
            pick = rng.choice(gs, max(1, round(0.3 * len(gs))), replace=False)
            te |= (dom == d) & np.isin(grp, pick)
        splits.append(te)
    val_rows = []
    for rho in (0.15, 0.20):
        cnt = {}
        for te in splits:
            tr = np.where(~te)[0]
            for m, (L, ES) in apply_methods(tr, rho).items():
                cnt[m] = cnt.get(m, 0) + (L[te].mean() > rho)
        print(f"  rho={rho:.2f}: " + "; ".join(f"{m}: {c / len(splits):.3f}" for m, c in cnt.items()))
        val_rows += [dict(rho=rho, method=m, exceed_rate=c / len(splits)) for m, c in cnt.items()]
    pd.DataFrame(val_rows).to_csv("figures/tierA/c1_framework_oof_validity.csv", index=False)


if __name__ == "__main__":
    main()
