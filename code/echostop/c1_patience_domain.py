"""Calibrated patience: pooled vs per-domain (Mondrian) calibration, out-of-fold on all 259 jobs, dataset-grouped folds.
EXPLORATORY (development-style): all data seen. Same folds/seeds as c1_framework_oof. Both bounds (HB, betting)."""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E, c1_eval as CE, tierA
from echostop.c1_framework import DOMS, RULES, mats
from echostop.calibration import precompute_score_trace
from echostop.framework_v2 import hb_p, select, wsr_pvalue
dev, eco_d, dom_d = tierA.load(); b1, dom_b = E.load_b1()
eco_b = {k: [precompute_score_trace(df, 0.5) for df in v] for k, v in b1.items()}
pfn_d = pickle.load(open("cache/lcpfn_traces.pkl", "rb")); pfn_b = pickle.load(open("cache/b1_lcpfn_traces.pkl", "rb"))
c1, _, _ = CE.load_logs("logs/c1/nlp"); pfn_c = pickle.load(open("cache/c1_lcpfn_traces.pkl", "rb"))
eco_c = {k: [precompute_score_trace(df, 0.5) for df in v] for k, v in c1.items()}
jobs, eco, pfn, dom, grp = {}, {}, {}, [], []
for k, v in dev.items(): n=f"dev:{k}"; jobs[n],eco[n],pfn[n]=v,eco_d[k],pfn_d[k]; dom.append(dom_d[k]); grp.append(n)
for k, v in b1.items(): n=f"b1:{k}"; jobs[n],eco[n],pfn[n]=v,eco_b[k],pfn_b[k]; dom.append(dom_b[k]); grp.append(n)
for k in sorted(c1): n=f"c1:{k}"; jobs[n],eco[n],pfn[n]=c1[k],eco_c[k],pfn_c[k]; dom.append("nlp"); grp.append("c1:"+k.split("|")[0])
dom, grp = np.array(dom), np.array(grp); J = len(dom)
r = RULES[1]; M = mats(jobs, eco, pfn)[r]; L, ES, et = M["L"], M["ES"], M["et"]
def folds(rng, K=5):
    f = np.zeros(J, int)
    for d in DOMS:
        gs = np.unique(grp[dom == d]); rng.shuffle(gs); gf = {g: i % K for i, g in enumerate(gs)}
        for j in np.where(dom == d)[0]: f[j] = gf[grp[j]]
    return f
rng = np.random.default_rng(0); R = 20
rows = []
uniq = np.unique(grp); gi = np.array([np.searchsorted(uniq, g) for g in grp])
wr = np.random.default_rng(1).multinomial(len(uniq), np.ones(len(uniq))/len(uniq), size=5000)[:, gi].astype(float)
store = {}
for rho in (0.10, 0.15, 0.20):
    acc = {m: (np.zeros((R, J)), np.zeros((R, J)), np.zeros(R)) for m in
           ("HB pooled", "HB per-domain", "betting pooled", "betting per-domain")}
    for rep in range(R):
        fold = folds(rng)
        for f in range(5):
            te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
            for tag, pf in (("HB", hb_p), ("betting", wsr_pvalue)):
                i = select(L[:, tr], rho, pf)
                if i is not None:
                    acc[f"{tag} pooled"][0][rep, te], acc[f"{tag} pooled"][1][rep, te] = L[i, te], ES[i, te]
                for d in DOMS:
                    trd, ted = tr[dom[tr] == d], te[dom[te] == d]
                    if len(ted) == 0: continue
                    k = select(L[:, trd], rho, pf)
                    if k is not None:
                        acc[f"{tag} per-domain"][0][rep, ted], acc[f"{tag} per-domain"][1][rep, ted] = L[k, ted], ES[k, ted]
    for m, (Lr, Er, _) in acc.items():
        Lm, Em = Lr.mean(0), Er.mean(0); store[(rho, m)] = Em
        rows.append(dict(rho=rho, method=m, risk=Lm.mean(), EW=100*Em.sum()/et.sum(),
                         **{f"risk_{d}": Lm[dom == d].mean() for d in DOMS}, **{f"EW_{d}": 100*Em[dom == d].sum()/et[dom == d].sum() for d in DOMS}))
df = pd.DataFrame(rows); pd.set_option("display.width", 230)
print(df.round(3).to_string(index=False))
print("\nper-domain minus pooled calibration, EW pp (dataset-cluster bootstrap):")
for rho in (0.10, 0.15, 0.20):
    for tag in ("HB", "betting"):
        d = store[(rho, f"{tag} per-domain")] - store[(rho, f"{tag} pooled")]
        bs = (wr @ d) / (wr @ et) * 100
        print(f"  rho={rho:.2f} {tag:8s}: {100*d.sum()/et.sum():+6.2f} [{np.percentile(bs,2.5):+6.2f}, {np.percentile(bs,97.5):+6.2f}]")
