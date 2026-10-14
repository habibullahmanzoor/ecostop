"""P8: cost of failure. A violation (stopped checkpoint more than eps worse than the full run) is charged c full-run energies
(c = 1: the job is re-run to completion; c = 0.5: half a run; c = 3: three runs of debugging/re-tuning). Pooled out-of-fold, 259 jobs.
  saving   s  = energy saved / energy of full runs        (energy-weighted, as elsewhere)
  loss     v  = sum_j L_j * E_j / sum_j E_j               (violations weighted by the energy of the job they hit)
  net(c)   = s - c * v ; break-even c* = s / v  (the largest per-violation cost at which stopping still pays)
Methods: framework (betting + selection), LTT-HB calibrated patience, never stop."""
import itertools, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.pool_loader import load_pool
from echostop.c1_framework import DOMS, NK, RULES, SHORT
from echostop.framework_v2 import BETA, hb_p, select, wsr_pvalue

M, dom, grp, tag, et = load_pool()
J = len(dom)
idxmap = {r: np.round(np.linspace(0, M[r]["P"] - 1, NK)).astype(int) for r in RULES}
assigns = [dict(zip(DOMS, c)) for c in itertools.product(RULES, repeat=3)]
comp = []
for a in assigns:
    L, ES = np.zeros((NK, J)), np.zeros((NK, J))
    for d in DOMS:
        m = dom == d; r = a[d]
        L[:, m], ES[:, m] = M[r]["L"][idxmap[r]][:, m], M[r]["ES"][idxmap[r]][:, m]
    comp.append((L, ES))


def folds(rng, K=5):
    f = np.zeros(J, int)
    for d in DOMS:
        gs = np.unique(grp[dom == d]); rng.shuffle(gs); gf = {g: i % K for i, g in enumerate(gs)}
        for j in np.where(dom == d)[0]:
            f[j] = gf[grp[j]]
    return f


rng = np.random.default_rng(0)
rows = []
for rho in (0.10, 0.15, 0.20):
    acc = {m: (np.zeros((20, J)), np.zeros((20, J))) for m in ("framework", "LTT-HB calibrated patience", "LTT-HB curve rule")}
    for r in range(20):
        fold = folds(rng)
        for k in range(5):
            te, tr = np.where(fold == k)[0], np.where(fold != k)[0]
            best, bew = None, -1
            for i, (L, ES) in enumerate(comp):
                s = select(L[:, tr], rho, wsr_pvalue, beta=BETA / len(assigns))
                if s is not None and ES[s, tr].sum() / et[tr].sum() > bew:
                    best, bew = (i, s), ES[s, tr].sum() / et[tr].sum()
            if best:
                acc["framework"][0][r, te], acc["framework"][1][r, te] = comp[best[0]][0][best[1], te], comp[best[0]][1][best[1], te]
            for name, rule in (("LTT-HB calibrated patience", RULES[1]), ("LTT-HB curve rule", RULES[0])):
                i = select(M[rule]["L"][:, tr], rho, hb_p)
                if i is not None:
                    acc[name][0][r, te], acc[name][1][r, te] = M[rule]["L"][i, te], M[rule]["ES"][i, te]
    for name, (lo, eo) in acc.items():
        Lm, Em = lo.mean(0), eo.mean(0)
        s, v = Em.sum() / et.sum(), (Lm * et).sum() / et.sum()
        rows.append(dict(rho=rho, method=name, saving=100 * s, violation_share=100 * v, risk=Lm.mean(),
                         net_c05=100 * (s - 0.5 * v), net_c1=100 * (s - v), net_c3=100 * (s - 3 * v), breakeven_c=s / v if v > 0 else np.inf))
df = pd.DataFrame(rows)
df.to_csv("figures/tierA/break_even.csv", index=False)
pd.set_option("display.width", 200)
print(df.round(2).to_string(index=False))
