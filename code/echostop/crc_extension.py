"""P7 lever: would a less conservative calibrator recover savings? Conformal risk control (Angelopoulos et al. 2024) controls the
EXPECTED loss, E[L(theta_hat)] <= rho, not a high-probability statement; its finite-sample guarantee needs monotone losses, which
our violation loss does not satisfy, so we use its fixed-sequence form (certify the largest index all of whose predecessors pass
(n/(n+1)) R_hat + 1/(n+1) <= rho). Compared with Hoeffding-Bentkus and betting certification on the pooled 259-job set,
dataset-grouped folds stratified by domain (20 repeats), plus 1,000 random 70/30 splits for the share of splits whose held-out
risk exceeds rho."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.pool_loader import load_pool
from echostop.c1_framework import RULES, SHORT
from echostop.framework_v2 import hb_p, wsr_pvalue

M, dom, grp, tag, et = load_pool()
J = len(dom)
DOMS = ["nlp", "tabular", "vision"]


def pick(Ltr, rho, method):
    n = Ltr.shape[1]
    last = None
    for i in range(Ltr.shape[0]):
        if method == "CRC":
            ok = (n / (n + 1)) * Ltr[i].mean() + 1.0 / (n + 1) <= rho
        elif method == "HB":
            ok = hb_p(Ltr[i], rho, 0.05) <= 0.05
        else:
            ok = wsr_pvalue(Ltr[i], rho, 0.05) <= 0.05
        if ok:
            last = i
        else:
            break
    return last


def folds(rng, K=5):
    f = np.zeros(J, int)
    for d in DOMS:
        gs = np.unique(grp[dom == d]); rng.shuffle(gs); gf = {g: i % K for i, g in enumerate(gs)}
        for j in np.where(dom == d)[0]:
            f[j] = gf[grp[j]]
    return f


rng = np.random.default_rng(0)
rows = []
for rule in (RULES[1], RULES[0]):
    L, ES = M[rule]["L"], M[rule]["ES"]
    for rho in (0.10, 0.15, 0.20):
        for method in ("HB", "betting", "CRC"):
            lo, eo = np.zeros((20, J)), np.zeros((20, J))
            for r in range(20):
                fold = folds(rng)
                for k in range(5):
                    te, tr = np.where(fold == k)[0], np.where(fold != k)[0]
                    i = pick(L[:, tr], rho, method)
                    if i is not None:
                        lo[r, te], eo[r, te] = L[i, te], ES[i, te]
            # random 70/30 dataset-grouped splits for the exceed rate
            exceed, nsp = 0, 300
            for _ in range(nsp):
                te = np.zeros(J, bool)
                for d in DOMS:
                    g = np.unique(grp[dom == d]); te |= (dom == d) & np.isin(grp, rng.choice(g, max(1, round(0.3 * len(g))), replace=False))
                i = pick(L[:, ~te], rho, method)
                exceed += (L[i, te].mean() > rho) if i is not None else 0
            Lm, Em = lo.mean(0), eo.mean(0)
            rows.append(dict(rule=SHORT[rule], rho=rho, method=method, risk=Lm.mean(), EW=100 * Em.sum() / et.sum(),
                             vision_risk=Lm[dom == "vision"].mean(), exceed=exceed / nsp))
df = pd.DataFrame(rows)
df.to_csv("figures/tierA/crc_extension.csv", index=False)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
