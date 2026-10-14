"""Framework improvements evaluated on existing logs (CPU only).

1. Tighter certification. The paper certifies with a Hoeffding-Bentkus (HB) p-value. Job-level losses are fractions in
   [0, 1] (share of violating seeds) with low variance, the case where the Waudby-Smith & Ramdas betting bound
   (as used for bounded losses by Bates et al., RCPS) is tighter. Both p-values are valid for H0: E[loss] >= rho; both
   go through the same fixed-sequence test.
2. Certified rule selection per domain (portfolio): see portfolio_* below.

All comparisons reuse the cached per-job matrices and the same stratified folds as the paper (tierA)."""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.tierA import hb_pvalue

BETA = 0.05


# ------------------------------------------------------------------ p-values for H0: E[L] >= rho, L in [0, 1]
def wsr_pvalue(x: np.ndarray, rho: float, beta: float = BETA, seed: int = 0) -> float:
    """Betting (Waudby-Smith & Ramdas) p-value. Capital K_t = prod (1 - nu_i (x_i - rho)) is a nonnegative
    supermartingale under H0 (E[x] >= rho), so p = 1 / max_t K_t is valid by Ville's inequality. nu_i is predictable
    (computed from x_1..x_{i-1}) as in the RCPS implementation. The order of the exchangeable jobs is a fixed random
    permutation (seeded), chosen before looking at the losses."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return 1.0
    x = x[np.random.default_rng(seed).permutation(n)]
    i = np.arange(1, n + 1)
    muhat = (np.cumsum(x) + 0.5) / (1 + i)
    s2 = (np.cumsum((x - muhat) ** 2) + 0.25) / (1 + i)
    s2 = np.concatenate([[0.25], s2[:-1]])                       # predictable
    nu = np.minimum(np.sqrt(2 * np.log(1 / beta) / (n * s2)), 1.0)
    logk = np.cumsum(np.log(1 - nu * (x - rho)))
    return float(min(1.0, np.exp(-max(0.0, logk.max()))))


def hb_p(x, rho, beta=BETA, seed=0):
    return hb_pvalue(float(np.mean(x)), len(x), rho)


def select(loss: np.ndarray, rho: float, pfun, beta: float = BETA):
    """Fixed-sequence over thetas ordered conservative -> aggressive; last certified index or None."""
    last = None
    for i in range(loss.shape[0]):
        if pfun(loss[i], rho, beta) <= beta:
            last = i
        else:
            break
    return last


def folds_for(strata, seed, K=5):
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(strata), int)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % K
    return fold


def crossfit(mats, strata, rho, chooser, K=5, R=20, seed0=0):
    """Generic out-of-fold evaluation. chooser(train_idx) -> (family_key, theta_index) or None (run to completion).
    mats[family] = (loss, saved, es, et). Returns per-job arrays averaged over repeats."""
    any_f = next(iter(mats))
    J = mats[any_f][0].shape[1]
    et = mats[any_f][3]
    L, S, ES = np.zeros((R, J)), np.zeros((R, J)), np.zeros((R, J))
    for r in range(R):
        fold = folds_for(strata, seed0 + r, K)
        for f in range(K):
            te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
            for idx, pick in chooser(tr, te):
                if pick is None:
                    continue
                fam, i = pick
                loss, saved, es, _ = mats[fam]
                L[r, idx], S[r, idx], ES[r, idx] = loss[i, idx], saved[i, idx], es[i, idx]
    return {"L": L.mean(0), "S": S.mean(0), "ES": ES.mean(0), "risk": L.mean(), "mean_saved": S.mean(),
            "EW": 100 * ES.mean(0).sum() / et.sum(), "risk_by_rep": L.mean(1)}


def load():
    mats, names, dom = pickle.load(open("cache/tierA_mats_all.pkl", "rb"))
    return mats, list(names), np.array(dom)
