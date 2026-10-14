"""C1 rule comparison with a CROSS-FITTED estimator (post hoc, reported as a sensitivity analysis next to the pre-registered one).
For each of R repeats the 41 datasets are split into K folds; in each fold every rule's setting is chosen on the other folds as the
most energy-saving setting with empirical job-level risk <= 0.10, then evaluated on the held-out fold. Realised risk and
energy-weighted saving are accumulated over held-out jobs, so the estimate is out-of-sample and its realised risk is reported.
Interval: bootstrap over datasets of the per-job held-out energy saved (selection noise is already inside the cross-fit)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import c1_eval as CE, c1_rules as CR

jobs, meta, _ = CE.load_logs("logs/c1/nlp")
names = sorted(jobs)
ds = np.array([k.split("|")[0] for k in names])
M = CE.matrices({k: jobs[k] for k in names}, False, "cache/none.pkl")
uds = np.unique(ds)
J = len(names)
R, K, CAP = 100, 5, CR.PRIMARY_RISK
rng = np.random.default_rng(0)
rules = [CR.ECO, CR.PAT_MD, CR.ECO_P, CR.PAT]
acc = {f: (np.zeros((R, J)), np.zeros((R, J))) for f in rules}       # held-out (loss, energy saved) per repeat
for r in range(R):
    perm = rng.permutation(uds)
    fold_of = {d: i % K for i, d in enumerate(perm)}
    fold = np.array([fold_of[d] for d in ds])
    for k in range(K):
        te, tr = np.where(fold == k)[0], np.where(fold != k)[0]
        for f in rules:
            L, ES = M[f]["L"], M[f]["ES"]
            risk = L[:, tr].mean(1)
            ok = np.where(risk <= CAP + 1e-12)[0]
            if len(ok) == 0:
                continue
            i = ok[np.argmax(ES[ok][:, tr].sum(1))]
            acc[f][0][r, te], acc[f][1][r, te] = L[i, te], ES[i, te]
et = M[CR.ECO]["et"]
W = CE.boot_weights(ds, np.random.default_rng(1), 10000)
print(f"cross-fitted, {R} repeats x {K} dataset-grouped folds, risk cap {CAP}")
cf = {}
for f in rules:
    Lm, Em = acc[f][0].mean(0), acc[f][1].mean(0)
    cf[f] = Em
    print(f"  {f:34s} realised risk {Lm.mean():.3f} | EW {100*Em.sum()/et.sum():5.1f}%")
for f in rules[2:] + [CR.ECO]:
    base = CR.PAT_MD
    if f == base:
        continue
    d = cf[f] - cf[base]
    bs = (W @ d) / (W @ et) * 100
    print(f"  {f} minus {base}: {100*d.sum()/et.sum():+6.2f}  95% [{np.percentile(bs,2.5):+6.2f}, {np.percentile(bs,97.5):+6.2f}]")
# repeat-to-repeat spread of the point estimate
pt = [100 * (acc[CR.ECO][1][r].sum() - acc[CR.PAT_MD][1][r].sum()) / et.sum() for r in range(R)]
print(f"  EcoStop-SR minus patience across repeats: mean {np.mean(pt):+.2f}, sd {np.std(pt):.2f}, range [{min(pt):+.1f}, {max(pt):+.1f}]")
