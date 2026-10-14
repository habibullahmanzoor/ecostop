"""Diagnose the C1 primary interval: why is the point estimate (-9.7) near the lower edge of [-14.1, +12.1]?
Compares (a) the pre-registered estimator (operating point re-chosen inside every bootstrap resample) with (b) fixed operating points
chosen on the full sample, (c) bias-corrected interval, and (d) the distribution of bootstrap differences."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import c1_eval as CE, c1_rules as CR

jobs, meta, _ = CE.load_logs("logs/c1/nlp")
names = sorted(jobs)
ds = np.array([k.split("|")[0] for k in names])
M = CE.matrices({k: jobs[k] for k in names}, False, "cache/none.pkl")
rng = np.random.default_rng(0)
W = CE.boot_weights(ds, rng, 10000)
W1 = np.ones((1, len(names)))
r = CR.PRIMARY_RISK
res = {}
for f in (CR.ECO, CR.PAT_MD):
    m = M[f]
    pt, ip = CE.front(W1, m["L"], m["ES"], m["et"], r)
    bt, ib = CE.front(W, m["L"], m["ES"], m["et"], r)
    res[f] = (pt[0], bt, int(ip[0]), ib)
pt = res[CR.ECO][0] - res[CR.PAT_MD][0]
d = res[CR.ECO][1] - res[CR.PAT_MD][1]
print(f"(a) pre-registered estimator: point {pt:+.2f}; bootstrap mean {d.mean():+.2f}, median {np.median(d):+.2f}; "
      f"2.5/97.5% [{np.percentile(d,2.5):+.2f}, {np.percentile(d,97.5):+.2f}]; share of resamples with diff>0: {(d>0).mean():.3f}")
print("    per-rule: point EW", {f: round(res[f][0], 2) for f in res}, "| bootstrap mean EW", {f: round(float(res[f][1].mean()), 2) for f in res})
print("    (the bootstrap re-chooses the best feasible setting in each resample; the maximum of noisy curves is biased upward, more so for the rule with the flatter/denser curve)")
# basic (reverse-percentile) bootstrap interval
lo_b, hi_b = 2 * pt - np.percentile(d, 97.5), 2 * pt - np.percentile(d, 2.5)
print(f"(c) basic bootstrap interval: [{lo_b:+.2f}, {hi_b:+.2f}]")
# BCa-lite: bias-correction only
from scipy.stats import norm
z0 = norm.ppf((d < pt).mean())
a, b = norm.cdf(2 * z0 + norm.ppf(0.025)), norm.cdf(2 * z0 + norm.ppf(0.975))
print(f"    bias-corrected percentile interval: [{np.percentile(d, 100*a):+.2f}, {np.percentile(d, 100*b):+.2f}] (z0={z0:+.2f})")
# (b) fixed operating points chosen on the full sample, resampled jobs only
ia, ib_ = res[CR.ECO][2], res[CR.PAT_MD][2]
def ew(f, i, w):
    m = M[f]; return 100 * (w @ m["ES"][i]) / (w @ m["et"])
db = ew(CR.ECO, ia, W) - ew(CR.PAT_MD, ib_, W)
dp = ew(CR.ECO, ia, W1)[0] - ew(CR.PAT_MD, ib_, W1)[0]
print(f"(b) fixed settings {M[CR.ECO]['labels'][ia]} vs {M[CR.PAT_MD]['labels'][ib_]}: point {dp:+.2f}, interval [{np.percentile(db,2.5):+.2f}, {np.percentile(db,97.5):+.2f}]; "
      f"realised risk SR {M[CR.ECO]['L'][ia].mean():.3f} vs patience {M[CR.PAT_MD]['L'][ib_].mean():.3f}")
# non-inferiority statement: what margin could be excluded?
print(f"(d) worst difference not excluded at 95% (lower limit of the pre-registered interval): {np.percentile(d,2.5):+.1f} pp; "
      f"a 3 pp non-inferiority margin would need the lower limit above -3: {'MET' if np.percentile(d,2.5) > -3 else 'NOT MET'}")
