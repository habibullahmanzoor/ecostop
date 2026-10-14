"""Evaluation A of fv2_srgates.py done properly: fine rho grid (0.03..0.40), pooled betting certification.
For each risk level r*, each method's operating point is the calibration target whose realised risk is the largest
value <= r* ("best saving at or below risk r*"), so neither method is compared at a higher risk than the other.
Dev: out-of-fold. Fresh: calibrated on dev, applied unchanged, scope card for every method. Paired job bootstrap."""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.framework_v2 import crossfit, select, wsr_pvalue
from echostop.fv2_srgates import PAT, boot, pooled_chooser

D, F, dom, fdom, ok = pickle.load(open("cache/srgates_mats.pkl", "rb"))
etd, etf = D["EcoStop"][3], F["EcoStop"][3]
GRID = np.round(np.arange(0.03, 0.401, 0.01), 2)
RULES = ["EcoStop-SR", "EcoStop", "LC-PFN + SR gates", PAT]


def points(rule):
    L = D[rule][0]
    dev, fr = [], []
    for rho in GRID:
        x = crossfit(D, dom, rho, pooled_chooser(L, rule, rho))
        dev.append((x["risk"], x["ES"], x["L"]))
        i = select(L, rho, wsr_pvalue)
        Lf, _, ESf, _ = F[rule]
        lf = (Lf[i] if i is not None else np.zeros(len(ok))) * ok
        ef = (ESf[i] if i is not None else np.zeros(len(ok))) * ok
        fr.append((lf.mean(), ef, lf))
    return dev, fr


def at_or_below(pts, r):
    ok_ = [p for p in pts if p[0] <= r + 1e-12]
    if not ok_:
        return None
    return max(ok_, key=lambda p: p[1].sum())


P = {r: points(r) for r in RULES}
rng = np.random.default_rng(0)
for k, (setname, et) in enumerate((("dev (out-of-fold)", etd), ("fresh (frozen, scope card)", etf))):
    print(f"\n=== {setname}: best energy-weighted saving at realised risk <= r*, minus calibrated patience")
    for r in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
        b = at_or_below(P[PAT][k], r)
        line = f"  r*={r:.2f} patience EW {100 * b[1].sum() / et.sum():5.1f} (risk {b[0]:.3f}) |" if b else f"  r*={r:.2f} patience none |"
        for rule in ("EcoStop-SR", "EcoStop", "LC-PFN + SR gates"):
            a = at_or_below(P[rule][k], r)
            if a is None or b is None:
                line += f" {rule}: n/a |"; continue
            pt, lo, hi = boot(a[1] - b[1], et, rng)
            line += f" {rule} {100 * a[1].sum() / et.sum():5.1f} ({a[0]:.3f}) {pt:+5.1f} [{lo:+5.1f},{hi:+5.1f}] |"
        print(line)
pickle.dump({r: ([(p[0], p[1]) for p in P[r][0]], [(p[0], p[1]) for p in P[r][1]]) for r in RULES},
            open("cache/srgates_points.pkl", "wb"))
