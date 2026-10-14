"""Quick re-analysis of existing logs to size the cheapest improvements."""
import sys, os, pickle
sys.path.insert(0, r"C:\python experiments\echostop")
os.chdir(r"C:\python experiments\echostop")
import numpy as np
from scipy.stats import beta as beta_dist
from echostop.analyze_pilot import discover_runs
from echostop.calibration import precompute_score_trace, simulate_from_trace, audited_loss, forward_fill_audit
from echostop.run_calibration_demo2 import discover_tabular_runs
from echostop.run_calibration_final_split import discover_nlp_runs, stratified_split, GAMMA_GRID
from echostop.gating_tests import patience_stop

EPS = 0.5
cache = os.path.join(os.path.dirname(__file__), "traces.pkl")
vision = {f"{d}_{m}": dfs for (d, m), dfs in discover_runs().items()}
tab = discover_tabular_runs(); nlp = discover_nlp_runs()
jobs = {**vision, **tab, **nlp}
dom = {k: ("nlp" if k in nlp else "vision" if k in vision else "tabular") for k in jobs}
print("jobs", len(jobs), {d: sum(v == d for v in dom.values()) for d in ("vision", "tabular", "nlp")})
if os.path.exists(cache):
    traces = pickle.load(open(cache, "rb"))
else:
    traces = {k: [precompute_score_trace(df, EPS) for df in dfs] for k, dfs in jobs.items()}
    pickle.dump(traces, open(cache, "wb"))

def run_stats(k, stop_fn):
    out = []
    for df, tr in zip(jobs[k], traces[k]):
        t = stop_fn(df, tr)
        e = df["cumulative_energy_joules"].values
        out.append((audited_loss(df, t, EPS), 100 * (1 - e[t] / e[-1]), e[t], e[-1],
                    forward_fill_audit(df)[-1] - forward_fill_audit(df)[t]))
    return np.array(out)

def eco(g):  return lambda df, tr: simulate_from_trace(tr, g, len(df))
def pat(p):  return lambda df, tr: patience_stop(df, p)[0]
def pat_frac(f): return lambda df, tr: patience_stop(df, max(1, round(f * len(df))))[0]

def job_losses(stop_fn, keys):
    maj, mean, gapm, sav, eT, eS = [], [], [], [], 0, 0
    for k in keys:
        s = run_stats(k, stop_fn)
        maj.append(int(s[:, 0].mean() > 0.5)); mean.append(s[:, 0].mean())
        gapm.append(int(s[:, 4].mean() > EPS)); sav.append(s[:, 1].mean())
        eS += (s[:, 3] - s[:, 2]).sum(); eT += s[:, 3].sum()
    return np.array(maj), np.array(mean), np.array(gapm), np.array(sav), 100 * eS / eT

calib, evalp = stratified_split(dict(jobs), 0, 12)
allk = list(jobs)
print("\n== gamma=0.54 on all 142 split-eligible jobs ==")
maj, mean, gapm, sav, ew = job_losses(eco(0.54), allk)
print(f"majority-vote viol {maj.sum()}/{len(allk)}  mean-of-seeds risk {mean.mean():.3f}  "
      f"mean-gap>eps viol {gapm.sum()}/{len(allk)}  mean saved {sav.mean():.1f}%  energy-weighted saved {ew:.1f}%")
for d in ("vision", "tabular", "nlp"):
    ks = [k for k in allk if dom[k] == d]
    m2, me2, g2, s2, ew2 = job_losses(eco(0.54), ks)
    print(f"  {d:8s} n={len(ks):3d} maj {m2.sum()} meanrisk {me2.mean():.3f} gapviol {g2.sum()} saved {s2.mean():.1f}% ew {ew2:.1f}%")
mc, mec, gc, _, _ = job_losses(eco(0.54), calib)
print(f"calibration pool only: majority {mc.sum()}/{len(calib)}  mean-gap {gc.sum()}/{len(calib)}")

# --- certification bounds on the calibration pool, per gamma (monotone grid) ---
def cp_upper(k, n, delta):  # exact binomial (Clopper-Pearson) one-sided upper
    return 1.0 if k == n else beta_dist.ppf(1 - delta, k + 1, n - k)

print("\n== certification: Hoeffding+Bonferroni(beta=.2) vs fixed-sequence binomial(beta=.05 / .10) ==")
grid = GAMMA_GRID
rows = []
for g in grid:
    m, me, *_ = job_losses(eco(g), calib)
    rows.append((g, m.sum(), me.mean()))
n = len(calib)
for g, k, rmean in rows:
    hb = k / n + np.sqrt(np.log(len(grid) / 0.2) / (2 * n))
    print(f"gamma={g:.2f} maj-viol={k:3d} hoeff_bonf(.2)={hb:.3f} CP_fixedseq(.05)={cp_upper(k, n, .05):.3f} "
          f"CP(.10)={cp_upper(k, n, .10):.3f} meanrisk={rmean:.3f} hoeff_mean(.05)={rmean + np.sqrt(np.log(1/.05)/(2*n)):.3f}")

# --- matched patience baselines (calibrated on same pool, majority vote, Hoeffding+Bonf rho=.2) ---
print("\n== patience variants, all 142 jobs ==")
for name, fn in [("fixed p=10", pat(10)), ("fixed p=5", pat(5)), ("fixed p=3", pat(3))] + \
        [(f"frac {f}", pat_frac(f)) for f in (0.1, 0.125, 0.15, 0.2, 0.25, 0.3)]:
    m, me, g, s, ew = job_losses(fn, allk)
    mc, *_ = job_losses(fn, calib)
    print(f"{name:12s} calib-maj {mc.sum():3d}/{len(calib)}  all-maj {m.sum():3d}  meanrisk {me.mean():.3f} "
          f"saved {s.mean():.1f}%  ew {ew:.1f}%")
