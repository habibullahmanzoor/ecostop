"""P4: does seed-to-seed variability undermine the risk label? The label is WITHIN a run: V = 1[A_T - A_tau > eps] for the same
trajectory, so between-run differences in absolute accuracy (e.g. the SEMEION repeats) do not enter it directly. What can enter is
(i) test-set sampling noise in A_T - A_tau and (ii) seed variability of the gap itself. We measure both and re-run the pooled
out-of-fold calibration with a job-specific tolerance eps_j = max(0.5, sigma_gap,j), sigma_gap = seed s.d. of the gap A_T - A_tau
at the job's own most-aggressive certified stop is avoided; instead we use the seed s.d. of the final test accuracy A_T and of the
within-run best-minus-final difference, which are tolerance-independent."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import tierA
from echostop.calibration import forward_fill_audit

jobs, traces, dom = tierA.load()
names = list(jobs)
dom_arr = np.array([dom[k] for k in names])
rows = []
for k in names:
    A = [forward_fill_audit(df) for df in jobs[k]]
    fin = np.array([a[-1] for a in A])
    best_gap = np.array([np.nanmax(a) - a[-1] for a in A])          # headroom the final checkpoint leaves (>=0), noise-floor proxy
    rows.append(dict(job=k, dom=dom[k], seeds=len(A), sd_final=float(np.std(fin, ddof=1)) if len(A) > 1 else np.nan,
                     range_final=float(fin.max() - fin.min()), mean_best_gap=float(best_gap.mean())))
d = pd.DataFrame(rows)
print("seed dispersion of final test accuracy (pp), per job:")
for g, x in list(d.groupby("dom")) + [("all", d)]:
    print(f"  {g:8s} jobs {len(x):3d} | median sd {x.sd_final.median():.2f} | share sd > 0.5 pp: {(x.sd_final > 0.5).mean():.2f} | "
          f"share sd > 2 pp: {(x.sd_final > 2).mean():.2f} | max sd {x.sd_final.max():.1f}")
# job-specific tolerance eps_j = max(0.5, sd_final_j): re-run the pooled out-of-fold calibration
EPS = 0.5
pol = tierA.build_policies(None)
eps_j = np.maximum(EPS, d.sd_final.fillna(0).values)
def loss_matrix(plist, eps):
    P, J = len(plist), len(names)
    L, S, ES, ET = np.zeros((P, J)), np.zeros((P, J)), np.zeros((P, J)), np.zeros(J)
    for j, k in enumerate(names):
        for df, tr in zip(jobs[k], traces[k]):
            e = df["cumulative_energy_joules"].values; a = forward_fill_audit(df); ET[j] += e[-1]
            for i, (_, fn) in enumerate(plist):
                t = fn(df, tr)
                L[i, j] += float(a[-1] - a[t] > eps[j]); S[i, j] += 100 * (1 - e[t] / e[-1]); ES[i, j] += e[-1] - e[t]
        L[:, j] /= len(jobs[k]); S[:, j] /= len(jobs[k])
    return L, S, ES, ET
print("\npooled out-of-fold, Hoeffding-Bentkus, fixed eps=0.5 vs job-specific eps_j=max(0.5, seed sd of final accuracy):")
out = []
for fam in ("EcoStop", "Patience (p=f*T, min-delta 0.1)", "Patience (fixed p)"):
    for tag, eps in (("eps=0.5", np.full(len(names), EPS)), ("eps_j", eps_j)):
        L, S, ES, ET = loss_matrix(pol[fam], eps)
        for rho in (0.15, 0.20):
            r = tierA.crossfit(L, S, ES, ET, dom_arr, rho, 0.05)
            out.append(dict(rule=fam, tol=tag, rho=rho, risk=r["risk"], EW=r["ew"], vision_risk=r["risk_by_dom"]["vision"]))
o = pd.DataFrame(out)
print(o.round(3).to_string(index=False))
print("\njobs whose tolerance rises above 0.5 pp under eps_j:", int((eps_j > 0.5 + 1e-9).sum()), "of", len(names))
