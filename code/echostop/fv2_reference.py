"""Reference-run setting: the same job has been trained before (seeds for error bars, periodic retraining, re-runs).

Declared before running (2026-10-12):
  Setting   per job, the first seed (sorted file order) is the completed reference run; the other seeds are deployed.
            Savings and risk are measured on deployed runs only (the reference's energy is spent in every arm).
            Job-level loss = fraction of deployed seeds with A_T - A_tau > 0.5 pp.
  EcoStop-Ref  EcoStop's continuation score with the reference's final best validation accuracy as the asymptote:
            predicted gain g_t = B_ref_T - B_t; stop at the first t >= N = max(1, round(0.125 T)) with g_t <= delta.
            (Equivalent to s_t < gamma for a fixed sigma.) delta grid -1.0 .. +4.0 (conservative -> aggressive).
  Baselines Calibrated patience (p = fT, min-delta 0.1); EcoStop-SR (fv2_srgates); Reference-epoch: stop at
            t_ref + round(f T), t_ref = first epoch at which the reference's best validation accuracy was within
            0.5 pp of its final value, f from +0.5 down to -0.3.
  Certification pooled betting bound; dev out-of-fold; fresh calibrated on dev and applied unchanged.
  Comparison best saving at realised risk <= r*, paired job bootstrap.
  WIN BAR   on FRESH, EcoStop-Ref above BOTH patience and Reference-epoch with the 95% interval above 0 at
            r* = 0.10 or r* = 0.15. Secondary: fresh risk of EcoStop-Ref WITHOUT the scope card."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E
from echostop import tierA
from echostop.ablation import sim
from echostop.calibration import forward_fill_audit, precompute_score_trace
from echostop.framework_v2 import crossfit, select, wsr_pvalue
from echostop.fv2_srgates import boot, g_sr, k_sr

EPS = 0.5
DELTAS = np.round(np.arange(-1.0, 4.001, 0.2), 2)
FREF = np.round(np.arange(0.5, -0.301, -0.05), 2)
FPAT = [0.5, 0.4, 0.33, 0.3, 0.25, 0.2, 0.175, 0.15, 0.125, 0.1, 0.08, 0.06]
GRID = np.round(np.arange(0.03, 0.401, 0.01), 2)


def ref_stats(df):
    b = df["val_acc"].cummax().values
    return b[-1], int(np.argmax(b >= b[-1] - EPS))


def stops(df, tr, ref):
    T = len(df)
    bT_ref, t_ref = ref
    b = df["val_acc"].cummax().values
    N = max(1, round(0.125 * T))
    out = {}
    out["EcoStop-Ref"] = []
    for d in DELTAS:
        idx = np.where((np.arange(T) >= N) & (bT_ref - b <= d))[0]
        out["EcoStop-Ref"].append(int(idx[0]) if len(idx) else T - 1)
    out["Reference-epoch"] = [int(np.clip(t_ref + round(f * T), N, T - 1)) for f in FREF]
    out["Patience (p=fT, min-delta 0.1)"] = [tierA.patience_stop_md(df, max(1, round(f * T)), 0.1) for f in FPAT]
    out["EcoStop-SR"] = [sim(tr, g, T, g_sr(T), k_sr(T)) for g in tierA.GAMMAS]
    return out


def matrices(jobs, traces):
    names = list(jobs)
    fams = None
    M = {}
    for j, k in enumerate(names):
        dfs, trs = jobs[k], traces[k]
        ref = ref_stats(dfs[0])
        n = len(dfs) - 1
        for df, tr in zip(dfs[1:], trs[1:]):
            st = stops(df, tr, ref)
            if fams is None:
                fams = list(st)
                M = {f: [np.zeros((len(st[f]), len(names))) for _ in range(3)] + [np.zeros(len(names))] for f in fams}
            a, e = forward_fill_audit(df), df["cumulative_energy_joules"].values
            for f in fams:
                t = np.array(st[f])
                M[f][0][:, j] += (a[-1] - a[t] > EPS) / n
                M[f][1][:, j] += 100 * (1 - e[t] / e[-1]) / n
                M[f][2][:, j] += e[-1] - e[t]
                M[f][3][j] += e[-1]
    return {f: tuple(v) for f, v in M.items()}


def at_or_below(pts, r):
    c = [p for p in pts if p[0] <= r + 1e-12]
    return max(c, key=lambda p: p[1].sum()) if c else None


def main():
    cache = "cache/reference_mats.pkl"
    if os.path.exists(cache):
        D, F, dom, fdom, ok = pickle.load(open(cache, "rb"))
    else:
        dev, eco_d, domd = tierA.load()
        fresh, fdd = E.load_b1()
        eco_f = {k: [precompute_score_trace(df, EPS) for df in dfs] for k, dfs in fresh.items()}
        D, F = matrices(dev, eco_d), matrices(fresh, eco_f)
        dom = np.array([domd[k] for k in dev]); fdom = np.array([fdd[k] for k in fresh])
        flags = pd.read_csv("figures/tierA/tier1_scope_flags.csv").set_index("job")
        ok = np.array([bool(flags.loc["b1:" + n, "in_scope"]) for n in fresh])
        pickle.dump((D, F, dom, fdom, ok), open(cache, "wb"))
    etd, etf = D["EcoStop-Ref"][3], F["EcoStop-Ref"][3]
    rules = list(D)
    pts = {}
    for r in rules:
        L = D[r][0]
        dev_p, fr_p, fr_nc = [], [], []
        for rho in GRID:
            x = crossfit(D, dom, rho, lambda tr, te, L=L, r=r, rho=rho: [(te, None if (i := select(L[:, tr], rho, wsr_pvalue)) is None else (r, i))])
            dev_p.append((x["risk"], x["ES"], x["L"]))
            i = select(L, rho, wsr_pvalue)
            Lf, _, ESf, _ = F[r]
            lf = Lf[i] if i is not None else np.zeros(len(ok)); ef = ESf[i] if i is not None else np.zeros(len(ok))
            fr_p.append(((lf * ok).mean(), ef * ok, lf * ok))
            fr_nc.append((lf.mean(), ef, lf))
        pts[r] = (dev_p, fr_p, fr_nc)

    rng = np.random.default_rng(0)
    base = ["Patience (p=fT, min-delta 0.1)", "Reference-epoch", "EcoStop-SR"]
    for k, (setname, et) in enumerate((("DEV out-of-fold", etd), ("FRESH frozen, scope card for every method", etf))):
        print(f"\n=== {setname}: best EW saving (%) at realised risk <= r*; EcoStop-Ref minus each baseline (pp, 95% CI)")
        for rr in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
            a = at_or_below(pts["EcoStop-Ref"][k], rr)
            line = f"  r*={rr:.2f} EcoStop-Ref {100 * a[1].sum() / et.sum():5.1f} ({a[0]:.3f}) |" if a else f"  r*={rr:.2f} EcoStop-Ref n/a |"
            for bname in base:
                b = at_or_below(pts[bname][k], rr)
                if a is None or b is None:
                    line += f" {bname[:15]}: n/a |"; continue
                pt, lo, hi = boot(a[1] - b[1], et, rng)
                line += f" {bname[:15]} {100 * b[1].sum() / et.sum():5.1f} ({b[0]:.3f}) {pt:+5.1f} [{lo:+5.1f},{hi:+5.1f}] |"
            print(line)

    print("\n=== FRESH without the scope card (all 35 jobs): realised risk and EW at the dev-calibrated target rho")
    for rho in (0.10, 0.15, 0.20):
        i = int(np.argmin(np.abs(GRID - rho)))
        s = f"  rho={rho:.2f} |"
        for r in rules:
            risk, ef, lf = pts[r][2][i]
            vis = lf[fdom == "vision"].mean()
            s += f" {r[:15]} risk {risk:.3f} (vision {vis:.3f}, out-of-scope {lf[~ok].mean():.3f}) EW {100 * ef.sum() / etf.sum():4.1f} |"
        print(s)


if __name__ == "__main__":
    main()
