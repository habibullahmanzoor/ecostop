"""A5 (quantitative ablation) and A6 (sensitivity). Every variant is re-fit on the
logged trajectories, calibrated with the same fixed-sequence HB wrapper, and
scored out-of-fold with tierA.crossfit. CPU only."""
from __future__ import annotations

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import tierA
from echostop.methods import _exp_saturation

OUT = "figures/tierA"
GAMMAS = tierA.GAMMAS


@dataclass(frozen=True)
class Cfg:
    eps: float = 0.5
    window: int | None = 15        # None = full history
    sigma: str = "cov"             # cov | resid | point
    fallback: bool = True          # a_max-pinned fit -> "converged" reading
    safeguard_frac: float | None = 0.125  # None -> fixed 5 epochs
    flat_tol: float = 0.10
    use_curve: bool = True


def fit(t, q, cfg: Cfg):
    if len(t) < 4:
        return None
    if cfg.window is not None and len(t) > cfg.window:
        t, q = t[-cfg.window:], q[-cfg.window:]
    qm = q.max()
    a0 = min(qm + 1.0, 100.0)
    try:
        popt, pcov = curve_fit(_exp_saturation, t, q, p0=[a0, max(a0 - q[0], 1.0), 0.1],
                               bounds=([0, 0, 1e-4], [100.0, 200.0, 5.0]), maxfev=5000)
    except RuntimeError:
        return None
    a, _, c = popt[0], popt[1], popt[2]
    pinned = a >= 100.0 - 0.5
    if pinned and cfg.fallback:
        return qm, max(float(np.std(q - qm)), 0.05)
    a = max(a, qm)
    if cfg.sigma == "point":
        s = 0.05
    elif cfg.sigma == "resid":
        s = float(np.std(q - _exp_saturation(t, *popt)))
    else:
        s = np.sqrt(pcov[0, 0]) if np.isfinite(pcov[0, 0]) else 100.0
    return a, max(float(s), 0.05)


def trace_for(df: pd.DataFrame, cfg: Cfg):
    T = len(df)
    n_min = max(2, round(cfg.safeguard_frac * T)) if cfg.safeguard_frac is not None else 5
    t_all, q_all, b = df["epoch"].values.astype(float), df["val_acc"].values.astype(float), df["B_t"].values
    out = []
    for t in range(T):
        flat = 0
        for s in range(t, 0, -1):
            if b[s] - b[s - 1] > cfg.flat_tol:
                break
            flat += 1
        if t < n_min:
            out.append({"t": t, "s_t": None, "epochs_flat": None}); continue
        if not cfg.use_curve:
            out.append({"t": t, "s_t": 0.0, "epochs_flat": flat}); continue
        f = fit(t_all[: t + 1], q_all[: t + 1], cfg)
        if f is None:
            out.append({"t": t, "s_t": None, "epochs_flat": None}); continue
        a, sg = f
        out.append({"t": t, "s_t": float(1 - norm.cdf(cfg.eps, loc=a - b[t], scale=sg)), "epochs_flat": flat})
    return out


def _job(args):
    dfs, cfg = args
    return [trace_for(df, cfg) for df in dfs]


def traces_for_cfg(jobs, cfg, pool):
    names = list(jobs)
    res = list(pool.map(_job, [(jobs[k], cfg) for k in names], chunksize=2))
    return dict(zip(names, res))


def sim(tr, g, T, gate, k):
    consec = 0
    for r in tr:
        fire = r["s_t"] is not None and r["s_t"] < g and r["epochs_flat"] >= gate
        consec = consec + 1 if fire else 0
        if consec >= k:
            return r["t"]
    return T - 1


def plist_for(cfg: Cfg, gate_mode="sched", k=2):
    def gate(T):
        if gate_mode == "none":
            return 0
        return max(2, round(cfg.safeguard_frac * T)) if cfg.safeguard_frac is not None else 5
    if not cfg.use_curve:  # parameter = required flat-epoch fraction (conservative -> aggressive)
        fr = [0.5, 0.4, 0.33, 0.25, 0.2, 0.15, 0.125, 0.1, 0.08, 0.06]
        return [(f"flat={f}", lambda df, tr, f=f: sim(tr, 1.0, len(df), max(1, round(f * len(df))), k)) for f in fr]
    return [(f"g={g}", lambda df, tr, g=g: sim(tr, g, len(df), gate(len(df)), k)) for g in GAMMAS]


def run_variant(name, jobs, dom_arr, traces, plist, rhos=(0.15, 0.20), beta=0.05):
    mats = tierA.evaluate_family(jobs, traces, plist)
    rows = []
    for rho in rhos:
        r = tierA.crossfit(*mats, dom_arr, rho, beta)
        rows.append({"variant": name, "rho": rho, "oof_risk": r["risk"], "risk_vision": r["risk_by_dom"]["vision"],
                     "mean_saved": r["saved"], "energy_weighted_saved": r["ew"], "uncertified_folds": r["n_uncert"]})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs, _, dom = tierA.load()
    names = list(jobs)
    dom_arr = np.array([dom[k] for k in names])
    base = Cfg()
    a5 = [
        ("Full EcoStop", base, "sched", 2),
        ("No curve score (flat-gate + k only)", replace(base, use_curve=False), "sched", 2),
        ("No flat-gate (score + k only)", base, "none", 2),
        ("k=1 (no consecutive-signal rule)", base, "sched", 1),
        ("Point estimate (no uncertainty)", replace(base, sigma="point"), "sched", 2),
        ("Residual sigma instead of parameter SE", replace(base, sigma="resid"), "sched", 2),
        ("Full-history fit (no trailing window)", replace(base, window=None), "sched", 2),
        ("No a_max fallback", replace(base, fallback=False), "sched", 2),
        ("Fixed 5-epoch safeguard", replace(base, safeguard_frac=None), "sched", 2),
    ]
    a6 = [(f"eps={e}", replace(base, eps=e), "sched", 2) for e in (0.1, 0.25, 1.0, 2.0)]
    a6 += [(f"window={w}", replace(base, window=w), "sched", 2) for w in (5, 8, 10, 20)]
    a6 += [(f"safeguard_frac={f}", replace(base, safeguard_frac=f), "sched", 2) for f in (0.05, 0.1, 0.2, 0.25)]
    a6 += [(f"flat_tol={x}", replace(base, flat_tol=x), "sched", 2) for x in (0.0, 0.05, 0.2)]
    a6 += [(f"k={k}", base, "sched", k) for k in (3, 4)]

    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as pool:
        for label, cfg, gm, k in a5 + a6:
            tag = "A5" if (label, cfg, gm, k) in a5 else "A6"
            print(f"[{tag}] {label}", flush=True)
            tr = traces_for_cfg(jobs, cfg, pool)
            if cfg.eps != tierA.EPS:  # loss definition depends on eps
                tierA.EPS = cfg.eps
            rs = run_variant(label, jobs, dom_arr, tr, plist_for(cfg, gm, k))
            tierA.EPS = 0.5
            for r in rs:
                r["study"] = tag
            rows += rs
            pd.DataFrame(rows).to_csv(f"{OUT}/ablation_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
