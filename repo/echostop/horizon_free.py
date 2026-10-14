"""Exploratory: the UNKNOWN-HORIZON regime. Rules that may not use the training length T:
  EcoStop-HF : EcoStop with the ORIGINAL fixed 5-epoch safeguard and 5-epoch flat gate (no T anywhere)
  Patience   : fixed p (epochs), with min-delta in {0, 0.1, 0.2, 0.5}
All calibrated with the same fixed-sequence Hoeffding-Bentkus test; scored out-of-fold (5 folds x 20 repeats) on the
142 development jobs; paired job-bootstrap for energy-weighted differences at matched realised risk.
(Exploratory: the ablation row for the fixed 5-epoch safeguard was seen before this analysis.)"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import tierA
from echostop.ablation import Cfg, plist_for, traces_for_cfg

RHOS = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]


def main():
    jobs, _, dom = tierA.load()
    names = list(jobs)
    dom_arr = np.array([dom[k] for k in names])
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as pool:
        tr = traces_for_cfg(jobs, Cfg(safeguard_frac=None), pool)
    fams = {"EcoStop-HF (fixed 5-epoch safeguard, no T)": plist_for(Cfg(safeguard_frac=None), "sched", 2)}
    for md in (0.0, 0.1, 0.2, 0.5):
        fams[f"Patience fixed p, min-delta {md}"] = [(f"p={p}", lambda df, t, p=p, md=md: tierA.patience_stop_md(df, p, md))
                                                     for p in range(20, 0, -1)]
    res, et = {}, None
    for name, pl in fams.items():
        loss, saved, es, e_t = tierA.evaluate_family(jobs, tr, pl)
        et = e_t
        for rho in RHOS:
            r = tierA.crossfit(loss, saved, es, e_t, dom_arr, rho, 0.05, extra=es)
            res[(name, rho)] = {"risk": r["risk"], "mean_saved": r["saved"], "ES": r["job_extra"], "EW": 100 * r["job_extra"].sum() / e_t.sum()}
        print("done", name, flush=True)
    rows = [{"method": n, "rho": r, "risk": v["risk"], "mean_saved": v["mean_saved"], "EW": v["EW"]} for (n, r), v in res.items()]
    df = pd.DataFrame(rows)
    df.to_csv("figures/tierA/horizon_free.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 48)
    print(df.round(3).to_string(index=False))

    # paired job-bootstrap at matched realised risk: for each patience variant and each of its rho points, pick the
    # EcoStop-HF point with the closest realised risk
    hf = "EcoStop-HF (fixed 5-epoch safeguard, no T)"
    rng = np.random.default_rng(0)
    print("\nEcoStop-HF minus best-matching patience at matched realised risk (energy-weighted pp, paired job bootstrap):")
    for name in fams:
        if name == hf:
            continue
        for rho_p in (0.12, 0.15, 0.20):
            p = res[(name, rho_p)]
            best = min(RHOS, key=lambda r: abs(res[(hf, r)]["risk"] - p["risk"]))
            h = res[(hf, best)]
            d = h["ES"] - p["ES"]
            idx = rng.integers(0, len(d), (5000, len(d)))
            bs = d[idx].sum(1) / et[idx].sum(1) * 100
            print(f"{name:34s} rho={rho_p:.2f} (risk {p['risk']:.3f}, EW {p['EW']:5.1f}) | HF rho={best:.2f} (risk {h['risk']:.3f}, EW {h['EW']:5.1f}) | "
                  f"dEW {d.sum() / et.sum() * 100:+6.2f} [{np.percentile(bs, 2.5):+6.2f}, {np.percentile(bs, 97.5):+6.2f}]")


if __name__ == "__main__":
    main()
