"""Out-of-sample check of the unknown-horizon result: calibrate EcoStop-HF (fixed 5-epoch safeguard, no T) and
fixed-p patience (min-delta 0, 0.1, 0.2, 0.5) on the 142 DEVELOPMENT jobs, apply the frozen thresholds unchanged to the
35 fresh jobs (never evaluated with these variants before). Paired job bootstrap for energy-weighted differences."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E
from echostop import tierA
from echostop.ablation import Cfg, plist_for, trace_for
from concurrent.futures import ProcessPoolExecutor
from echostop.ablation import traces_for_cfg

HF = "EcoStop-HF (fixed 5-epoch safeguard, no T)"


def main():
    dev, _, dom = tierA.load()
    fresh, fdom = E.load_b1()
    cfg = Cfg(safeguard_frac=None)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 1)) as pool:
        trd = traces_for_cfg(dev, cfg, pool)
        trf = traces_for_cfg(fresh, cfg, pool)
    fams = {HF: plist_for(cfg, "sched", 2)}
    for md in (0.0, 0.1, 0.2, 0.5):
        fams[f"Patience fixed p, min-delta {md}"] = [(f"p={p}", lambda df, t, p=p, md=md: tierA.patience_stop_md(df, p, md))
                                                     for p in range(20, 0, -1)]
    rng = np.random.default_rng(0)
    out = {}
    for rho in (0.15, 0.20):
        for name, pl in fams.items():
            ld, sd, esd, etd = tierA.evaluate_family(dev, trd, pl)
            i = tierA.select(ld, rho, 0.05)                       # calibrated on development jobs only
            lf, sf, esf, etf = tierA.evaluate_family(fresh, trf, pl)
            if i is None:
                out[(name, rho)] = (pl[0][0], 0.0, 0.0, np.zeros(len(etf)), etf)
            else:
                out[(name, rho)] = (pl[i][0], lf[i].mean(), sf[i].mean(), esf[i], etf)
    rows = []
    for (name, rho), (par, risk, ms, es, et) in out.items():
        rows.append({"rho": rho, "method": name, "param": par, "fresh_risk": risk, "fresh_mean_saved": ms, "fresh_EW": 100 * es.sum() / et.sum()})
    df = pd.DataFrame(rows)
    df.to_csv("figures/tierA/horizon_free_fresh.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 48)
    print(df.round(3).to_string(index=False))
    print("\nFresh jobs (35): EcoStop-HF minus patience, energy-weighted pp at the SAME calibration target rho (paired job bootstrap):")
    for rho in (0.15, 0.20):
        h = out[(HF, rho)]
        for name in fams:
            if name == HF:
                continue
            p = out[(name, rho)]
            d = h[3] - p[3]
            idx = rng.integers(0, len(d), (5000, len(d)))
            bs = d[idx].sum(1) / h[4][idx].sum(1) * 100
            print(f"rho={rho:.2f} vs {name:34s} risk HF {h[1]:.3f} vs {p[1]:.3f} | dEW {d.sum() / h[4].sum() * 100:+6.2f} [{np.percentile(bs, 2.5):+6.2f}, {np.percentile(bs, 97.5):+6.2f}]")


if __name__ == "__main__":
    main()
