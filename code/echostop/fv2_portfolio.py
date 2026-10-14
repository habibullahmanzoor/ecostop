"""Improvement 2: certified rule selection per domain (a portfolio).

For every assignment a: domain -> rule (rules from CANDIDATES), build ONE composite ordered family by mapping a shared
aggressiveness index k (0 = most conservative) to each rule's own ordered grid by rank. Certify every assignment's
path with fixed-sequence testing at beta / |assignments| (Bonferroni over assignments). Because every assignment is
tested simultaneously at familywise level beta, choosing the certified (assignment, k) with the highest energy saving
on the calibration jobs keeps the guarantee. Out-of-fold evaluation on the same folds as everything else."""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.framework_v2 import BETA, crossfit, hb_p, load, select, wsr_pvalue

mats, names, dom = load()
DOMS = ["nlp", "tabular", "vision"]
NK = 25


def composite(assign):
    """loss/saved/es for the composite rule, shape [NK, J]."""
    J = len(dom)
    L, S, ES = np.zeros((NK, J)), np.zeros((NK, J)), np.zeros((NK, J))
    for d in DOMS:
        loss, saved, es, _ = mats[assign[d]]
        P = loss.shape[0]
        idx = np.round(np.linspace(0, P - 1, NK)).astype(int)
        m = dom == d
        L[:, m], S[:, m], ES[:, m] = loss[idx][:, m], saved[idx][:, m], es[idx][:, m]
    return L, S, ES, mats[assign[DOMS[0]]][3]


def run(candidates, pf, rho, label):
    assigns = [dict(zip(DOMS, combo)) for combo in itertools.product(candidates, repeat=len(DOMS))]
    comp = {i: composite(a) for i, a in enumerate(assigns)}
    beta_each = BETA / len(assigns)
    et = mats[candidates[0]][3]

    def chooser(tr, te):
        best, best_ew = None, -1.0
        for i in comp:
            L, S, ES, _ = comp[i]
            k = select(L[:, tr], rho, pf, beta=beta_each)
            if k is not None:
                ew = ES[k, tr].sum() / et[tr].sum()
                if ew > best_ew:
                    best, best_ew = (i, k), ew
        return [(te, best)]

    allm = {i: comp[i] for i in comp}
    r = crossfit(allm, dom, rho, chooser)
    return {"rho": rho, "method": label, "risk": r["risk"], "mean_saved": r["mean_saved"], "EW": r["EW"], "ES": r["ES"],
            "risk_vision": r["L"][dom == "vision"].mean()}


def single(rule, pf, rho):
    loss = mats[rule][0]
    ch = lambda tr, te: [(te, None if (i := select(loss[:, tr], rho, pf)) is None else (rule, i))]
    r = crossfit(mats, dom, rho, ch)
    return {"rho": rho, "method": f"{rule} alone", "risk": r["risk"], "mean_saved": r["mean_saved"], "EW": r["EW"], "ES": r["ES"],
            "risk_vision": r["L"][dom == "vision"].mean()}


if __name__ == "__main__":
    et = mats["EcoStop"][3]
    rows, keep = [], {}
    for rho in (0.15, 0.20):
        for rule in ("EcoStop", "Patience (p=f*T, min-delta 0.1)", "LC-PFN + EcoStop gates"):
            x = single(rule, wsr_pvalue, rho); rows.append(x); keep[(x["method"], rho)] = x
        for cands, lab in ((["EcoStop", "Patience (p=f*T, min-delta 0.1)"], "Portfolio {EcoStop, patience}"),
                           (["EcoStop", "Patience (p=f*T, min-delta 0.1)", "LC-PFN + EcoStop gates"], "Portfolio {EcoStop, patience, LC-PFN}")):
            x = run(cands, wsr_pvalue, rho, lab); rows.append(x); keep[(lab, rho)] = x
        print("done", rho, flush=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "ES"} for r in rows])
    df.to_csv("figures/tierA/fv2_portfolio.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 44)
    print(df.round(3).to_string(index=False))
    rng = np.random.default_rng(0)
    print("\nPaired job bootstrap, energy-weighted pp (all certified with the betting bound, same folds):")
    for rho in (0.15, 0.20):
        a = keep[("Portfolio {EcoStop, patience}", rho)]
        for other in ("EcoStop alone", "Patience (p=f*T, min-delta 0.1) alone"):
            b = keep[(other, rho)]
            d = a["ES"] - b["ES"]
            idx = rng.integers(0, len(d), (5000, len(d)))
            bs = d[idx].sum(1) / et[idx].sum(1) * 100
            print(f"rho={rho:.2f} portfolio minus {other:38s}: {d.sum() / et.sum() * 100:+6.2f} [{np.percentile(bs, 2.5):+6.2f}, {np.percentile(bs, 97.5):+6.2f}]  "
                  f"(risk {a['risk']:.3f} vs {b['risk']:.3f})")
