"""C1-F: held-out test of the EcoStop FRAMEWORK on the 82 C1 jobs, as specified in c1_framework_plan.md.
Everything is calibrated on the 177 existing jobs (142 development + 35 B1) only, then applied unchanged to C1.
  Baselines  LTT-HB (Hoeffding-Bentkus, fixed-sequence) with ONE rule: EcoStop (paper gates), calibrated patience, LC-PFN+gates
  Framework  betting certificate + certified per-domain rule selection (27 assignments, beta/27) + scope card
             (all C1 jobs are in scope: NLP fine-tuning, linear decay, horizons 8 and 12 within x2 of 10 and 15)."""
from __future__ import annotations

import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E
from echostop import c1_eval as CE
from echostop import tierA
from echostop.ablation import sim
from echostop.calibration import precompute_score_trace
from echostop.framework_v2 import BETA, hb_p, select, wsr_pvalue
from echostop.lcpfn_real import PFN_GRID

RULES = ["EcoStop", "Patience (p=f*T, min-delta 0.1)", "LC-PFN + EcoStop gates"]
SHORT = {RULES[0]: "EcoStop", RULES[1]: "calibrated patience", RULES[2]: "LC-PFN + gates"}
DOMS, NK, RHOS = ["nlp", "tabular", "vision"], 25, (0.10, 0.15, 0.20)


def families():
    pol = tierA.build_policies(None)
    pfn = [(f"g={g}", (lambda df, t, g=g: sim(t, g, len(df), max(2, round(0.125 * len(df))), 2))) for g in PFN_GRID]
    return {RULES[0]: (pol["EcoStop"], "eco"), RULES[1]: (pol[RULES[1]], "eco"), RULES[2]: (pfn, "pfn")}


def mats(jobs, eco, pfn):
    out = {}
    for r, (pl, kind) in families().items():
        L, S, ES, et = tierA.evaluate_family(jobs, eco if kind == "eco" else pfn, pl)
        out[r] = dict(L=L, ES=ES, et=et, P=len(pl))
    return out


def main():
    dev, eco_d, dom_d = tierA.load()
    b1, dom_b = E.load_b1()
    eco_b = {k: [precompute_score_trace(df, 0.5) for df in v] for k, v in b1.items()}
    pfn_d = pickle.load(open("cache/lcpfn_traces.pkl", "rb"))
    pfn_b = pickle.load(open("cache/b1_lcpfn_traces.pkl", "rb"))
    cal_jobs = {**{f"dev:{k}": v for k, v in dev.items()}, **{f"b1:{k}": v for k, v in b1.items()}}
    cal_eco = {**{f"dev:{k}": v for k, v in eco_d.items()}, **{f"b1:{k}": v for k, v in eco_b.items()}}
    cal_pfn = {**{f"dev:{k}": v for k, v in pfn_d.items()}, **{f"b1:{k}": v for k, v in pfn_b.items()}}
    cal_dom = np.array([dom_d[k[4:]] if k.startswith("dev:") else dom_b[k[3:]] for k in cal_jobs])
    C = mats(cal_jobs, cal_eco, cal_pfn)
    print(f"calibration jobs: {len(cal_jobs)} ({int((cal_dom == 'nlp').sum())} NLP)")

    c1_jobs, _, dropped = CE.load_logs("logs/c1/nlp")
    names = sorted(c1_jobs)
    ds = np.array([k.split("|")[0] for k in names])
    jobs1 = {k: c1_jobs[k] for k in names}
    eco1 = {k: [precompute_score_trace(df, 0.5) for df in v] for k, v in jobs1.items()}
    pfn1 = pickle.load(open("cache/c1_lcpfn_traces.pkl", "rb"))
    X = mats(jobs1, eco1, {k: pfn1[k] for k in names})
    et1 = X[RULES[0]]["et"]
    print(f"C1 jobs: {len(names)} ({len(set(ds))} datasets); dropped {dropped}")

    idxmap = {r: np.round(np.linspace(0, C[r]["P"] - 1, NK)).astype(int) for r in RULES}

    def composite(assign):
        J = len(cal_dom)
        L, ES = np.zeros((NK, J)), np.zeros((NK, J))
        for d in DOMS:
            m = cal_dom == d
            r = assign[d]
            L[:, m], ES[:, m] = C[r]["L"][idxmap[r]][:, m], C[r]["ES"][idxmap[r]][:, m]
        return L, ES

    assigns = [dict(zip(DOMS, c)) for c in itertools.product(RULES, repeat=3)]
    comp = [composite(a) for a in assigns]
    et_cal = C[RULES[0]]["et"]
    rng = np.random.default_rng(0)
    Wb = CE.boot_weights(ds, rng)
    picks, rows = {}, []
    for rho in RHOS:
        best, best_ew = None, -1.0
        for i, (L, ES) in enumerate(comp):
            k = select(L, rho, wsr_pvalue, beta=BETA / len(assigns))
            if k is not None and ES[k].sum() / et_cal.sum() > best_ew:
                best, best_ew = (i, k), ES[k].sum() / et_cal.sum()
        meth = {}
        if best is None:
            meth["EcoStop framework"] = (None, None)
        else:
            i, k = best
            r = assigns[i]["nlp"]
            meth["EcoStop framework"] = (r, int(idxmap[r][k]))
        for r in RULES:
            j = select(C[r]["L"], rho, hb_p)
            meth[f"LTT-HB, {SHORT[r]}"] = (r, j)
            j2 = select(C[r]["L"], rho, wsr_pvalue)
            meth[f"betting only, {SHORT[r]}"] = (r, j2)
        for name, (r, j) in meth.items():
            if r is None or j is None:
                L1, ES1 = np.zeros(len(names)), np.zeros(len(names))
                lab = "never stop"
            else:
                L1, ES1 = X[r]["L"][j], X[r]["ES"][j]
                lab = f"{SHORT[r]} @ {j}"
            picks[(rho, name)] = (L1, ES1)
            rows.append(dict(rho=rho, method=name, setting=lab, C1_risk=L1.mean(), C1_EW=100 * ES1.sum() / et1.sum(),
                             within_target=bool(L1.mean() <= rho + 1e-12)))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_colwidth", 44)
    print(df.round(3).to_string(index=False))
    df.to_csv("figures/tierA/c1_framework.csv", index=False)

    print("\nFramework minus each baseline, EW pp at the same rho (dataset-cluster bootstrap, 10,000):")
    out = []
    for rho in RHOS:
        fL, fE = picks[(rho, "EcoStop framework")]
        hb = [(n, picks[(rho, n)]) for n in df[df.rho == rho].method if n.startswith("LTT-HB")]
        best_hb = max(hb, key=lambda t: t[1][1].sum())[0]
        for n, (bL, bE) in hb:
            d = fE - bE
            bs = (Wb @ d) / (Wb @ et1) * 100
            star = " <-- best single-rule LTT-HB" if n == best_hb else ""
            print(f"  rho={rho:.2f} vs {n:28s}: {100 * d.sum() / et1.sum():+6.2f}  [{np.percentile(bs, 2.5):+6.2f}, {np.percentile(bs, 97.5):+6.2f}]{star}")
            out.append(dict(rho=rho, baseline=n, diff=100 * d.sum() / et1.sum(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), best=n == best_hb))
    pd.DataFrame(out).to_csv("figures/tierA/c1_framework_diffs.csv", index=False)


if __name__ == "__main__":
    main()
