"""End-to-end EcoStop framework vs the standard certification (Learn-then-Test with a Hoeffding-Bentkus p-value).

Specified before running (2026-10-12):
  Baseline  LTT-HB(rule): one rule, Hoeffding-Bentkus p-value, fixed-sequence test (Angelopoulos et al.), as in paper v1.
  EcoStop   betting (WSR) p-value + certified per-domain selection over {curve, patience p=fT, LC-PFN} (Bonferroni over
            27 assignments) + scope card (abstain = run to completion outside the calibration population).
  Comparison at the SAME target rho in {0.10, 0.15, 0.20}, beta = 0.05: same guarantee, more saving wins (the standard
  comparison for risk-control methods). Primary: EcoStop vs LTT-HB(calibrated patience), the strongest single rule.
  Dev: out-of-fold (5 folds x 20 repeats, stratified by domain); every dev job is in scope by construction.
  Fresh: everything calibrated on the 142 dev jobs and applied unchanged to the 35 fresh jobs. On fresh the baseline is
  ALSO given the scope card (favours the baseline); its risk without the card is reported as well."""
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E
from echostop import tierA
from echostop.ablation import sim
from echostop.calibration import precompute_score_trace
from echostop.framework_v2 import BETA, crossfit, hb_p, select, wsr_pvalue
from echostop.fv2_portfolio import DOMS, NK, dom as dev_dom, mats as dev_mats, run as portfolio_oof
from echostop.lcpfn_real import PFN_GRID

SINGLE = ["EcoStop", "Patience (p=f*T, min-delta 0.1)", "LC-PFN + EcoStop gates"]
SHORT = {"EcoStop": "curve", "Patience (p=f*T, min-delta 0.1)": "patience", "LC-PFN + EcoStop gates": "LC-PFN"}
RHOS = (0.10, 0.15, 0.20)
PRIMARY = "Patience (p=f*T, min-delta 0.1)"


def composite(mats, dom, assign):
    J = len(dom)
    L, ES = np.zeros((NK, J)), np.zeros((NK, J))
    for d in DOMS:
        loss, _, es, _ = mats[assign[d]]
        idx = np.round(np.linspace(0, loss.shape[0] - 1, NK)).astype(int)
        m = dom == d
        L[:, m], ES[:, m] = loss[idx][:, m], es[idx][:, m]
    return L, ES


def fresh_matrices():
    fresh, fdom = E.load_b1()
    names = list(fresh)
    tr = {k: [precompute_score_trace(df, 0.5) for df in dfs] for k, dfs in fresh.items()}
    pf_tr = pickle.load(open("cache/b1_lcpfn_traces.pkl", "rb"))
    pol = tierA.build_policies(None)
    pfn_pl = [(f"g={g}", (lambda df, t, g=g: sim(t, g, len(df), max(2, round(0.125 * len(df))), 2))) for g in PFN_GRID]
    fm = {"EcoStop": tierA.evaluate_family(fresh, tr, pol["EcoStop"]),
          PRIMARY: tierA.evaluate_family(fresh, tr, pol[PRIMARY]),
          "LC-PFN + EcoStop gates": tierA.evaluate_family(fresh, pf_tr, pfn_pl)}
    for r in SINGLE:
        assert dev_mats[r][0].shape[0] == fm[r][0].shape[0], r
    flags = pd.read_csv("figures/tierA/tier1_scope_flags.csv").set_index("job")
    ok = np.array([bool(flags.loc["b1:" + n, "in_scope"]) for n in names])
    return fm, np.array([fdom[k] for k in names]), ok


def boot(d, et, rng, B=5000):
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = d[idx].sum(1) / et[idx].sum(1) * 100
    return d.sum() / et.sum() * 100, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    et_dev = dev_mats["EcoStop"][3]
    rows, dev_es, fr_es = [], {}, {}

    # ---------------- development, out-of-fold
    for rho in RHOS:
        for rule in SINGLE:
            loss = dev_mats[rule][0]
            for tag, pf in (("LTT-HB", hb_p), ("betting", wsr_pvalue)):
                ch = lambda tr, te, loss=loss, pf=pf, rule=rule: [(te, None if (i := select(loss[:, tr], rho, pf)) is None else (rule, i))]
                r = crossfit(dev_mats, dev_dom, rho, ch)
                lab = f"{tag} ({SHORT[rule]})"
                dev_es[(lab, rho)] = r["ES"]
                rows.append({"set": "dev", "rho": rho, "method": lab, "risk": r["risk"], "risk_in_scope": r["risk"], "EW": r["EW"]})
        r = portfolio_oof(SINGLE, wsr_pvalue, rho, "EcoStop")
        dev_es[("EcoStop", rho)] = r["ES"]
        rows.append({"set": "dev", "rho": rho, "method": "EcoStop (betting + selection + scope)", "risk": r["risk"],
                     "risk_in_scope": r["risk"], "EW": r["EW"]})
        print("dev done", rho, flush=True)

    # ---------------- fresh, calibrated on dev only
    fm, fdom, ok = fresh_matrices()
    et_fr = fm["EcoStop"][3]
    assigns = [dict(zip(DOMS, c)) for c in itertools.product(SINGLE, repeat=3)]
    dev_comp = [composite(dev_mats, dev_dom, a) for a in assigns]
    fr_comp = [composite(fm, fdom, a) for a in assigns]
    for rho in RHOS:
        for rule in SINGLE:
            for tag, pf in (("LTT-HB", hb_p), ("betting", wsr_pvalue)):
                j = select(dev_mats[rule][0], rho, pf)
                Lf, _, ESf, _ = fm[rule]
                L = Lf[j] if j is not None else np.zeros(len(ok))
                ES = ESf[j] if j is not None else np.zeros(len(ok))
                lab = f"{tag} ({SHORT[rule]})"
                fr_es[(lab + " no card", rho)] = ES
                fr_es[(lab, rho)] = ES * ok
                rows.append({"set": "fresh", "rho": rho, "method": lab + ", no scope card", "risk": L.mean(),
                             "risk_in_scope": L[ok].mean(), "EW": 100 * ES.sum() / et_fr.sum()})
                rows.append({"set": "fresh", "rho": rho, "method": lab + " + scope card", "risk": (L * ok).mean(),
                             "risk_in_scope": L[ok].mean(), "EW": 100 * (ES * ok).sum() / et_fr.sum()})
        best, best_ew = None, -1.0
        for i, (L, ES) in enumerate(dev_comp):
            k = select(L, rho, wsr_pvalue, beta=BETA / len(assigns))
            if k is not None and ES[k].sum() / et_dev.sum() > best_ew:
                best, best_ew = (i, k), ES[k].sum() / et_dev.sum()
        i, k = best
        L, ES = fr_comp[i][0][k], fr_comp[i][1][k]
        fr_es[("EcoStop", rho)] = ES * ok
        rows.append({"set": "fresh", "rho": rho, "method": "EcoStop (betting + selection + scope)", "risk": (L * ok).mean(),
                     "risk_in_scope": L[ok].mean(), "EW": 100 * (ES * ok).sum() / et_fr.sum(),
                     "assignment": str({d: SHORT[assigns[i][d]] for d in DOMS})})

    df = pd.DataFrame(rows)
    df.to_csv("figures/tierA/fv2_pipeline.csv", index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    print(df.round(3).to_string(index=False))

    rng = np.random.default_rng(0)
    print("\nEcoStop framework minus standard LTT-HB, energy-weighted pp at the SAME target rho (paired job bootstrap).")
    print("Fresh baselines also get the scope card. Primary row marked *.")
    for setname, store, et in (("dev", dev_es, et_dev), ("fresh", fr_es, et_fr)):
        for rho in RHOS:
            for rule in SINGLE:
                pt, lo, hi = boot(store[("EcoStop", rho)] - store[(f"LTT-HB ({SHORT[rule]})", rho)], et, rng)
                star = "*" if rule == PRIMARY else " "
                print(f" {star}{setname:5s} rho={rho:.2f} vs LTT-HB ({SHORT[rule]:8s}): {pt:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]")


if __name__ == "__main__":
    main()
