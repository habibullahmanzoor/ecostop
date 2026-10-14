"""Fix of EcoStop's remaining fixed constants: schedule-relative gates.

Diagnosis (dev data, 2026-10-12): EcoStop fires only when (i) the curve score is below gamma, (ii) the best checkpoint
has been flat for >= max(2, round(0.125 T)) epochs and (iii) (i)-(ii) hold on 2 consecutive epochs. The floors of 2
are fixed constants: on 10-epoch fine-tuning they make EcoStop at least as slow as patience 3, which caps its NLP
saving at 33.7% (oracle 75%; calibrated patience 44%). The same cap appears for LC-PFN with EcoStop's gates. This
is the failure the paper already fixed for the 5-epoch safeguard.

Variant, declared before running (primary):
  EcoStop-SR: flat gate = max(1, round(0.125 T)); confirmations k = min(2, max(1, round(0.05 T))).
  For T >= 30 both equal the paper rule exactly, so vision and tabular stops are unchanged; only short runs change
  (dev NLP T=10: gate 2->1, k 2->1; fresh NLP T=15: gate 2, k 2->1). Applied identically to LC-PFN's gates.
Sensitivity (secondary): EcoStop-SRall: k = max(1, round(0.05 T)) without the cap of 2 (changes long runs too).

Evaluations (declared before running):
  A. Pooled certification (betting bound), out-of-fold on dev, matched realised risk, paired job bootstrap.
  B. Per-domain (Mondrian) certification: each domain certified separately at rho (beta = 0.05 per domain), a
     STRONGER guarantee than pooled; out-of-fold; same rho -> same guarantee. PRIMARY: EcoStop-SR vs calibrated
     patience under B at rho in {0.15, 0.20}, energy-weighted saving.
  C. Fresh 35 jobs: thresholds calibrated on dev (pooled and per-domain) applied unchanged; scope card for every
     method; paired bootstrap."""
from __future__ import annotations

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
from echostop.framework_v2 import crossfit, select, wsr_pvalue
from echostop.lcpfn_real import PFN_GRID

GAM = tierA.GAMMAS
PAT = "Patience (p=f*T, min-delta 0.1)"
RHOS = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
DOMS = ["nlp", "tabular", "vision"]


def g_paper(T): return max(2, round(0.125 * T))
def g_sr(T): return max(1, round(0.125 * T))
def k_paper(T): return 2
def k_sr(T): return min(2, max(1, round(0.05 * T)))
def k_srall(T): return max(1, round(0.05 * T))


def families(pol):
    f = {}
    f["EcoStop"] = [(f"g={g}", lambda df, tr, g=g: sim(tr["eco"], g, len(df), g_paper(len(df)), k_paper(len(df)))) for g in GAM]
    f["EcoStop-SR"] = [(f"g={g}", lambda df, tr, g=g: sim(tr["eco"], g, len(df), g_sr(len(df)), k_sr(len(df)))) for g in GAM]
    f["EcoStop-SRall"] = [(f"g={g}", lambda df, tr, g=g: sim(tr["eco"], g, len(df), g_sr(len(df)), k_srall(len(df)))) for g in GAM]
    f["LC-PFN + gates"] = [(f"g={g}", lambda df, tr, g=g: sim(tr["pfn"], g, len(df), g_paper(len(df)), 2)) for g in PFN_GRID]
    f["LC-PFN + SR gates"] = [(f"g={g}", lambda df, tr, g=g: sim(tr["pfn"], g, len(df), g_sr(len(df)), k_sr(len(df)))) for g in PFN_GRID]
    f[PAT] = [(lab, lambda df, tr, fn=fn: fn(df, None)) for lab, fn in pol[PAT]]
    return f


def mats_for(jobs, eco, pfn, fams):
    tr = {k: [{"eco": e, "pfn": p} for e, p in zip(eco[k], pfn[k])] for k in jobs}
    return {name: tierA.evaluate_family(jobs, tr, pl) for name, pl in fams.items()}


def pooled_chooser(L, rule, rho):
    return lambda tr, te: [(te, None if (i := select(L[:, tr], rho, wsr_pvalue)) is None else (rule, i))]


def mondrian_chooser(L, rule, rho, dom):
    def ch(tr, te):
        out = []
        for d in DOMS:
            trd, ted = tr[dom[tr] == d], te[dom[te] == d]
            i = select(L[:, trd], rho, wsr_pvalue)
            out.append((ted, None if i is None else (rule, i)))
        return out
    return ch


def boot(d, et, rng, B=5000):
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = d[idx].sum(1) / et[idx].sum(1) * 100
    return d.sum() / et.sum() * 100, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    cache = "cache/srgates_mats.pkl"
    pol = tierA.build_policies(None)
    fams = families(pol)
    if os.path.exists(cache):
        D, F, dom, fdom, ok = pickle.load(open(cache, "rb"))
    else:
        dev, eco_d, domd = tierA.load()
        fresh, fdd = E.load_b1()
        eco_f = {k: [precompute_score_trace(df, 0.5) for df in dfs] for k, dfs in fresh.items()}
        D = mats_for(dev, eco_d, pickle.load(open("cache/lcpfn_traces.pkl", "rb")), fams)
        F = mats_for(fresh, eco_f, pickle.load(open("cache/b1_lcpfn_traces.pkl", "rb")), fams)
        dom = np.array([domd[k] for k in dev]); fdom = np.array([fdd[k] for k in fresh])
        flags = pd.read_csv("figures/tierA/tier1_scope_flags.csv").set_index("job")
        ok = np.array([bool(flags.loc["b1:" + n, "in_scope"]) for n in fresh])
        pickle.dump((D, F, dom, fdom, ok), open(cache, "wb"))
    etd, etf = D["EcoStop"][3], F["EcoStop"][3]
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 230)

    # ---- in-sample per-domain frontier (diagnostic only)
    print("Diagnostic, in-sample: best NLP energy-weighted saving at NLP risk <= .05/.10/.15/.20")
    m = dom == "nlp"
    for r in fams:
        L, _, ES, _ = D[r]; R = L[:, m].mean(1); W = 100 * ES[:, m].sum(1) / etd[m].sum()
        print(f"  {r:34s}", " ".join(f"{(W[R <= t].max() if (R <= t).any() else 0):5.1f}" for t in (.05, .10, .15, .20)))

    rows, store = [], {}
    for mode in ("pooled", "per-domain"):
        for r in fams:
            L = D[r][0]
            for rho in RHOS:
                ch = pooled_chooser(L, r, rho) if mode == "pooled" else mondrian_chooser(L, r, rho, dom)
                x = crossfit(D, dom, rho, ch)
                store[("dev", mode, r, rho)] = (x["risk"], x["ES"], x["L"])
                # fresh: calibrate on all dev, apply unchanged, scope card for everyone
                Lf, _, ESf, _ = F[r]
                if mode == "pooled":
                    i = select(L, rho, wsr_pvalue)
                    pick = np.full(len(fdom), -1 if i is None else i)
                else:
                    pick = np.full(len(fdom), -1)
                    for d in DOMS:
                        i = select(L[:, dom == d], rho, wsr_pvalue)
                        pick[fdom == d] = -1 if i is None else i
                lf = np.array([Lf[p, j] if p >= 0 else 0.0 for j, p in enumerate(pick)]) * ok
                ef = np.array([ESf[p, j] if p >= 0 else 0.0 for j, p in enumerate(pick)]) * ok
                store[("fresh", mode, r, rho)] = (lf.mean(), ef, lf)
                rows.append({"mode": mode, "rule": r, "rho": rho,
                             "dev_risk": x["risk"], "dev_EW": x["EW"],
                             **{f"dev_risk_{d}": x["L"][dom == d].mean() for d in DOMS},
                             "fresh_risk": lf.mean(), "fresh_EW": 100 * ef.sum() / etf.sum(),
                             **{f"fresh_risk_{d}": lf[fdom == d].mean() for d in DOMS}})
    df = pd.DataFrame(rows)
    df.to_csv("figures/tierA/fv2_srgates.csv", index=False)
    for mode in ("pooled", "per-domain"):
        print(f"\n=== {mode} certification (betting bound). EW % and realised risk; dev out-of-fold, fresh frozen + scope card")
        x = df[(df["mode"] == mode) & df.rho.isin([0.10, 0.15, 0.20])]
        print(x.drop(columns="mode").round(3).to_string(index=False))

    print("\n=== PRIMARY: per-domain certification, same rho (same guarantee): EcoStop-SR minus calibrated patience, EW pp")
    for setname, et in (("dev", etd), ("fresh", etf)):
        for rho in (0.10, 0.15, 0.20):
            for r in ("EcoStop-SR", "EcoStop", "LC-PFN + SR gates"):
                a, b = store[(setname, "per-domain", r, rho)], store[(setname, "per-domain", PAT, rho)]
                pt, lo, hi = boot(a[1] - b[1], et, rng)
                print(f"  {setname:5s} rho={rho:.2f} {r:18s} - patience: {pt:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]   risk {a[0]:.3f} vs {b[0]:.3f}")

    print("\n=== Pooled certification at MATCHED realised risk (paper standard): EcoStop-SR minus patience, EW pp")
    for setname, et in (("dev", etd), ("fresh", etf)):
        for rho_b in (0.10, 0.15, 0.20):
            rb, eb, _ = store[(setname, "pooled", PAT, rho_b)]
            for r in ("EcoStop-SR", "EcoStop"):
                q = min(RHOS, key=lambda q: abs(store[(setname, "pooled", r, q)][0] - rb))
                ra, ea, _ = store[(setname, "pooled", r, q)]
                pt, lo, hi = boot(ea - eb, et, rng)
                print(f"  {setname:5s} {r:12s} risk {ra:.3f} vs patience {rb:.3f}: {pt:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]")


if __name__ == "__main__":
    main()
