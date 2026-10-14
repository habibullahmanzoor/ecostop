"""C1 evaluation (hash-locked by c1_freeze.py; see c1_analysis_plan.md for the plan this implements).

usage:  python -m echostop.c1_eval                       # real run: logs/c1/nlp, c1_protocol.json (refuses if hashes differ)
        python -m echostop.c1_eval --mock logs/b1/nlp    # dry run on existing B1 NLP logs, writes nothing to figures/

Unit of inference: the DATASET (cluster). A job is (dataset, arm); both arms of a dataset enter or leave a bootstrap
resample together. Rules are scored on job-level risk (share of seeds with A_T - A_tau > 0.5 pp) and energy-weighted
saving (EW). Operating points are re-optimised inside every bootstrap resample, so the interval includes the selection."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import c1_rules as CR
from echostop import tierA
from echostop.calibration import precompute_score_trace

B = 10000
SEEDS_REQUIRED = 3


def load_logs(logdir):
    runs, meta = {}, {}
    for p in sorted(glob.glob(f"{logdir}/*.csv")):
        mp = p.replace(".csv", ".meta.json")
        if not os.path.exists(mp):
            continue                                                  # unfinished run
        m = re.match(r"^(.*)_([A-Za-z0-9]+)_seed(\d+)\.csv$", os.path.basename(p))
        name, arm, seed = m.groups()
        df = pd.read_csv(p)
        df["B_t"] = df["val_acc"].cummax()
        runs.setdefault(f"{name}|{arm}", {})[int(seed)] = df
        meta[f"{name}|{arm}"] = json.load(open(mp))
    jobs, dropped = {}, {}
    for k, d in runs.items():
        if len(d) < SEEDS_REQUIRED or len({len(x) for x in d.values()}) != 1:
            dropped[k] = f"{len(d)} complete seeds"
            continue
        jobs[k] = [d[s] for s in sorted(d)[:SEEDS_REQUIRED]]
    return jobs, meta, dropped


def lcpfn_traces(jobs, cache):
    store = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}
    miss = [k for k in jobs if k not in store]
    if miss:
        import torch
        from lcpfn import LCPFN
        from echostop import lcpfn_real as LP
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = LCPFN().to(dev)
        LP._patch_layers(model)
        model.eval()
        for k in miss:
            store[k] = [LP.pfn_trace(model, df, dev) for df in jobs[k]]
        pickle.dump(store, open(cache, "wb"))
    return {k: store[k] for k in jobs}


def matrices(jobs, with_pfn, pfn_cache):
    eco = {k: [precompute_score_trace(df, 0.5) for df in dfs] for k, dfs in jobs.items()}
    pfn = lcpfn_traces(jobs, pfn_cache) if with_pfn else None
    tr = {k: [{"eco": eco[k][i], "pfn": (pfn[k][i] if pfn else None)} for i in range(len(jobs[k]))] for k in jobs}
    fams = CR.families(with_pfn)
    M = {}
    for f, pl in fams.items():
        L, S, ES, et = tierA.evaluate_family(jobs, tr, pl)
        L, S, ES = CR.with_never_stop(L, S, ES)
        M[f] = dict(L=L, S=S, ES=ES, et=et, labels=[lab for lab, _ in pl] + ["never stop"])
    return M


def boot_weights(cluster_of_job, rng, nb=B):
    cl = np.unique(cluster_of_job)
    pos = {c: i for i, c in enumerate(cl)}
    ci = np.array([pos[c] for c in cluster_of_job])
    counts = rng.multinomial(len(cl), np.ones(len(cl)) / len(cl), size=nb)       # [nb, n_clusters]
    return counts[:, ci].astype(float)                                           # [nb, J] job weights


def front(W, L, ES, et, r):
    """W [nb, J] weights -> EW of the best feasible setting, per resample (and its index)"""
    risk = (W @ L.T) / W.sum(1, keepdims=True)                                   # [nb, P]
    ew = 100 * (W @ ES.T) / (W @ et)[:, None]
    ew = np.where(risk <= r + 1e-12, ew, -np.inf)
    return ew.max(1), ew.argmax(1)


def ci(x):
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def frontier_table(M, mask, cluster, rng, rules, base, risks):
    W1 = np.ones((1, mask.sum()))
    Wb = boot_weights(cluster[mask], rng)
    out = {}
    for r in risks:
        fb = {}
        for f in rules:
            m = M[f]
            pt, _ = front(W1, m["L"][:, mask], m["ES"][:, mask], m["et"][mask], r)
            bt, _ = front(Wb, m["L"][:, mask], m["ES"][:, mask], m["et"][mask], r)
            fb[f] = (float(pt[0]), bt)
        for f in rules:
            if f == base:
                out[(r, f)] = dict(ew=fb[f][0])
                continue
            d = fb[f][1] - fb[base][1]
            lo, hi = ci(d)
            out[(r, f)] = dict(ew=fb[f][0], base=fb[base][0], delta=fb[f][0] - fb[base][0], lo=lo, hi=hi,
                               p_le0=float((d <= 0).mean()))
    return out


def frozen_table(M, mask, cluster, rng, protocol, base, rules):
    Wb = boot_weights(cluster[mask], rng)
    W1 = np.ones((1, mask.sum()))
    rows = []
    for tag, table in protocol["frozen"].items():
        picks = {}
        for f in rules:
            lab = table.get(f)
            labs = M[f]["labels"]
            picks[f] = labs.index(lab) if lab in labs else len(labs) - 1               # None -> never stop
        stat = {}
        for f in rules:
            i = picks[f]
            L, ES, S, et = M[f]["L"][i, mask], M[f]["ES"][i, mask], M[f]["S"][i, mask], M[f]["et"][mask]
            pt = (L.mean(), 100 * ES.sum() / et.sum(), S.mean())
            bt_ew = 100 * (Wb @ ES) / (Wb @ et)
            bt_risk = (Wb @ L) / Wb.sum(1)
            stat[f] = (pt, bt_ew, bt_risk, M[f]["labels"][i])
        for f in rules:
            pt, bt_ew, bt_risk, lab = stat[f]
            row = dict(setting=tag, rule=f, param=lab, risk=pt[0], EW=pt[1], mean_saved=pt[2])
            if f != base:
                d = bt_ew - stat[base][1]
                row.update(dEW=pt[1] - stat[base][0][1], dEW_lo=ci(d)[0], dEW_hi=ci(d)[1])
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", default=None)
    ap.add_argument("--no-pfn", action="store_true")
    a = ap.parse_args()
    real = a.mock is None
    logdir = a.mock or "logs/c1/nlp"
    protocol = json.load(open("c1_protocol.json" if real else "cache/c1_protocol_dryrun.json"))
    if real:
        for f, h in protocol["hashes"].items():
            cur = hashlib.sha256(open(f, "rb").read()).hexdigest()
            if cur != h:
                raise SystemExit(f"HASH MISMATCH {f}: the frozen protocol and the code/registry/analysis differ. Refusing to evaluate.")
    jobs, meta, dropped = load_logs(logdir)
    names = sorted(jobs)
    ds = np.array([k.split("|")[0] for k in names]); arm = np.array([k.split("|")[1] for k in names])
    print(f"{len(names)} complete jobs, {len(set(ds))} datasets; dropped (incomplete): {dropped}")
    with_pfn = not a.no_pfn
    M = matrices({k: jobs[k] for k in names}, with_pfn, "cache/c1_lcpfn_traces.pkl" if real else "cache/c1_lcpfn_traces_mock.pkl")
    rules = [f for f in CR.FAMILIES if f in M]
    rng = np.random.default_rng(0)
    allmask = np.ones(len(names), bool)
    pd.set_option("display.width", 250, "display.max_colwidth", 40)

    print("\n=== PRIMARY: matched-risk frontier. EcoStop-SR minus Patience p=fT, min-delta at realised risk <= 0.10 ===")
    T = frontier_table(M, allmask, ds, rng, [CR.PRIMARY_RULE, CR.PRIMARY_BASE], CR.PRIMARY_BASE, [CR.PRIMARY_RISK])
    p = T[(CR.PRIMARY_RISK, CR.PRIMARY_RULE)]
    print(f"EcoStop-SR EW {p['ew']:.1f}% vs patience EW {p['base']:.1f}%: difference {p['delta']:+.2f} pp, "
          f"95% cluster-bootstrap CI [{p['lo']:+.2f}, {p['hi']:+.2f}]  (P[diff<=0] = {p['p_le0']:.4f})")
    verdict = "SUPPORTED (lower limit > 0)" if p["lo"] > 0 else "NOT SUPPORTED (lower limit <= 0)"
    print(f"H1 {verdict}")

    print("\n=== S4: per-dataset energy saved at the two primary operating points (full sample, risk <= 0.10) ===")
    from scipy.stats import binomtest
    sel = {}
    for f in (CR.PRIMARY_RULE, CR.PRIMARY_BASE):
        m_ = M[f]
        _, ix = front(np.ones((1, len(names))), m_["L"], m_["ES"], m_["et"], CR.PRIMARY_RISK)
        sel[f] = int(ix[0])
    rowsd = []
    for c in sorted(set(ds)):
        m_ = ds == c
        e_t = M[CR.ECO]["et"][m_].sum()
        a_ = M[CR.ECO]["ES"][sel[CR.ECO]][m_].sum()
        b_ = M[CR.PRIMARY_BASE]["ES"][sel[CR.PRIMARY_BASE]][m_].sum()
        rowsd.append(dict(dataset=c, EW_SR=100 * a_ / e_t, EW_pat=100 * b_ / e_t, diff=100 * (a_ - b_) / e_t))
    S4 = pd.DataFrame(rowsd)
    nb, nw = int((S4["diff"] > 1e-9).sum()), int((S4["diff"] < -1e-9).sum())
    pv = binomtest(nb, nb + nw, 0.5).pvalue if nb + nw else float("nan")
    print(f"settings: EcoStop-SR {M[CR.ECO]['labels'][sel[CR.ECO]]}, patience {M[CR.PRIMARY_BASE]['labels'][sel[CR.PRIMARY_BASE]]}; "
          f"datasets favouring SR {nb}, favouring patience {nw}, ties {len(S4) - nb - nw}; sign test p = {pv:.4f}")

    print("\n=== SECONDARY S1: frontier differences vs patience p=fT, min-delta, all risk levels (95% CI) ===")
    T = frontier_table(M, allmask, ds, rng, rules, CR.PRIMARY_BASE, CR.RISKS)
    rows = []
    for (r, f), v in T.items():
        rows.append(dict(risk_cap=r, rule=f, EW=v["ew"], **({k: v[k] for k in ("delta", "lo", "hi")} if "delta" in v else {})))
    S1 = pd.DataFrame(rows)
    print(S1.round(2).to_string(index=False))

    print("\n=== S2: per arm ===")
    S2 = []
    for ar in sorted(set(arm)):
        m = arm == ar
        T2 = frontier_table(M, m, ds, rng, [CR.ECO, CR.PRIMARY_BASE], CR.PRIMARY_BASE, CR.RISKS)
        for r in CR.RISKS:
            v = T2[(r, CR.ECO)]
            S2.append(dict(arm=ar, jobs=int(m.sum()), risk_cap=r, EW_SR=v["ew"], EW_pat=v["base"], delta=v["delta"], lo=v["lo"], hi=v["hi"]))
    S2 = pd.DataFrame(S2)
    print(S2.round(2).to_string(index=False))

    print("\n=== S3: frozen operating points calibrated on the 177 existing jobs (and on the 35 existing NLP jobs) ===")
    S3 = frozen_table(M, allmask, ds, rng, protocol, CR.PRIMARY_BASE, [f for f in rules if f in protocol["frozen"][next(iter(protocol["frozen"]))]])
    print(S3.round(3).to_string(index=False))

    print("\n=== S6: sensitivity (primary contrast at risk <= 0.10) ===")
    sens = []
    final_acc = np.array([np.mean([df["test_acc_at_checkpoint"].replace("", np.nan).astype(float).ffill().iloc[-1] for df in jobs[k]]) for k in names])
    maj = np.array([meta[k].get("test_majority_pct", np.nan) for k in names])
    learn = (final_acc - maj >= 3.0) | np.isnan(maj)
    T6 = frontier_table(M, learn, ds, rng, [CR.ECO, CR.PRIMARY_BASE], CR.PRIMARY_BASE, [0.10])[(0.10, CR.ECO)]
    sens.append(dict(variant=f"exclude jobs that barely learn (final acc - majority < 3 pp): {int((~learn).sum())} jobs removed",
                     delta=T6["delta"], lo=T6["lo"], hi=T6["hi"]))
    thin = {f: dict(v) for f, v in M.items()}
    keep = np.r_[np.arange(0, len(M[CR.ECO]["labels"]) - 1, 2), len(M[CR.ECO]["labels"]) - 1]
    for k_ in ("L", "S", "ES"):
        thin[CR.ECO][k_] = M[CR.ECO][k_][keep]
    T7 = frontier_table(thin, allmask, ds, rng, [CR.ECO, CR.PRIMARY_BASE], CR.PRIMARY_BASE, [0.10])[(0.10, CR.ECO)]
    sens.append(dict(variant="EcoStop-SR gamma grid thinned to every other value", delta=T7["delta"], lo=T7["lo"], hi=T7["hi"]))
    deltas = []
    for c in sorted(set(ds)):
        m = ds != c
        t = frontier_table(M, m, ds, rng, [CR.ECO, CR.PRIMARY_BASE], CR.PRIMARY_BASE, [0.10])[(0.10, CR.ECO)]
        deltas.append((t["delta"], c))
    sens.append(dict(variant=f"leave one dataset out: min {min(deltas)[0]:+.2f} ({min(deltas)[1]}), max {max(deltas)[0]:+.2f} ({max(deltas)[1]})",
                     delta=float(np.mean([d for d, _ in deltas])), lo=min(deltas)[0], hi=max(deltas)[0]))
    S6 = pd.DataFrame(sens)
    print(S6.round(2).to_string(index=False))

    print("\n=== S7: GPU state before runs (shared-GPU check) ===")
    util = np.array([meta[k].get("pre_util_pct_mean", np.nan) for k in names])
    pw = np.array([meta[k].get("pre_power_w_mean", np.nan) for k in names])
    print(f"pre-run GPU utilisation: median {np.nanmedian(util):.1f}%, max {np.nanmax(util):.1f}%;  pre-run board power: median {np.nanmedian(pw):.1f} W, max {np.nanmax(pw):.1f} W")

    if real:
        os.makedirs("figures/tierA", exist_ok=True)
        S1.to_csv("figures/tierA/c1_frontier.csv", index=False)
        S2.to_csv("figures/tierA/c1_per_arm.csv", index=False)
        S3.to_csv("figures/tierA/c1_frozen.csv", index=False)
        S6.to_csv("figures/tierA/c1_sensitivity.csv", index=False)
        S4.to_csv("figures/tierA/c1_per_dataset.csv", index=False)
        json.dump({"n_jobs": len(names), "n_datasets": len(set(ds)), "primary": p, "verdict": verdict, "dropped": dropped},
                  open("figures/tierA/c1_primary.json", "w"), indent=1, default=float)
        print("\nwritten: figures/tierA/c1_*.csv, c1_primary.json")


if __name__ == "__main__":
    main()
