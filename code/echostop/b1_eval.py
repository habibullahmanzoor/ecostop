"""Evaluate the FROZEN protocol (b1_protocol.json) on the B1 fresh jobs. Verifies hashes first.
Reports, per target (rho, beta) and per calibration scheme (pooled / per-domain):
realised per-run risk with a job-bootstrap CI and exact one-sided binomial-style check,
mean and energy-weighted savings, per domain.  usage: python -m echostop.b1_eval [lcpfn]
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.calibration import forward_fill_audit, precompute_score_trace, simulate_from_trace
from echostop.tierA import patience_stop_md

EPS = 0.5
RNG = np.random.default_rng(0)


def check_hashes(proto):
    import hashlib
    bad = [f for f, h in proto["hashes"].items() if hashlib.sha256(open(f, "rb").read()).hexdigest() != h]
    if bad:
        raise SystemExit(f"FROZEN FILES CHANGED: {bad}. Fresh-test results would not be pre-registered.")


def load_b1():
    jobs, dom = {}, {}
    for d in ("vision", "nlp", "tabular"):
        for p in sorted(glob.glob(f"logs/b1/{d}/*.csv")):
            df = pd.read_csv(p)
            meta = json.load(open(p.replace(".csv", ".meta.json"))) if os.path.exists(p.replace(".csv", ".meta.json")) else None
            if meta is None:        # unfinished run
                continue
            job = re.sub(r"_seed\d+\.csv$", "", os.path.basename(p))
            df["B_t"] = df["val_acc"].cummax()
            jobs.setdefault(job, []).append(df)
            dom[job] = d
    return jobs, dom


def stop_fn(family, label):
    v = float(label.split("=")[1])
    if family == "EcoStop":
        return lambda df, tr: simulate_from_trace(tr, v, len(df))
    if family == "Patience (fixed p)":
        return lambda df, tr: patience_stop_md(df, int(v))
    if family.startswith("Patience (p=f*T"):
        d = 0.1 if "min-delta" in family else 0.0
        return lambda df, tr: patience_stop_md(df, max(1, round(v * len(df))), d)
    if family == "Fixed budget fraction":
        return lambda df, tr: int(round(v * (len(df) - 1)))
    if family == "LC-PFN + EcoStop gates":
        from echostop.ablation import sim
        return lambda df, tr: sim(tr, v, len(df), max(2, round(0.125 * len(df))), 2)
    raise ValueError(family)


def score(jobs, traces, fn):
    names = list(jobs)
    loss, saved, es, et = [], [], 0.0, 0.0
    for k in names:
        ls, ss = [], []
        for df, tr in zip(jobs[k], traces[k]):
            t = fn(df, tr)
            a = forward_fill_audit(df)
            e = df["cumulative_energy_joules"].values
            ls.append(float(a[-1] - a[t] > EPS)); ss.append(100 * (1 - e[t] / e[-1]) if e[-1] > 0 else 0.0)
            es += e[-1] - e[t]; et += e[-1]
        loss.append(np.mean(ls)); saved.append(np.mean(ss))
    return np.array(loss), np.array(saved), (100 * es / et if et > 0 else 0.0)


def boot_ci(x, B=4000):
    idx = RNG.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    proto = json.load(open("b1_protocol.json"))
    check_hashes(proto)
    jobs, dom = load_b1()
    names = list(jobs)
    print(f"{len(names)} fresh jobs, {sum(len(v) for v in jobs.values())} runs; "
          + str({d: sum(v == d for v in dom.values()) for d in ('vision', 'nlp', 'tabular')}))
    traces = {k: [precompute_score_trace(df, EPS) for df in dfs] for k, dfs in jobs.items()}
    if len(sys.argv) > 1 and sys.argv[1] == "lcpfn":
        pf = pickle.load(open("cache/b1_lcpfn_traces.pkl", "rb"))
    dom_arr = np.array([dom[k] for k in names])
    rows = []
    for tkey, fams in proto["targets"].items():
        rho = float(tkey.split(",")[0].split("=")[1])
        for family, entry in fams.items():
            if family.startswith("LC-PFN"):
                if not (len(sys.argv) > 1 and sys.argv[1] == "lcpfn"):
                    continue
                tr_use = pf
            else:
                tr_use = traces
            for scheme in ("pooled", "per_domain"):
                loss = np.zeros(len(names)); saved = np.zeros(len(names)); ew_num = ew_den = 0.0
                for d in ("vision", "nlp", "tabular"):
                    idx = np.where(dom_arr == d)[0]
                    if len(idx) == 0:
                        continue
                    lab = entry["pooled"] if scheme == "pooled" else entry["per_domain"][d]
                    sub = {names[i]: jobs[names[i]] for i in idx}
                    if lab is None:  # not certifiable -> run to completion
                        loss[idx] = 0.0; saved[idx] = 0.0
                        et = sum(df["cumulative_energy_joules"].iloc[-1] for k in sub for df in sub[k])
                        ew_den += et
                        continue
                    l, s, ew = score(sub, {k: tr_use[k] for k in sub}, stop_fn(family, lab))
                    et = sum(df["cumulative_energy_joules"].iloc[-1] for k in sub for df in sub[k])
                    loss[idx], saved[idx] = l, s
                    ew_num += ew / 100 * et; ew_den += et
                lo, hi = boot_ci(loss)
                rows.append({"rho": rho, "method": family, "scheme": scheme, "jobs": len(names),
                             "risk": loss.mean(), "risk_ci_lo": lo, "risk_ci_hi": hi,
                             "risk_vision": loss[dom_arr == "vision"].mean() if (dom_arr == "vision").any() else np.nan,
                             "risk_nlp": loss[dom_arr == "nlp"].mean() if (dom_arr == "nlp").any() else np.nan,
                             "risk_tabular": loss[dom_arr == "tabular"].mean() if (dom_arr == "tabular").any() else np.nan,
                             "mean_saved": saved.mean(), "energy_weighted_saved": 100 * ew_num / ew_den})
    out = pd.DataFrame(rows)
    os.makedirs("figures/tierA", exist_ok=True)
    out.to_csv("figures/tierA/b1_fresh_test.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
