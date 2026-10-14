"""Analysis for B2 (freeze/thaw variable-cost regime) and B4 (energy-measurement validation).
Writes figures/tierA/b2_results.csv, b4_results.json and tables/b2.tex, tables/b4.tex."""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E
from echostop.calibration import forward_fill_audit, precompute_score_trace

EPS = 0.5
NL, BS = chr(10), chr(92)


def load_dir(pattern, key_re):
    jobs = {}
    for p in sorted(glob.glob(pattern)):
        if not os.path.exists(p.replace(".csv", ".meta.json")):
            continue
        df = pd.read_csv(p)
        df["B_t"] = df["val_acc"].cummax()
        jobs.setdefault(re.sub(key_re, "", os.path.basename(p)), []).append(df)
    return jobs


def oracle_stop(df):
    q = df["val_acc"].max()
    return int(df.index[df["B_t"] >= q - EPS][0])


def regime_stats(jobs):
    """Epoch-saved vs energy-saved at the eps-tolerance oracle stop, per job (mean over seeds)."""
    rows = []
    for k, dfs in jobs.items():
        ep, en = [], []
        for df in dfs:
            t = oracle_stop(df)
            e = df["cumulative_energy_joules"].values
            ep.append(100 * (1 - (t + 1) / len(df)))
            en.append(100 * (1 - e[t] / e[-1]))
        rows.append({"job": k, "oracle_epochs_saved": np.mean(ep), "oracle_energy_saved": np.mean(en)})
    return pd.DataFrame(rows)


def b2():
    proto = json.load(open("b1_protocol.json"))
    E.check_hashes(proto)
    ft = load_dir("logs/b2/*_resnet9freezethaw_seed*.csv", r"_resnet9freezethaw_seed\d+\.csv$")
    if not ft:
        print("no B2 runs yet"); return
    corpus, _ = load_corpus_counterparts(list(ft))
    rs_ft, rs_c = regime_stats(ft), regime_stats(corpus) if corpus else pd.DataFrame()
    out = {"n_jobs": len(ft), "oracle_epochs_saved_ft": rs_ft.oracle_epochs_saved.mean(),
           "oracle_energy_saved_ft": rs_ft.oracle_energy_saved.mean(),
           "gap_ft_pp": (rs_ft.oracle_energy_saved - rs_ft.oracle_epochs_saved).mean()}
    if len(rs_c):
        m = rs_ft.merge(rs_c, on="job", suffixes=("_ft", "_c"))
        out.update({"oracle_epochs_saved_ctrl": m.oracle_epochs_saved_c.mean(), "oracle_energy_saved_ctrl": m.oracle_energy_saved_c.mean(),
                    "gap_ctrl_pp": (m.oracle_energy_saved_c - m.oracle_epochs_saved_c).mean(), "n_paired": len(m)})
    # frozen rules on the freeze/thaw jobs
    tr = {k: [precompute_score_trace(df, EPS) for df in dfs] for k, dfs in ft.items()}
    rows = []
    for tkey, fams in proto["targets"].items():
        rho = float(tkey.split(",")[0].split("=")[1])
        for fam in ("EcoStop", "Patience (p=f*T, min-delta 0.1)", "Patience (p=f*T)", "Patience (fixed p)"):
            for scheme in ("pooled", "per_domain"):
                lab = fams[fam]["pooled"] if scheme == "pooled" else fams[fam]["per_domain"]["vision"]
                if lab is None:  # not certifiable: run to completion
                    rows.append({"rho": rho, "rule": fam, "scheme": scheme, "param": "none", "risk": 0.0,
                                 "epochs_saved": 0.0, "energy_saved_mean": 0.0, "energy_saved_weighted": 0.0})
                    continue
                fn = E.stop_fn(fam, lab)
                l, s_, ew = E.score(ft, tr, fn)
                ep_saved = [100 * (1 - (fn(df, t_) + 1) / len(df)) for k in ft for df, t_ in zip(ft[k], tr[k])]
                rows.append({"rho": rho, "rule": fam, "scheme": scheme, "param": lab, "risk": l.mean(),
                             "epochs_saved": np.mean(ep_saved), "energy_saved_mean": s_.mean(), "energy_saved_weighted": ew})
    pd.DataFrame(rows).to_csv("figures/tierA/b2_results.csv", index=False)
    json.dump(out, open("figures/tierA/b2_summary.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float)); print(pd.DataFrame(rows).round(3).to_string(index=False))


def load_corpus_counterparts(ds_names):
    from echostop import tierA
    jobs, _, dom = tierA.load()
    keep = {}
    for k in jobs:
        for d in ds_names:
            if k == f"{d}_resnet9":
                keep[d] = jobs[k]
    return keep, dom


def b4():
    out = {}
    p = "logs/b4/idle_and_sampling.json"
    if os.path.exists(p):
        out.update(json.load(open(p)))
    reps = load_dir("logs/b4/semeion_resnet9_repeat*_seed0.csv", r"_seed0\.csv$")
    reps = [df for v in reps.values() for df in v]
    if len(reps) >= 2:
        e = np.array([df["cumulative_energy_joules"].iloc[-1] for df in reps])
        d = np.array([df["cumulative_duration_seconds"].iloc[-1] for df in reps])
        acc = np.array([df["val_acc"].max() for df in reps])
        out.update({"repeats": len(reps), "energy_J_mean": e.mean(), "energy_J_cv_pct": 100 * e.std(ddof=1) / e.mean(),
                    "duration_s_cv_pct": 100 * d.std(ddof=1) / d.mean(), "energy_J_minmax": [e.min(), e.max()],
                    "best_val_acc_range": [acc.min(), acc.max()]})
        # does run-to-run energy noise change the stopping decision? same rule on each repeat
        stops = []
        for df in reps:
            tr = precompute_score_trace(df, EPS)
            from echostop.calibration import simulate_from_trace
            t = simulate_from_trace(tr, 0.46, len(df))
            e_ = df["cumulative_energy_joules"].values
            stops.append((t, 100 * (1 - e_[t] / e_[-1])))
        out["frozen_ecostop_stop_epochs"] = [int(s[0]) for s in stops]
        out["frozen_ecostop_energy_saved_pct"] = [float(s[1]) for s in stops]
    # idle-subtracted savings on the full development corpus
    if "idle_power_w_60s" in out:
        from echostop import tierA
        from echostop.calibration import simulate_from_trace
        jobs, traces, dom = tierA.load()
        P0 = out["idle_power_w_60s"]
        raw, dyn = {"saved": 0.0, "tot": 0.0}, {"saved": 0.0, "tot": 0.0}
        for k in jobs:
            for df, tr in zip(jobs[k], traces[k]):
                t = simulate_from_trace(tr, 0.46, len(df))
                e = df["cumulative_energy_joules"].values
                s = df["cumulative_duration_seconds"].values
                ed = e - P0 * s
                raw["saved"] += e[-1] - e[t]; raw["tot"] += e[-1]
                dyn["saved"] += ed[-1] - ed[t]; dyn["tot"] += ed[-1]
        out["corpus_energy_weighted_saved_raw_pct"] = 100 * raw["saved"] / raw["tot"]
        out["corpus_energy_weighted_saved_idle_subtracted_pct"] = 100 * dyn["saved"] / dyn["tot"]
        out["idle_share_of_total_energy_pct"] = 100 * (1 - dyn["tot"] / raw["tot"])
    json.dump(out, open("figures/tierA/b4_results.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs("figures/tierA", exist_ok=True)
    if which in ("b4", "all"):
        b4()
    if which in ("b2", "all"):
        b2()
