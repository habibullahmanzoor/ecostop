"""Regenerate figures/energy_report_data.json (feeding energy_report.md) on
the full corpus. Per-dataset (job) rows, seeds aggregated to means. Includes
EcoStop-Meta as a 5th method and, for every method, the mean audited
accuracy gap (A_full - A_method, in pp) alongside the feasibility
percentage -- matching the paper's §6.6/§6.7 analysis, at dataset (not just
domain) granularity."""

from __future__ import annotations

import json

import pandas as pd

from echostop.analyze_pilot import discover_runs
from echostop.calibration import (
    precompute_score_trace, simulate_from_trace, audited_loss, forward_fill_audit,
)
from echostop.gating_tests import eps_tolerance_oracle, patience_stop, meta_fraction_stop
from echostop.baseline_lcpfn_style import lcpfn_style_stop
from echostop.run_calibration_demo2 import discover_tabular_runs
from echostop.run_calibration_final_split import discover_nlp_runs

GAMMA_HAT = 0.54
P_HAT = 10
F_HAT = 0.927
EPS = 0.5


def eval_job(domain: str, dataset: str, dfs: list[pd.DataFrame]) -> dict:
    methods = ["oracle", "calibrated", "patience", "lcpfn", "meta"]
    acc = {m: {"saved": [], "feas": [], "gap": []} for m in methods}
    full_es = []

    for df in dfs:
        full_e = float(df["cumulative_energy_joules"].iloc[-1])
        full_es.append(full_e)
        a_t = forward_fill_audit(df)
        a_full = a_t[-1]

        t_oracle, e_oracle = eps_tolerance_oracle(df, EPS)
        acc["oracle"]["saved"].append(100 * (1 - e_oracle / full_e))
        acc["oracle"]["feas"].append(100 * (1 - audited_loss(df, t_oracle, EPS)))
        acc["oracle"]["gap"].append(float(a_full - a_t[t_oracle]))

        trace = precompute_score_trace(df, EPS)
        t_online = simulate_from_trace(trace, GAMMA_HAT, len(df))
        e_online = float(df["cumulative_energy_joules"].iloc[t_online])
        acc["calibrated"]["saved"].append(100 * (1 - e_online / full_e))
        acc["calibrated"]["feas"].append(100 * (1 - audited_loss(df, t_online, EPS)))
        acc["calibrated"]["gap"].append(float(a_full - a_t[t_online]))

        t_pat, e_pat = patience_stop(df, P_HAT)
        acc["patience"]["saved"].append(100 * (1 - e_pat / full_e))
        acc["patience"]["feas"].append(100 * (1 - audited_loss(df, t_pat, EPS)))
        acc["patience"]["gap"].append(float(a_full - a_t[t_pat]))

        t_lc = lcpfn_style_stop(df)
        e_lc = float(df["cumulative_energy_joules"].iloc[t_lc])
        acc["lcpfn"]["saved"].append(100 * (1 - e_lc / full_e))
        acc["lcpfn"]["feas"].append(100 * (1 - audited_loss(df, t_lc, EPS)))
        acc["lcpfn"]["gap"].append(float(a_full - a_t[t_lc]))

        t_meta, e_meta = meta_fraction_stop(df, F_HAT)
        acc["meta"]["saved"].append(100 * (1 - e_meta / full_e))
        acc["meta"]["feas"].append(100 * (1 - audited_loss(df, t_meta, EPS)))
        acc["meta"]["gap"].append(float(a_full - a_t[t_meta]))

    n = len(dfs)
    prefix = {"oracle": "oracle", "calibrated": "calibrated", "patience": "patience",
              "lcpfn": "lcpfn", "meta": "meta"}
    out = {
        "domain": domain, "dataset": dataset, "n_seeds": float(n),
        "full_e_mean": round(sum(full_es) / n, 1),
    }
    for m, pfx in prefix.items():
        out[f"{pfx}_saved_pct"] = round(sum(acc[m]["saved"]) / n, 1)
        out[f"{pfx}_feasible_pct"] = round(sum(acc[m]["feas"]) / n, 1)
        out[f"{pfx}_gap_pp"] = round(sum(acc[m]["gap"]) / n, 2)
    return out


def main() -> None:
    records = []

    for (dataset, model), dfs in discover_runs().items():
        records.append(eval_job("Vision (cosine, from-scratch)", f"{dataset}_{model}", dfs))

    step_dfs = []
    for seed in [0, 1, 2]:
        df = pd.read_csv(f"logs/step_decay_test/stepdecay_cifar10_resnet9_seed{seed}.csv")
        df["B_t"] = df["val_acc"].cummax()
        step_dfs.append(df)
    records.append(eval_job("Vision (step-decay, from-scratch)", "cifar10_resnet9_stepdecay", step_dfs))

    for job, dfs in discover_tabular_runs().items():
        records.append(eval_job("Tabular / OpenML", job, dfs))

    for name, dfs in discover_nlp_runs().items():
        clean_name = name.replace("_distilbert", "")
        records.append(eval_job("NLP (fine-tuning)", clean_name, dfs))

    with open("figures/energy_report_data.json", "w") as f:
        json.dump(records, f)

    print(f"Total jobs: {len(records)}")
    by_domain = {}
    for r in records:
        by_domain.setdefault(r["domain"], []).append(r)
    for d, rows in by_domain.items():
        print(f"  {d}: {len(rows)} jobs")


if __name__ == "__main__":
    main()
