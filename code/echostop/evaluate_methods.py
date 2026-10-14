"""Compare all stopping methods against the oracle on the pilot data
(plan §21 Main Result Format). No new training runs -- pure analysis on
the CSVs already collected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from echostop.analyze_pilot import discover_runs, EPS_LADDER
from echostop.gating_tests import eps_tolerance_oracle, patience_stop, PATIENCE_GRID
from echostop.methods import gain_per_joule_stop, predictive_point_stop, ecostop_online_uncalibrated

TAU_GRID = np.logspace(-5, -1, 15)
RHO_GRID = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]


def evaluate_run(df: pd.DataFrame, eps: float) -> list[dict]:
    q_max = df["val_acc"].max()
    t_oracle, e_oracle = eps_tolerance_oracle(df, eps)
    rows = []

    def feasible_energy(t: int) -> tuple[float, bool]:
        e = float(df["cumulative_energy_joules"].iloc[t])
        is_feasible = df["B_t"].iloc[t] >= q_max - eps
        return e, is_feasible

    # oracle (reference row, regret = 0 by construction)
    rows.append({"method": "oracle", "stop_epoch": t_oracle, "energy_j": e_oracle,
                  "feasible": True, "param": None})

    # patience -- best feasible over the grid (plan §19.1 "oracle-tuned envelope")
    best = None
    for p in PATIENCE_GRID:
        t, e = patience_stop(df, p)
        e_check, feas = feasible_energy(t)
        if feas and (best is None or e < best[1]):
            best = (p, e, t)
    if best:
        rows.append({"method": "patience", "stop_epoch": best[2], "energy_j": best[1],
                      "feasible": True, "param": best[0]})
    else:
        rows.append({"method": "patience", "stop_epoch": len(df) - 1,
                      "energy_j": float(df["cumulative_energy_joules"].iloc[-1]),
                      "feasible": False, "param": None})

    # B7 gain-per-joule -- best feasible tau
    best = None
    for tau in TAU_GRID:
        t = gain_per_joule_stop(df, tau)
        e, feas = feasible_energy(t)
        if feas and (best is None or e < best[1]):
            best = (tau, e, t)
    if best:
        rows.append({"method": "gain_per_joule", "stop_epoch": best[2], "energy_j": best[1],
                      "feasible": True, "param": best[0]})
    else:
        rows.append({"method": "gain_per_joule", "stop_epoch": len(df) - 1,
                      "energy_j": float(df["cumulative_energy_joules"].iloc[-1]),
                      "feasible": False, "param": None})

    # predictive point-estimate -- parameterized directly by eps, no separate sweep
    t = predictive_point_stop(df, eps)
    e, feas = feasible_energy(t)
    rows.append({"method": "predictive_point", "stop_epoch": t, "energy_j": e,
                  "feasible": feas, "param": eps})

    # EcoStop-Online (uncalibrated) -- best feasible rho
    best = None
    for rho in RHO_GRID:
        t, _trace = ecostop_online_uncalibrated(df, eps, rho)
        e, feas = feasible_energy(t)
        if feas and (best is None or e < best[1]):
            best = (rho, e, t)
    if best:
        rows.append({"method": "ecostop_uncalibrated", "stop_epoch": best[2], "energy_j": best[1],
                      "feasible": True, "param": best[0]})
    else:
        rows.append({"method": "ecostop_uncalibrated", "stop_epoch": len(df) - 1,
                      "energy_j": float(df["cumulative_energy_joules"].iloc[-1]),
                      "feasible": False, "param": None})

    for r in rows:
        r["oracle_energy_j"] = e_oracle
        r["regret_pct"] = 100 * (r["energy_j"] - e_oracle) / e_oracle
    return rows


def main() -> None:
    runs = discover_runs()
    records = []
    for (dataset, model), dfs in runs.items():
        for i, df in enumerate(dfs):
            for eps in EPS_LADDER:
                for row in evaluate_run(df, eps):
                    row.update({"dataset": dataset, "model": model, "seed_idx": i, "eps": eps})
                    records.append(row)

    full_df = pd.DataFrame(records)
    Path("figures").mkdir(exist_ok=True)
    full_df.to_csv("figures/method_comparison_full.csv", index=False)

    print("Feasibility rate by method (plan §11.6 -- feasibility before efficiency):")
    print(full_df.groupby("method")["feasible"].mean().sort_values(ascending=False).to_string())

    print("\nMean regret (% of oracle energy), feasible runs only, by method and eps:")
    feasible_only = full_df[full_df.feasible]
    summary = (
        feasible_only.groupby(["method", "eps"])["regret_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    print(summary.to_string(index=False))
    summary.to_csv("figures/method_comparison_summary.csv", index=False)

    print("\nMean regret (% of oracle energy), feasible only, averaged over eps and cells:")
    print(feasible_only.groupby("method")["regret_pct"].mean().sort_values().to_string())


if __name__ == "__main__":
    main()
