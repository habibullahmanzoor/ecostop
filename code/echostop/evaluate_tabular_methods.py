"""Method comparison on the tabular/OpenML corpus (plan §21), where real
early-stopping signal actually exists -- unlike the vision pilot's cosine-
schedule curves, where every online method (including patience) essentially
never fires (see session finding). Evaluates: oracle, patience,
calibrated EcoStop (fixed gamma_hat, see evaluate_calibrated.py for why it
changed from 0.2516 to 0.0663 after the predictor fix), LC-PFN-style
(fixed 0.05, zero-margin).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from echostop.calibration import precompute_score_trace, simulate_from_trace
from echostop.gating_tests import patience_stop, PATIENCE_GRID, eps_tolerance_oracle
from echostop.baseline_lcpfn_style import lcpfn_style_stop

GAMMA_HAT = 0.54  # certified at rho=0.20, beta=0.2, n_calib=130, non-degenerate threshold (anytime-valid: rho=0.23)
EPS = 0.5


def discover_all_tabular_jobs() -> dict[str, list[pd.DataFrame]]:
    runs: dict[str, list[pd.DataFrame]] = {}
    for base_dir, prefix in [("logs/calibration_tabular", "tabular_"),
                              ("logs/calibration_openml", "openml_")]:
        for csv_path in sorted(Path(base_dir).glob("*.csv")):
            stem = csv_path.stem
            # NOT stem.split("_seed")[0] -- a dataset literally named "seeds"
            # (OpenML batch 5) makes that split fire on the wrong "_seed"
            # substring (inside "seeds" itself), truncating to just the
            # prefix. Strip the prefix first, then the trailing _seedN only.
            dataset = re.sub(r"_seed\d+$", "", stem[len(prefix):] if stem.startswith(prefix) else stem)
            df = pd.read_csv(csv_path)
            if len(df) < 60:
                continue
            df["B_t"] = df["val_acc"].cummax()
            runs.setdefault(dataset, []).append(df)
    return runs


def main() -> None:
    jobs = discover_all_tabular_jobs()
    records = []

    for job_name, dfs in jobs.items():
        for i, df in enumerate(dfs):
            q_max = df["val_acc"].max()
            t_oracle, e_oracle = eps_tolerance_oracle(df, EPS)
            full_energy = df["cumulative_energy_joules"].iloc[-1]

            # patience, best feasible
            best_p = None
            for p in PATIENCE_GRID:
                t_p, e_p = patience_stop(df, p)
                if df["B_t"].iloc[t_p] >= q_max - EPS and (best_p is None or e_p < best_p[1]):
                    best_p = (p, e_p, t_p)
            e_patience = best_p[1] if best_p else full_energy
            feas_patience = best_p is not None

            # calibrated EcoStop, fixed gamma
            trace = precompute_score_trace(df, EPS)
            t_cal = simulate_from_trace(trace, GAMMA_HAT, len(df))
            e_cal = float(df["cumulative_energy_joules"].iloc[t_cal])
            feas_cal = df["B_t"].iloc[t_cal] >= q_max - EPS

            # LC-PFN-style
            t_lcpfn = lcpfn_style_stop(df)
            e_lcpfn = float(df["cumulative_energy_joules"].iloc[t_lcpfn])
            feas_lcpfn = df["B_t"].iloc[t_lcpfn] >= q_max - EPS

            records.append({
                "job": job_name, "seed_idx": i,
                "oracle_energy_j": e_oracle, "full_energy_j": full_energy,
                "patience_energy_j": e_patience, "patience_feasible": feas_patience,
                "patience_fired_early": e_patience < full_energy * 0.999,
                "calibrated_energy_j": e_cal, "calibrated_feasible": feas_cal,
                "calibrated_fired_early": e_cal < full_energy * 0.999,
                "lcpfn_energy_j": e_lcpfn, "lcpfn_feasible": feas_lcpfn,
                "lcpfn_fired_early": e_lcpfn < full_energy * 0.999,
            })

    df_out = pd.DataFrame(records)
    df_out.to_csv("figures/tabular_method_comparison.csv", index=False)

    print(f"n = {len(df_out)} seed-runs across {df_out.job.nunique()} distinct tabular/OpenML jobs\n")

    for method in ["patience", "calibrated", "lcpfn"]:
        fired = df_out[f"{method}_fired_early"].mean() * 100
        feas = df_out[f"{method}_feasible"].mean() * 100
        e_col = f"{method}_energy_j"
        regret = 100 * (df_out[e_col] - df_out["oracle_energy_j"]) / df_out["oracle_energy_j"]
        feas_mask = df_out[f"{method}_feasible"]
        print(f"{method:12s}: fired early {fired:5.1f}% of runs | feasible {feas:5.1f}% | "
              f"mean regret (all) {regret.mean():6.1f}% | "
              f"mean regret (feasible only, n={feas_mask.sum()}) {regret[feas_mask].mean():6.1f}%")


if __name__ == "__main__":
    main()
