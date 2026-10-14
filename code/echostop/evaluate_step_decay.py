"""Deploy the calibrated gamma_hat on the step-decay bounded test (plan
§24 A11), out-of-sample relative to the calibration corpus -- step-decay
data was never part of run_calibration_final_split.py's calibration or
evaluation pools, so this is a genuine held-out check of whether the fix
in methods.fit_exp_saturation (session diagnostic: windowed fit + boundary-
degeneracy fallback + parameter-uncertainty sigma) actually lets the
predictor exploit step-decay's genuine post-drop plateaus, not just the
cosine pilot it was tuned against.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from echostop.calibration import precompute_score_trace, simulate_from_trace
from echostop.gating_tests import patience_stop, PATIENCE_GRID, eps_tolerance_oracle

GAMMA_HAT = 0.54  # certified at rho=0.20, beta=0.2, n_calib=130, non-degenerate threshold (anytime-valid: rho=0.23)
EPS = 0.5


def main() -> None:
    records = []
    for seed in [0, 1, 2]:
        path = Path(f"logs/step_decay_test/stepdecay_cifar10_resnet9_seed{seed}.csv")
        df = pd.read_csv(path)
        df["B_t"] = df["val_acc"].cummax()
        q_max = df["val_acc"].max()
        full_e = df["cumulative_energy_joules"].iloc[-1]

        t_oracle, e_oracle = eps_tolerance_oracle(df, EPS)

        trace = precompute_score_trace(df, EPS)
        t_cal = simulate_from_trace(trace, GAMMA_HAT, len(df))
        e_cal = float(df["cumulative_energy_joules"].iloc[t_cal])
        feas_cal = bool(df["B_t"].iloc[t_cal] >= q_max - EPS)

        best_p = None
        for p in PATIENCE_GRID:
            t_p, e_p = patience_stop(df, p)
            if df["B_t"].iloc[t_p] >= q_max - EPS and (best_p is None or e_p < best_p[1]):
                best_p = (p, e_p, t_p)
        e_patience = best_p[1] if best_p else full_e
        t_patience = best_p[2] if best_p else len(df) - 1

        records.append({
            "seed": seed, "oracle_epoch": t_oracle, "oracle_energy_j": e_oracle,
            "calibrated_epoch": t_cal, "calibrated_energy_j": e_cal,
            "calibrated_feasible": feas_cal,
            "calibrated_energy_saved_pct": 100 * (1 - e_cal / full_e),
            "calibrated_regret_pct": 100 * (e_cal - e_oracle) / e_oracle,
            "patience_epoch": t_patience, "patience_energy_j": e_patience,
            "patience_energy_saved_pct": 100 * (1 - e_patience / full_e),
            "patience_regret_pct": 100 * (e_patience - e_oracle) / e_oracle,
            "full_energy_j": full_e,
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv("figures/step_decay_evaluation.csv", index=False)
    print(df_out.to_string(index=False))
    print(f"\nMean energy saved: calibrated {df_out.calibrated_energy_saved_pct.mean():.1f}% | "
          f"patience {df_out.patience_energy_saved_pct.mean():.1f}%")
    print(f"Feasibility: calibrated {df_out.calibrated_feasible.mean()*100:.0f}%")


if __name__ == "__main__":
    main()
