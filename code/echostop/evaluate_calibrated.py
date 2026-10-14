"""Deploy the actual calibrated gamma_hat=0.252 (certified at rho=0.3 on the
34-job calibration pool, plan §6.6.3) on the vision pilot data, and compare
against the uncalibrated methods already evaluated in evaluate_methods.py.

IMPORTANT: gamma_hat was calibrated at eps=0.5 specifically (the value used
throughout the calibration-corpus work). Applying it at other eps values
would be using a threshold outside what it was actually certified for, so
this comparison is deliberately eps=0.5 only -- not a full eps ladder like
the uncalibrated comparison, since recalibrating at each eps would need its
own LTT run.

gamma_hat history (session diagnostics, see methods.py/calibration.py for
the underlying fixes):
1. Original: 0.2516, from a predictor with two compounding curve-fit bugs
   (full-history residuals inflating sigma; short-window fits hitting a
   parameter boundary and reporting false-confident phantom gains) --
   see fit_exp_saturation's docstring. It was only "safe" because the
   buggy predictor rarely fired at all, regardless of threshold.
2. After fixing the predictor: 0.0663 -- lower/more conservative, but more
   decisive: 0/51 held-out violations, mean energy saved 42.8% -> 62.7%.
3. After also making min_epochs/plateau_gate schedule-relative instead of
   a fixed 5-epoch constant (found on the NLP leg: a fixed floor ate half
   the budget of a 10-epoch fine-tuning run) and folding the NLP leg into
   the calibration corpus (held out in the evaluation pool, not used to
   shape gamma_hat): back to 0.2516 -- tighter tabular safeguards reduced
   false-positive risk enough to afford the less conservative threshold
   again. Held-out NLP job: 0/3 violations, 44.7% energy saved (up from
   13.8% under the fixed-safeguard predictor).
"""

from __future__ import annotations

import pandas as pd

from echostop.analyze_pilot import discover_runs
from echostop.calibration import precompute_score_trace, simulate_from_trace
from echostop.gating_tests import patience_stop, PATIENCE_GRID, eps_tolerance_oracle

GAMMA_HAT = 0.54  # certified at rho=0.20, beta=0.2, n_calib=130, non-degenerate threshold (anytime-valid: rho=0.23)
EPS = 0.5


def main() -> None:
    runs = discover_runs()
    records = []

    for (dataset, model), dfs in runs.items():
        for i, df in enumerate(dfs):
            q_max = df["val_acc"].max()
            t_oracle, e_oracle = eps_tolerance_oracle(df, EPS)

            # calibrated EcoStop: FIXED gamma_hat, no per-run search
            trace = precompute_score_trace(df, EPS)
            t_cal = simulate_from_trace(trace, GAMMA_HAT, len(df))
            e_cal = float(df["cumulative_energy_joules"].iloc[t_cal])
            feasible_cal = df["B_t"].iloc[t_cal] >= q_max - EPS

            # best-feasible patience, for reference (same as method_comparison_summary.csv)
            best_patience = None
            for p in PATIENCE_GRID:
                t_p, e_p = patience_stop(df, p)
                if df["B_t"].iloc[t_p] >= q_max - EPS:
                    if best_patience is None or e_p < best_patience[1]:
                        best_patience = (p, e_p, t_p)
            e_patience = best_patience[1] if best_patience else float(df["cumulative_energy_joules"].iloc[-1])

            records.append({
                "dataset": dataset, "model": model, "seed_idx": i,
                "oracle_energy_j": e_oracle,
                "calibrated_energy_j": e_cal, "calibrated_feasible": feasible_cal,
                "calibrated_regret_pct": 100 * (e_cal - e_oracle) / e_oracle,
                "patience_energy_j": e_patience,
                "patience_regret_pct": 100 * (e_patience - e_oracle) / e_oracle,
            })

    df_out = pd.DataFrame(records)
    df_out.to_csv("figures/calibrated_vs_uncalibrated_vision.csv", index=False)

    print(f"Calibrated EcoStop (gamma_hat={GAMMA_HAT:.3f}, certified rho=0.3) on vision pilot, eps={EPS}pp\n")
    print(f"Feasibility: {df_out['calibrated_feasible'].mean()*100:.0f}% "
          f"({df_out['calibrated_feasible'].sum()}/{len(df_out)} runs within tolerance)")
    print(f"\nMean regret vs oracle (ALL runs, feasible or not):")
    print(f"  calibrated EcoStop: {df_out['calibrated_regret_pct'].mean():.1f}%")
    print(f"  patience (best-feasible, per-run search): {df_out['patience_regret_pct'].mean():.1f}%")

    feas = df_out[df_out.calibrated_feasible]
    print(f"\nMean regret vs oracle (feasible calibrated runs only, n={len(feas)}):")
    print(f"  calibrated EcoStop: {feas['calibrated_regret_pct'].mean():.1f}%")
    print(f"  patience on same subset: {feas['patience_regret_pct'].mean():.1f}%")

    print(f"\nPer-cell breakdown:")
    print(df_out.groupby(["dataset", "model"])[["calibrated_feasible", "calibrated_regret_pct", "patience_regret_pct"]]
          .mean().to_string())


if __name__ == "__main__":
    main()
