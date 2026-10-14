"""B12 (plan §19): LC-PFN-style stopping rule (Adriaensen et al., NeurIPS
2023) -- stop once P(future value beats best-so-far) < 5%, no tolerance
margin, no calibration (plan §4.0, verified against the AutoML.org writeup).

We don't have LC-PFN's actual pretrained network (trained on 10M synthetic
curves), so this substitutes our own online curve-fit predictor (§9) for
theirs and replicates their DECISION RULE specifically: eps=0 (strictly
beat the incumbent, not "within eps"), fixed threshold=0.05, no calibration.

Honesty note kept deliberately visible: the plateau-gate safeguard is KEPT
here even though LC-PFN's literal description has no such gate -- it exists
to correct a known degeneracy in OUR substitute predictor (§15, found and
fixed earlier this session), not because it's part of LC-PFN's actual
method. Removing it would make this baseline artificially weak by exposing
our predictor's specific weakness rather than testing LC-PFN's rule. This
comparison isolates the DECISION RULE (zero-margin, fixed 5%, uncalibrated)
holding the predictor mechanism fixed -- it is explicitly not a faithful
reimplementation of LC-PFN's actual (pretrained) predictor.
"""

from __future__ import annotations

import pandas as pd

from echostop.calibration import precompute_score_trace, simulate_from_trace
from echostop.gating_tests import eps_tolerance_oracle

LCPFN_THRESHOLD = 0.05  # verified exact value, AutoML.org writeup


def lcpfn_style_stop(df: pd.DataFrame, min_epochs: int | None = None, patience_k: int = 2,
                      plateau_gate: int | None = None) -> int:
    trace = precompute_score_trace(df, eps=0.0, min_epochs=min_epochs)
    return simulate_from_trace(trace, LCPFN_THRESHOLD, len(df), patience_k, plateau_gate)


def evaluate_lcpfn_style(runs: dict, eps_for_evaluation: float = 0.5) -> pd.DataFrame:
    """Run the LC-PFN-style rule, then score the resulting stopping epoch
    against OUR eps tolerance for apples-to-apples comparison with the rest
    of the method-comparison table."""
    records = []
    for (dataset, model), dfs in runs.items():
        for i, df in enumerate(dfs):
            q_max = df["val_acc"].max()
            _, e_oracle = eps_tolerance_oracle(df, eps_for_evaluation)

            t_stop = lcpfn_style_stop(df)
            e_stop = float(df["cumulative_energy_joules"].iloc[t_stop])
            feasible = df["B_t"].iloc[t_stop] >= q_max - eps_for_evaluation

            records.append({
                "dataset": dataset, "model": model, "seed_idx": i,
                "method": "lcpfn_style", "stop_epoch": t_stop,
                "energy_j": e_stop, "oracle_energy_j": e_oracle,
                "feasible": feasible,
                "regret_pct": 100 * (e_stop - e_oracle) / e_oracle,
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    from echostop.analyze_pilot import discover_runs

    runs = discover_runs()
    df_out = evaluate_lcpfn_style(runs)
    df_out.to_csv("figures/baseline_lcpfn_style_vision.csv", index=False)

    print(f"LC-PFN-style rule (threshold={LCPFN_THRESHOLD}, eps=0 internally, "
          f"evaluated at eps=0.5 tolerance)\n")
    print(f"Feasibility: {df_out['feasible'].mean()*100:.0f}% "
          f"({df_out['feasible'].sum()}/{len(df_out)} runs)")
    print(f"\nMean regret vs oracle (all runs): {df_out['regret_pct'].mean():.1f}%")
    feas = df_out[df_out.feasible]
    print(f"Mean regret vs oracle (feasible only, n={len(feas)}): "
          f"{feas['regret_pct'].mean():.1f}%" if len(feas) else "  (no feasible runs)")
    print(f"\nPer-cell breakdown:")
    print(df_out.groupby(["dataset", "model"])[["feasible", "stop_epoch", "regret_pct"]].mean().to_string())
