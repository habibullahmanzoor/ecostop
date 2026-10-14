"""Fair patience baseline (session addition, user-prompted): the patience
numbers used everywhere else in this project (evaluate_*.py, the energy
report) are "best of PATIENCE_GRID, chosen with hindsight per run, keeping
only settings that happened to be feasible for that specific run" -- an
oracle-tuned envelope (gating_tests.py's test_a explicitly documents this),
not a deployable rule. No real system gets to pick its patience value after
seeing whether it worked.

This script holds patience to the SAME standard calibrated EcoStop is held
to: ONE fixed patience value, selected once via the same LTT procedure
(Hoeffding UCB + Bonferroni correction across the candidate grid) on the
SAME calibration pool, deployed unchanged on the SAME held-out evaluation
pool as run_calibration_final_split.py's gamma_hat. That is the fair
comparison: one pre-registered rule vs. another, not a rule vs. an
oracle-tuned envelope of itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from echostop.analyze_pilot import discover_runs
from echostop.calibration import audited_loss, hoeffding_ucb
from echostop.gating_tests import patience_stop, PATIENCE_GRID
from echostop.run_calibration_demo2 import discover_tabular_runs
from echostop.run_calibration_final_split import discover_nlp_runs, stratified_split

BETA = 0.20
EPS_FOR_DEMO = 0.5
SPLIT_SEED = 0
N_EVAL = 12


def job_level_patience_outcome(dfs: list[pd.DataFrame], patience_k: int, eps: float) -> int:
    """Majority vote across a job's seeds (plan §6.6.2 job-not-seed unit),
    matching run_calibration_demo2.job_level_outcome's convention exactly."""
    losses = []
    for df in dfs:
        t_stop, _ = patience_stop(df, patience_k)
        losses.append(audited_loss(df, t_stop, eps))
    return int(np.mean(losses) > 0.5)


def calibrate_patience(calib_jobs: dict, rho: float, beta: float) -> dict:
    beta_corrected = beta / len(PATIENCE_GRID)
    certified, ucbs = [], []
    for p in PATIENCE_GRID:
        losses = [job_level_patience_outcome(dfs, p, EPS_FOR_DEMO) for dfs in calib_jobs.values()]
        ucb = hoeffding_ucb(losses, beta_corrected)
        ucbs.append(ucb)
        if ucb <= rho:
            certified.append(p)
    # smaller patience = fires sooner = more aggressive -> pick the SMALLEST certified
    p_hat = min(certified) if certified else None
    return {"p_hat": p_hat, "certified": certified, "ucbs": ucbs, "beta_corrected": beta_corrected}


def main() -> None:
    vision_jobs = {f"{d}_{m}": dfs for (d, m), dfs in discover_runs().items()}
    tabular_jobs = discover_tabular_runs()
    nlp_jobs = discover_nlp_runs()
    all_jobs = {**vision_jobs, **tabular_jobs, **nlp_jobs}

    calib_names, eval_names = stratified_split(all_jobs, SPLIT_SEED, N_EVAL)
    calib_jobs = {k: all_jobs[k] for k in calib_names}
    print(f"Calibration pool (n={len(calib_names)}), evaluation pool (n={len(eval_names)}): "
          f"{eval_names}\n")

    print("=" * 70)
    print(f"Calibrating a SINGLE fixed patience value, sweeping rho, "
          f"grid={PATIENCE_GRID}, beta={BETA}")
    print("=" * 70)
    for rho in [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
        result = calibrate_patience(calib_jobs, rho, BETA)
        print(f"rho={rho:.2f} | certified={result['certified']} | p_hat={result['p_hat']} | "
              f"best UCB={min(result['ucbs']):.3f}")

    TARGET_RHO = 0.20  # match gamma_hat's target -- see run_calibration_final_split.py
    final = calibrate_patience(calib_jobs, rho=TARGET_RHO, beta=BETA)
    p_hat = final["p_hat"]
    print(f"\nFINAL: p_hat={p_hat} (certified at rho={TARGET_RHO}, beta={BETA}, "
          f"n_calibration_jobs={len(calib_names)})\n")

    if p_hat is None:
        print("No patience value certified -- cannot proceed to evaluation.")
        return

    records = []
    for name in eval_names:
        for df in all_jobs[name]:
            t_stop, _ = patience_stop(df, p_hat)
            loss = audited_loss(df, t_stop, EPS_FOR_DEMO)
            full_e = df["cumulative_energy_joules"].iloc[-1]
            e_stop = df["cumulative_energy_joules"].iloc[t_stop]
            saved = 100 * (1 - e_stop / full_e) if full_e > 0 else 0
            records.append({"job": name, "loss": loss, "energy_saved_pct": saved})

    rec_df = pd.DataFrame(records)
    rec_df.to_csv("figures/calibration_patience_eval_results.csv", index=False)
    print(rec_df.groupby("job")[["loss", "energy_saved_pct"]].mean().to_string())

    print(f"\nFixed-patience(p={p_hat}) on evaluation pool: "
          f"realized violation rate={rec_df['loss'].mean():.3f} "
          f"(n={len(rec_df)} seed-evals across {len(eval_names)} jobs), "
          f"mean energy saved={rec_df['energy_saved_pct'].mean():.1f}%")


if __name__ == "__main__":
    main()
