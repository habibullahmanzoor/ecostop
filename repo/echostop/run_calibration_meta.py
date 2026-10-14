"""EcoStop-Meta: a warm-start counterpart to cold-start EcoStop-Online,
built to quantify the "cold-start cost" the plan (§4.1) flagged early on
and never followed up -- how much energy-saving is actually given up by
refusing to use historical/cross-job information at all?

Design (deliberately simple and honest, not a claim to reimplement a
pretrained curve-extrapolation network like LC-PFN's): stop at a fixed
FRACTION of the run's own epoch budget, where the candidate fractions are
themselves derived from the empirical distribution of oracle-stop-fractions
observed across the CALIBRATION pool only (never the evaluation pool, never
the target job) -- this is what makes it "meta" rather than an arbitrary
heuristic: the candidates come from historical cross-job behavior, not a
guess. The candidate grid is then calibrated via the identical LTT
procedure (Hoeffding UCB + Bonferroni, job-level independence unit) used
for gamma_hat and the fair patience baseline, on the identical calibration
pool, deployed on the identical held-out evaluation pool -- so the
cold-start-vs-warm-start comparison is apples-to-apples, not an unfair
fight between a calibrated and an uncalibrated rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from echostop.analyze_pilot import discover_runs
from echostop.calibration import audited_loss, hoeffding_ucb
from echostop.gating_tests import meta_fraction_stop, eps_tolerance_oracle
from echostop.run_calibration_demo2 import discover_tabular_runs
from echostop.run_calibration_final_split import discover_nlp_runs, stratified_split

BETA = 0.20
EPS_FOR_DEMO = 0.5
SPLIT_SEED = 0
N_EVAL = 12
TARGET_RHO = 0.20  # match gamma_hat's and fair-patience's target


def derive_meta_fraction_grid(calib_jobs: dict, eps: float) -> list[float]:
    """The candidates come from the calibration pool's own observed
    oracle-stop fractions -- this is the "meta" (cross-job) information
    the cold-start predictor never gets to use."""
    fractions = []
    for dfs in calib_jobs.values():
        for df in dfs:
            t_star, _ = eps_tolerance_oracle(df, eps)
            fractions.append(t_star / max(len(df) - 1, 1))
    fractions = np.array(fractions)
    grid = sorted(set(round(float(q), 3) for q in np.percentile(fractions, [10, 25, 40, 50, 60, 75, 90, 99])))
    return grid


def job_level_meta_outcome(dfs: list[pd.DataFrame], fraction: float, eps: float) -> int:
    """Majority vote across a job's seeds (plan §6.6.2 job-not-seed unit),
    matching run_calibration_patience.py's convention exactly."""
    losses = []
    for df in dfs:
        t_stop, _ = meta_fraction_stop(df, fraction)
        losses.append(audited_loss(df, t_stop, eps))
    return int(np.mean(losses) > 0.5)


def calibrate_meta(calib_jobs: dict, grid: list[float], rho: float, beta: float) -> dict:
    beta_corrected = beta / len(grid)
    certified, ucbs = [], []
    for f in grid:
        losses = [job_level_meta_outcome(dfs, f, EPS_FOR_DEMO) for dfs in calib_jobs.values()]
        ucb = hoeffding_ucb(losses, beta_corrected)
        ucbs.append(ucb)
        if ucb <= rho:
            certified.append(f)
    # smaller fraction = fires sooner = more aggressive -> pick the SMALLEST certified
    f_hat = min(certified) if certified else None
    return {"f_hat": f_hat, "certified": certified, "ucbs": ucbs, "beta_corrected": beta_corrected}


def main() -> None:
    vision_jobs = {f"{d}_{m}": dfs for (d, m), dfs in discover_runs().items()}
    tabular_jobs = discover_tabular_runs()
    nlp_jobs = discover_nlp_runs()
    all_jobs = {**vision_jobs, **tabular_jobs, **nlp_jobs}

    calib_names, eval_names = stratified_split(all_jobs, SPLIT_SEED, N_EVAL)
    calib_jobs = {k: all_jobs[k] for k in calib_names}
    print(f"Calibration pool (n={len(calib_names)}), evaluation pool (n={len(eval_names)}): "
          f"{eval_names}\n")

    grid = derive_meta_fraction_grid(calib_jobs, EPS_FOR_DEMO)
    print(f"Meta-derived candidate fractions (from calibration-pool oracle-stop quantiles): {grid}\n")

    print("=" * 70)
    print(f"Calibrating a SINGLE fixed meta-fraction, sweeping rho, beta={BETA}")
    print("=" * 70)
    for rho in [0.3, 0.25, 0.20, 0.15, 0.1, 0.05]:
        result = calibrate_meta(calib_jobs, grid, rho, BETA)
        print(f"rho={rho:.2f} | certified={result['certified']} | f_hat={result['f_hat']} | "
              f"best UCB={min(result['ucbs']):.3f}")

    final = calibrate_meta(calib_jobs, grid, rho=TARGET_RHO, beta=BETA)
    f_hat = final["f_hat"]
    print(f"\nFINAL: f_hat={f_hat} (certified at rho={TARGET_RHO}, beta={BETA}, "
          f"n_calibration_jobs={len(calib_names)})\n")

    if f_hat is None:
        print("No fraction certified -- cannot proceed to evaluation.")
        return

    records = []
    for name in eval_names:
        for df in all_jobs[name]:
            t_stop, _ = meta_fraction_stop(df, f_hat)
            loss = audited_loss(df, t_stop, EPS_FOR_DEMO)
            full_e = df["cumulative_energy_joules"].iloc[-1]
            e_stop = df["cumulative_energy_joules"].iloc[t_stop]
            saved = 100 * (1 - e_stop / full_e) if full_e > 0 else 0
            records.append({"job": name, "loss": loss, "energy_saved_pct": saved})

    rec_df = pd.DataFrame(records)
    rec_df.to_csv("figures/calibration_meta_eval_results.csv", index=False)
    print(rec_df.groupby("job")[["loss", "energy_saved_pct"]].mean().to_string())

    print(f"\nEcoStop-Meta(f={f_hat}) on evaluation pool: "
          f"realized violation rate={rec_df['loss'].mean():.3f} "
          f"(n={len(rec_df)} seed-evals across {len(eval_names)} jobs), "
          f"mean energy saved={rec_df['energy_saved_pct'].mean():.1f}%")


if __name__ == "__main__":
    main()
