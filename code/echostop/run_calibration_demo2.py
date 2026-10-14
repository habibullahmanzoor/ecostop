"""Calibration demo, round 2: fold the tabular jobs in properly.

plan §6.6.2's independence unit is the JOB, not the seed -- combined
vision+tabular+OpenML gives n=27 distinct jobs (was n=4 vision-only),
aggregating each job's seeds into ONE outcome before counting n.

Performance note: traces (the expensive curve-refit-per-epoch part) are
precomputed ONCE per (job, seed) up front and reused across every gamma
and every leave-one-out fold. The first version of this script recomputed
them inside the gamma loop inside the per-job loop inside the fold loop --
about 500x more curve fits than necessary, which is why it ran for the
better part of an hour before being killed by accident. Fixed here.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from echostop.analyze_pilot import discover_runs
from echostop.calibration import (
    precompute_score_trace, simulate_from_trace, audited_loss, hoeffding_ucb,
)

GAMMA_GRID = np.linspace(0.02, 0.98, 25)  # widened after n=54 saturated the old 0.9 ceiling
TARGET_RHO = 0.20  # tightened again once the much-expanded (n=135) corpus saturated rho=0.25 to the grid ceiling
BETA = 0.20
EPS_FOR_DEMO = 0.5


def discover_tabular_runs() -> dict[str, list[pd.DataFrame]]:
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


def job_level_outcome(job_traces: list[tuple[pd.DataFrame, list[dict]]], gamma: float) -> int:
    """Aggregate a job's seeds into ONE outcome (plan §6.6.2): majority vote
    across seeds on whether deploying gamma on this job violates the
    tolerance. Uses precomputed traces -- no curve fitting happens here."""
    losses = []
    for df, trace in job_traces:
        t_stop = simulate_from_trace(trace, gamma, len(df))
        losses.append(audited_loss(df, t_stop, EPS_FOR_DEMO))
    return int(np.mean(losses) > 0.5)


def ltt_calibrate_job_level(calibration_job_traces: dict[str, list[tuple]],
                             gamma_grid: np.ndarray, rho: float, beta: float) -> dict:
    beta_corrected = beta / len(gamma_grid)
    certified, ucbs = [], []
    n_jobs = len(calibration_job_traces)
    for gamma in gamma_grid:
        losses = [job_level_outcome(job_traces, gamma) for job_traces in calibration_job_traces.values()]
        n = len(losses)
        p_hat = np.mean(losses)
        margin = np.sqrt(np.log(1 / beta_corrected) / (2 * n))
        ucb = min(p_hat + margin, 1.0)
        ucbs.append(ucb)
        if ucb <= rho:
            certified.append(gamma)
    return {"certified": certified, "gamma_hat": max(certified) if certified else None,
            "ucbs": ucbs, "n_jobs": n_jobs, "beta_corrected": beta_corrected}


NLP_JOB_NAMES = ["sst2", "rotten_tomatoes", "trec", "emotion", "tweet_eval_sentiment",
                 "poem_sentiment", "banking77", "cola", "mrpc", "rte", "qnli", "wnli",
                 "tweet_eval_irony", "tweet_eval_hate", "tweet_eval_offensive", "tweet_eval_emoji",
                 "tweet_eval_stance_abortion", "tweet_eval_stance_atheism", "tweet_eval_stance_climate",
                 "tweet_eval_stance_feminist", "tweet_eval_stance_hillary", "imdb", "ag_news",
                 "yelp_polarity", "amazon_polarity", "clinc_oos", "sms_spam"]


def discover_nlp_runs() -> dict[str, list[pd.DataFrame]]:
    jobs: dict[str, list[pd.DataFrame]] = {}
    for name in NLP_JOB_NAMES:
        dfs = []
        for csv_path in sorted(Path("logs/nlp_pilot").glob(f"{name}_distilbert_seed*.csv")):
            df = pd.read_csv(csv_path)
            df["B_t"] = df["val_acc"].cummax()
            dfs.append(df)
        if dfs:
            jobs[f"{name}_distilbert"] = dfs
    return jobs


def main() -> None:
    vision_jobs = {f"{d}_{m}": dfs for (d, m), dfs in discover_runs().items()}
    tabular_jobs = discover_tabular_runs()
    nlp_jobs = discover_nlp_runs()
    all_jobs = {**vision_jobs, **tabular_jobs, **nlp_jobs}

    print(f"Distinct jobs available: {len(all_jobs)} -> {list(all_jobs.keys())}\n")

    beta_corrected = BETA / len(GAMMA_GRID)
    for n in [4, 8, 26, 27, 59]:
        margin = np.sqrt(np.log(1 / beta_corrected) / (2 * n))
        print(f"  best-case (zero losses) UCB at n={n}: {margin:.3f}  "
              f"{'[reachable at rho=0.3]' if margin <= 0.3 else '[NOT reachable at rho=0.3]'}")
    print()

    print("Precomputing score traces (once per job-seed, reused across all gammas/folds)...")
    total_seeds = sum(len(dfs) for dfs in all_jobs.values())
    all_traces: dict[str, list[tuple]] = {}
    done = 0
    for name, dfs in all_jobs.items():
        all_traces[name] = []
        for df in dfs:
            trace = precompute_score_trace(df, EPS_FOR_DEMO)
            all_traces[name].append((df, trace))
            done += 1
    print(f"  done: {done}/{total_seeds} traces precomputed\n")

    print("=" * 70)
    print(f"Leave-one-job-out, n={len(all_jobs)} total jobs, eps={EPS_FOR_DEMO}pp, rho={TARGET_RHO}, beta={BETA}")
    print("=" * 70)
    records = []
    for held_out_name in all_jobs:
        calibration_traces = {k: v for k, v in all_traces.items() if k != held_out_name}
        result = ltt_calibrate_job_level(calibration_traces, GAMMA_GRID, rho=TARGET_RHO, beta=BETA)
        gamma_hat = result["gamma_hat"]
        print(f"held out: {held_out_name:35s} | n_calib_jobs={result['n_jobs']} | "
              f"certified={len(result['certified'])}/{len(GAMMA_GRID)} | "
              f"gamma_hat={gamma_hat} | best UCB={min(result['ucbs']):.3f}")
        if gamma_hat is None:
            continue
        for df, trace in all_traces[held_out_name]:
            t_stop = simulate_from_trace(trace, gamma_hat, len(df))
            loss = audited_loss(df, t_stop, EPS_FOR_DEMO)
            full_e = df["cumulative_energy_joules"].iloc[-1]
            saved = 100 * (1 - df["cumulative_energy_joules"].iloc[t_stop] / full_e) if full_e > 0 else 0
            records.append({"held_out_job": held_out_name, "gamma_hat": gamma_hat,
                             "loss": loss, "energy_saved_pct": saved})

    if records:
        rec_df = pd.DataFrame(records)
        rec_df.to_csv("figures/calibration_demo2_job_level.csv", index=False)
        print()
        print(rec_df.groupby("held_out_job")[["loss", "energy_saved_pct"]].mean().to_string())
        print(f"\noverall realized loss rate (n={len(rec_df)} seed-level evals across "
              f"{rec_df.held_out_job.nunique()} held-out jobs, descriptive): {rec_df['loss'].mean():.2f}")
    else:
        print(f"\nNo gamma certified in any fold at n={len(all_jobs)}.")


if __name__ == "__main__":
    main()
