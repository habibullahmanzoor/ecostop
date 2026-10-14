"""Recompute the domain-alone UCB table (§6.3.1, Figure 4) on the expanded
vision (n=23, +eurosat/emnist_byclass/imagenette) and NLP (n=27,
+yelp_polarity/amazon_polarity/clinc_oos/sms_spam) corpora, using the exact
job-level LTT procedure from run_calibration_demo2.py (majority-vote per job
across seeds, then a Hoeffding UCB across jobs) -- the same method that
produced every other domain-alone number reported in the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from echostop.analyze_pilot import discover_runs
from echostop.calibration import precompute_score_trace
from echostop.run_calibration_demo2 import (
    GAMMA_GRID, BETA, EPS_FOR_DEMO, ltt_calibrate_job_level,
)

NLP_JOB_NAMES = [
    "sst2", "rotten_tomatoes", "trec", "emotion", "tweet_eval_sentiment",
    "poem_sentiment", "banking77", "cola", "mrpc", "rte", "qnli", "wnli",
    "tweet_eval_irony", "tweet_eval_hate", "tweet_eval_offensive", "tweet_eval_emoji",
    "tweet_eval_stance_abortion", "tweet_eval_stance_atheism", "tweet_eval_stance_climate",
    "tweet_eval_stance_feminist", "tweet_eval_stance_hillary", "imdb", "ag_news",
    "yelp_polarity", "amazon_polarity", "clinc_oos", "sms_spam",
]

RHO = 0.30


def build_traces(jobs: dict[str, list[pd.DataFrame]]) -> dict[str, list[tuple]]:
    return {name: [(df, precompute_score_trace(df, EPS_FOR_DEMO)) for df in dfs]
            for name, dfs in jobs.items()}


def vision_jobs() -> dict[str, list[pd.DataFrame]]:
    jobs = {f"{d}_{m}": dfs for (d, m), dfs in discover_runs().items()
            if d not in ("stl10", "food101")}
    # stl10/food101 were dropped mid-run this round (killed after a CUDA OOM
    # forced a choice between finishing them slowly or freeing the GPU for
    # the rest of the corpus) -- any leftover single-seed CSV from that
    # attempt is real data but not a complete 3-seed job under this
    # project's stated recipe (Sec 5), so excluded from this round's count
    # rather than silently counted as an under-seeded job.
    step_decay_dfs = []
    for seed in [0, 1, 2]:
        df = pd.read_csv(f"logs/step_decay_test/stepdecay_cifar10_resnet9_seed{seed}.csv")
        df["B_t"] = df["val_acc"].cummax()
        step_decay_dfs.append(df)
    jobs["cifar10_resnet9_stepdecay"] = step_decay_dfs
    return jobs


def nlp_jobs() -> dict[str, list[pd.DataFrame]]:
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


def report(label: str, jobs: dict[str, list[pd.DataFrame]]) -> None:
    n = len(jobs)
    traces = build_traces(jobs)
    result = ltt_calibrate_job_level(traces, GAMMA_GRID, RHO, BETA)
    best_ucb = min(result["ucbs"])
    n_certified = len(result["certified"])
    non_degenerate = n_certified > 1  # more than just the safest grid edge
    print(f"{label}: n={n}")
    print(f"  best achievable UCB (at rho={RHO}) = {best_ucb:.3f}")
    print(f"  certified gammas at rho={RHO}: {n_certified}/{len(GAMMA_GRID)}"
          f" (gamma_hat={result['gamma_hat']})")
    print(f"  certifiable non-degenerately? {'Yes' if non_degenerate else 'No'}")
    print()


def main() -> None:
    report("Vision (cosine + step-decay)", vision_jobs())
    report("NLP (fine-tuning)", nlp_jobs())


if __name__ == "__main__":
    main()
