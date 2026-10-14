"""Two figures from the calibration-corpus phase (plan §33):
- Method comparison: regret vs oracle, by method and eps.
- Calibration: n (calibration jobs) vs achievable UCB, plus the
  leave-one-out realized-rate distribution at the certified target.
Colorblind-safe fixed categorical colors (Okabe-Ito), single axis, no
rainbow -- same principles as the pilot-phase figures.
"""

from __future__ import annotations

import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from echostop.analyze_pilot import discover_runs
from echostop.calibration import (
    precompute_score_trace, simulate_from_trace, audited_loss, hoeffding_ucb,
)

FIG_DIR = "figures"

METHOD_COLOR = {
    "oracle": "#000000",
    "patience": "#0072B2",
    "gain_per_joule": "#E69F00",
    "predictive_point": "#009E73",
    "ecostop_uncalibrated": "#D55E00",
}
METHOD_ORDER = ["oracle", "patience", "gain_per_joule", "predictive_point", "ecostop_uncalibrated"]
METHOD_LABEL = {
    "oracle": "oracle", "patience": "patience", "gain_per_joule": "gain/joule (B7)",
    "predictive_point": "predictive (point est.)", "ecostop_uncalibrated": "EcoStop (uncalibrated)",
}


def figure_method_comparison() -> None:
    df = pd.read_csv(f"{FIG_DIR}/method_comparison_summary.csv")
    eps_vals = sorted(df.eps.unique())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.15
    x = np.arange(len(eps_vals))
    for i, method in enumerate(METHOD_ORDER):
        sub = df[df.method == method].set_index("eps").reindex(eps_vals)
        offset = (i - (len(METHOD_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset, sub["mean"], width=width, yerr=sub["std"],
            color=METHOD_COLOR[method], label=METHOD_LABEL[method], capsize=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:g}pp" for e in eps_vals])
    ax.set_xlabel("Tolerance ε")
    ax.set_ylabel("Mean regret vs. oracle (% of oracle energy)")
    ax.set_title("Method comparison on the fixed-resolution pilot (feasible runs only, plan §21)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout()
    out = f"{FIG_DIR}/retired_pilot_method_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def discover_all_jobs() -> dict[str, list[pd.DataFrame]]:
    from pathlib import Path
    vision_jobs = {f"{d}_{m}": dfs for (d, m), dfs in discover_runs().items()}
    tabular_jobs: dict[str, list[pd.DataFrame]] = {}
    for base_dir, prefix in [("logs/calibration_tabular", "tabular_"),
                              ("logs/calibration_openml", "openml_")]:
        for csv_path in sorted(Path(base_dir).glob("*.csv")):
            stem = csv_path.stem
            # NOT stem.split("_seed")[0] -- see run_calibration_demo2.py's
            # discover_tabular_runs docstring (the "seeds" dataset name bug).
            dataset = re.sub(r"_seed\d+$", "", stem[len(prefix):] if stem.startswith(prefix) else stem)
            df = pd.read_csv(csv_path)
            if len(df) < 60:
                continue
            df["B_t"] = df["val_acc"].cummax()
            tabular_jobs.setdefault(dataset, []).append(df)
    nlp_jobs: dict[str, list[pd.DataFrame]] = {}
    for name in ["sst2", "rotten_tomatoes", "trec", "emotion", "tweet_eval_sentiment", "poem_sentiment", "banking77", "cola", "mrpc", "rte", "qnli", "wnli", "tweet_eval_irony", "tweet_eval_hate", "tweet_eval_offensive", "tweet_eval_emoji", "tweet_eval_stance_abortion", "tweet_eval_stance_atheism", "tweet_eval_stance_climate", "tweet_eval_stance_feminist", "tweet_eval_stance_hillary", "imdb", "ag_news", "yelp_polarity", "amazon_polarity", "clinc_oos", "sms_spam"]:
        dfs = []
        for csv_path in sorted(Path("logs/nlp_pilot").glob(f"{name}_distilbert_seed*.csv")):
            df = pd.read_csv(csv_path)
            df["B_t"] = df["val_acc"].cummax()
            dfs.append(df)
        if dfs:
            nlp_jobs[f"{name}_distilbert"] = dfs
    return {**vision_jobs, **tabular_jobs, **nlp_jobs}


def figure_calibration() -> None:
    """Performance note: the n-sweep below resamples job subsets 55 times
    (11 n-values x 5 repeats) with heavy overlap between draws. The first
    version called ltt_calibrate() directly on each sample, which recomputes
    precompute_score_trace (the expensive per-epoch curve-fit sweep) from
    scratch every single call -- the exact same redundant-recomputation
    pattern already found and fixed once this session in
    run_calibration_demo2.py, just never propagated here. A single pass
    over the ~200-run corpus costs ~1 minute of curve-fitting; doing that
    ~30-60x over (most draws overlap heavily, especially at large n) turned
    a ~1 minute figure into a 30+ minute one. Precompute each job's trace
    ONCE up front and reuse it across every (n, repeat) draw instead."""
    all_jobs = discover_all_jobs()
    job_names = list(all_jobs.keys())
    eps = 0.5
    gamma_grid = np.linspace(0.02, 0.98, 25)
    beta = 0.20
    rho_target = 0.23
    beta_corrected = beta / len(gamma_grid)

    print("Precomputing traces for n-sweep (once per job, reused across all draws)...")
    job_dfs_flat = {name: all_jobs[name] for name in job_names}
    job_traces_flat = {
        name: [(df, precompute_score_trace(df, eps)) for df in dfs]
        for name, dfs in job_dfs_flat.items()
    }
    print("  done")

    def best_ucb_for_sample(sample_names: list[str]) -> float:
        traces = [pair for name in sample_names for pair in job_traces_flat[name]]
        best = 1.0
        for gamma in gamma_grid:
            losses = [audited_loss(df, simulate_from_trace(trace, gamma, len(df)), eps)
                      for df, trace in traces]
            ucb = hoeffding_ucb(losses, beta_corrected)
            best = min(best, ucb)
        return best

    n_values = [4, 8, 12, 16, 20, 24, 28, 32, 40, 54, 66, 80, 95, 108, 123, 130, 142]
    n_values = [n for n in n_values if n <= len(job_names)]
    repeats = 5
    rng = random.Random(42)

    ucb_by_n = {}
    for n in n_values:
        ucbs = []
        for _ in range(repeats):
            sample_names = rng.sample(job_names, n)
            ucbs.append(best_ucb_for_sample(sample_names))
        ucb_by_n[n] = ucbs
        print(f"  n={n}: mean best UCB = {np.mean(ucbs):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.rcParams.update({"font.size": 14})

    ax = axes[0]
    means = [np.mean(ucb_by_n[n]) for n in n_values]
    stds = [np.std(ucb_by_n[n]) for n in n_values]
    ax.errorbar(n_values, means, yerr=stds, color="#0072B2", marker="o", markersize=5,
                linewidth=2, capsize=3)
    ax.axhline(rho_target, color="#D55E00", linestyle="--", linewidth=1.5,
               label=f"target ρ={rho_target}")
    ax.set_xlabel("n (calibration jobs, repeated random subsamples)", fontsize=15)
    ax.set_ylabel("Best achievable UCB (seed-pooled)", fontsize=15)
    ax.set_title("Certification threshold vs. calibration pool size", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=13)

    ax = axes[1]
    loo_df = pd.read_csv(f"{FIG_DIR}/calibration_demo2_job_level.csv")
    # Majority-vote each held-out job's seeds into one Bernoulli outcome
    # before averaging (sec:calib-formulation's independence-unit rule) --
    # NOT the raw per-job seed-fraction, which double-counts correlated
    # seeds of the same job and doesn't match the binary quantity
    # Proposition 1 actually bounds.
    per_job = (loo_df.groupby("held_out_job")["loss"].mean() > 0.5).astype(int).values
    jitter = np.random.RandomState(0).uniform(-0.08, 0.08, size=len(per_job))
    ax.scatter(np.zeros(len(per_job)) + jitter, per_job, color="#0072B2", alpha=0.7, s=30,
               zorder=3)
    ax.axhline(rho_target, color="#D55E00", linestyle="--", linewidth=1.5,
               label=f"target ρ={rho_target}")
    ax.axhline(np.mean(per_job), color="#000000", linestyle=":", linewidth=1.2,
               label=f"mean realized = {np.mean(per_job):.3f}")
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("Realized violation rate (job-level, majority vote)", fontsize=14)
    ax.set_title(f"Target vs. realized, leave-one-out (n={len(per_job)} jobs)",
                 fontsize=16)
    ax.tick_params(labelsize=14)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=13)

    fig.suptitle("Best achievable UCB vs. calibration pool size;\nleave-one-out target vs. realized violation rate", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = f"{FIG_DIR}/figure3.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    figure_method_comparison()
    figure_calibration()
