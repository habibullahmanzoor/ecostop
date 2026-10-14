"""Accuracy-preservation figure for the paper: distribution of the audited
test-accuracy gap (A_full - A_method) per method, against the eps=0.5pp
tolerance the whole calibration story is built around, one point per JOB
(mean gap across that job's seeds), not one point per seed-run. Seeds of
the same job are correlated, nested observations, not independent draws
(sec:calib-formulation's independence-unit rule) -- an earlier version of
this figure and the DKW band it carries plugged n=538 (seed-runs) into a
formula that requires i.i.d. draws, which overstated precision. Aggregating
to n=143 (jobs) here matches what the calibration procedure itself already
does when it selects gamma_hat.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from echostop.analyze_pilot import discover_runs
from echostop.calibration import precompute_score_trace, simulate_from_trace, forward_fill_audit
from echostop.gating_tests import patience_stop, meta_fraction_stop, eps_tolerance_oracle
from echostop.run_calibration_demo2 import discover_tabular_runs

GAMMA_HAT = 0.54
P_HAT = 10
F_HAT = 0.927  # certified at rho=0.20 fixed-n (anytime-valid: rho=0.23), n=130 calib corpus
EPS = 0.5

METHOD_ORDER = ["oracle", "online", "patience", "meta"]
METHOD_LABEL = {"oracle": "Oracle", "online": "EcoStop", "patience": "Patience (fair)", "meta": "EcoStop-Meta"}
METHOD_COLOR = {"oracle": "#000000", "online": "#D55E00", "patience": "#0072B2", "meta": "#999999"}


def collect() -> pd.DataFrame:
    """One row per JOB: each gap column is that job's mean over its own
    seeds, matching the "mean over seeds after majority-vote aggregation"
    aggregation sec:metrics already specifies for energy saved / accuracy
    gap. Violation itself is then thresholded once per job downstream, on
    this already-aggregated mean gap."""
    per_seed = []

    def eval_one(df, domain, dataset):
        a_t = forward_fill_audit(df)
        A_full = a_t[-1]
        t_oracle, _ = eps_tolerance_oracle(df, EPS)
        trace = precompute_score_trace(df, EPS)
        t_online = simulate_from_trace(trace, GAMMA_HAT, len(df))
        t_pat, _ = patience_stop(df, P_HAT)
        t_meta, _ = meta_fraction_stop(df, F_HAT)
        per_seed.append({
            "domain": domain, "dataset": dataset,
            "gap_oracle": A_full - a_t[t_oracle],
            "gap_online": A_full - a_t[t_online],
            "gap_patience": A_full - a_t[t_pat],
            "gap_meta": A_full - a_t[t_meta],
        })

    runs = discover_runs()
    for (dataset, model), dfs in runs.items():
        for df in dfs:
            eval_one(df, "Vision (cosine)", f"{dataset}_{model}")
    for seed in [0, 1, 2]:
        df = pd.read_csv(f"logs/step_decay_test/stepdecay_cifar10_resnet9_seed{seed}.csv")
        df["B_t"] = df["val_acc"].cummax()
        eval_one(df, "Vision (step-decay)", "cifar10_resnet9_stepdecay")
    for job, dfs in discover_tabular_runs().items():
        for df in dfs:
            eval_one(df, "Tabular / OpenML", job)
    for name in ["sst2", "rotten_tomatoes", "trec", "emotion", "tweet_eval_sentiment", "poem_sentiment", "banking77", "cola", "mrpc", "rte", "qnli", "wnli", "tweet_eval_irony", "tweet_eval_hate", "tweet_eval_offensive", "tweet_eval_emoji", "tweet_eval_stance_abortion", "tweet_eval_stance_atheism", "tweet_eval_stance_climate", "tweet_eval_stance_feminist", "tweet_eval_stance_hillary", "imdb", "ag_news", "yelp_polarity", "amazon_polarity", "clinc_oos", "sms_spam"]:
        for seed in [0, 1, 2]:
            df = pd.read_csv(f"logs/nlp_pilot/{name}_distilbert_seed{seed}.csv")
            df["B_t"] = df["val_acc"].cummax()
            eval_one(df, "NLP (fine-tuning)", name)

    seed_df = pd.DataFrame(per_seed)
    gap_cols = ["gap_oracle", "gap_online", "gap_patience", "gap_meta"]
    job_df = seed_df.groupby(["domain", "dataset"], as_index=False)[gap_cols].mean()
    return job_df


def main() -> None:
    df = collect()
    n = len(df)

    # Dvoretzky-Kiefer-Wolfowitz: a distribution-free, non-asymptotic
    # simultaneous confidence band around the whole ECDF -- same Hoeffding-
    # type tail bound already used throughout the paper's calibration
    # machinery, just applied to the empirical CDF instead of a single
    # Bernoulli rate. sup_x |F_n(x) - F(x)| <= eps with prob >= 1-alpha for
    # eps = sqrt(log(2/alpha) / (2n)).
    ALPHA_CI = 0.05
    eps_dkw = np.sqrt(np.log(2 / ALPHA_CI) / (2 * n))

    fig, ax = plt.subplots(figsize=(3.6, 4.6))
    x_max = 3.0
    for m in METHOD_ORDER:
        vals = np.sort(df[f"gap_{m}"].values)
        y = np.arange(1, n + 1) / n
        # extend the last step to the right edge so the line doesn't stop short
        x_plot = np.concatenate([vals, [x_max]])
        y_plot = np.concatenate([y, [y[-1]]])
        y_lo = np.clip(y_plot - eps_dkw, 0, 1)
        y_hi = np.clip(y_plot + eps_dkw, 0, 1)
        ax.fill_between(x_plot, y_lo, y_hi, step="post", color=METHOD_COLOR[m],
                         alpha=0.15, linewidth=0, zorder=2)
        ax.step(x_plot, y_plot, where="post", color=METHOD_COLOR[m],
                linewidth=2.4, label=METHOD_LABEL[m].replace("\n", " "), zorder=3)

    ax.axvline(EPS, color="#D55E00", linestyle="--", linewidth=1.3, zorder=1,
               label=f"tolerance ε={EPS}pp")
    band_patch = plt.Rectangle((0, 0), 1, 1, color="#888888", alpha=0.25,
                                label=f"95% DKW band (±{eps_dkw:.3f})")
    ax.set_xlim(-1, x_max)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Accuracy gap vs. full training (pp)\nhigher = worse than full training", fontsize=11)
    ax.set_ylabel("Fraction of runs with gap ≤ x", fontsize=11)
    ax.set_title(f"Empirical CDF of accuracy gap vs.\nfull training, all methods (n={n})", fontsize=11.5)
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(band_patch)
    labels.append(band_patch.get_label())
    ax.legend(handles=handles, labels=labels, frameon=False, fontsize=7.8, loc="lower right")
    fig.tight_layout()
    out = "figures/figure9.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out}")

    print()
    print("Mean gap:", df[[f"gap_{m}" for m in METHOD_ORDER]].mean().round(3).to_dict())
    print("N exceeding tolerance:",
          {m: int((df[f"gap_{m}"] > EPS).sum()) for m in METHOD_ORDER}, f"/ {n}")
    print("share within tolerance:",
          {m: round(100*(df[f'gap_{m}'] <= EPS).mean(), 1) for m in METHOD_ORDER})
    print("share exactly zero:",
          {m: round(100*(df[f'gap_{m}'] == 0).mean(), 1) for m in METHOD_ORDER})
    df.to_csv("figures/accuracy_gap_job_level.csv", index=False)
    print("wrote figures/accuracy_gap_job_level.csv")


if __name__ == "__main__":
    main()
