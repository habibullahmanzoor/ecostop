"""Small comparison figure for the paper's EcoStop-Meta subsection (§6.6):
Oracle / EcoStop-Online / Patience (fair) / EcoStop-Meta, mean % energy
saved by domain, on the full 101-job corpus. Same conventions as the other
figures in this project: colorblind-safe fixed categorical colors
(Okabe-Ito), single axis, no rainbow.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from echostop.analyze_pilot import discover_runs
from echostop.calibration import precompute_score_trace, simulate_from_trace
from echostop.gating_tests import patience_stop, meta_fraction_stop, eps_tolerance_oracle
from echostop.run_calibration_demo2 import discover_tabular_runs

GAMMA_HAT = 0.54
P_HAT = 10
F_HAT = 0.927  # certified at rho=0.20 fixed-n (anytime-valid: rho=0.23, see paper section 6.3.2), n=130 calib corpus
EPS = 0.5

METHOD_COLOR = {
    "oracle": "#000000",
    "online": "#D55E00",
    "patience": "#0072B2",
    "meta": "#999999",
}
METHOD_LABEL = {
    "oracle": "Oracle (ceiling)",
    "online": "EcoStop (Online)",
    "patience": "Patience (fair)",
    "meta": "EcoStop-Meta",
}
DOMAIN_ORDER = ["NLP\n(fine-tuning)", "Tabular", "Vision\n(cosine)", "Vision\n(step-decay)"]


def collect() -> pd.DataFrame:
    records = []

    def eval_one(df, domain):
        q_max = df["val_acc"].max()
        full_e = df["cumulative_energy_joules"].iloc[-1]
        _, e_oracle = eps_tolerance_oracle(df, EPS)
        trace = precompute_score_trace(df, EPS)
        t_online = simulate_from_trace(trace, GAMMA_HAT, len(df))
        e_online = df["cumulative_energy_joules"].iloc[t_online]
        _, e_pat = patience_stop(df, P_HAT)
        _, e_meta = meta_fraction_stop(df, F_HAT)
        records.append({
            "domain": domain,
            "oracle": 100 * (1 - e_oracle / full_e),
            "online": 100 * (1 - e_online / full_e),
            "patience": 100 * (1 - e_pat / full_e),
            "meta": 100 * (1 - e_meta / full_e),
        })

    runs = discover_runs()
    for (dataset, model), dfs in runs.items():
        for df in dfs:
            eval_one(df, "Vision\n(cosine)")

    for seed in [0, 1, 2]:
        df = pd.read_csv(f"logs/step_decay_test/stepdecay_cifar10_resnet9_seed{seed}.csv")
        df["B_t"] = df["val_acc"].cummax()
        eval_one(df, "Vision\n(step-decay)")

    for job, dfs in discover_tabular_runs().items():
        for df in dfs:
            eval_one(df, "Tabular")

    for name in ["sst2", "rotten_tomatoes", "trec", "emotion", "tweet_eval_sentiment", "poem_sentiment", "banking77", "cola", "mrpc", "rte", "qnli", "wnli", "tweet_eval_irony", "tweet_eval_hate", "tweet_eval_offensive", "tweet_eval_emoji", "tweet_eval_stance_abortion", "tweet_eval_stance_atheism", "tweet_eval_stance_climate", "tweet_eval_stance_feminist", "tweet_eval_stance_hillary", "imdb", "ag_news", "yelp_polarity", "amazon_polarity", "clinc_oos", "sms_spam"]:
        for seed in [0, 1, 2]:
            df = pd.read_csv(f"logs/nlp_pilot/{name}_distilbert_seed{seed}.csv")
            df["B_t"] = df["val_acc"].cummax()
            eval_one(df, "NLP\n(fine-tuning)")

    return pd.DataFrame(records)


def main() -> None:
    df = collect()
    means = df.groupby("domain")[["oracle", "online", "patience", "meta"]].mean().reindex(DOMAIN_ORDER)

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    x = np.arange(len(DOMAIN_ORDER))
    width = 0.2
    methods = ["oracle", "online", "patience", "meta"]
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, means[method], width=width, color=METHOD_COLOR[method],
               label=METHOD_LABEL[method], zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_ORDER, fontsize=9.5)
    ax.set_ylabel("Mean energy saved vs.\nfull training (%)", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_title("Mean energy saved by method and domain", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")
    fig.tight_layout()
    out = "figures/figure7.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out}")
    print(means.round(1).to_string())


if __name__ == "__main__":
    main()
