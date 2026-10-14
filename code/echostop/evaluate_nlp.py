"""Deploy the calibrated gamma_hat on the NLP leg (SST-2, rotten_tomatoes,
TREC + DistilBERT, plan section 16) -- these datasets are stratified into
the calibration/evaluation split like every other domain
(run_calibration_final_split.py), so which ones are "held out" here depends
on that split; this script just reports all three for a complete picture.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from echostop.calibration import precompute_score_trace, simulate_from_trace
from echostop.gating_tests import patience_stop, PATIENCE_GRID, eps_tolerance_oracle
from echostop.baseline_lcpfn_style import lcpfn_style_stop

GAMMA_HAT = 0.54  # certified at rho=0.20, beta=0.2, n_calib=130, non-degenerate threshold (anytime-valid: rho=0.23)
EPS = 0.5
NLP_DATASETS = ["sst2", "rotten_tomatoes", "trec", "emotion", "tweet_eval_sentiment", "poem_sentiment", "banking77", "cola", "mrpc", "rte", "qnli", "wnli", "tweet_eval_irony", "tweet_eval_hate", "tweet_eval_offensive", "tweet_eval_emoji", "tweet_eval_stance_abortion", "tweet_eval_stance_atheism", "tweet_eval_stance_climate", "tweet_eval_stance_feminist", "tweet_eval_stance_hillary", "imdb", "ag_news", "yelp_polarity", "amazon_polarity", "clinc_oos", "sms_spam"]


def main() -> None:
    records = []
    for name in NLP_DATASETS:
        for seed in [0, 1, 2]:
            path = Path(f"logs/nlp_pilot/{name}_distilbert_seed{seed}.csv")
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

            t_lcpfn = lcpfn_style_stop(df)
            e_lcpfn = float(df["cumulative_energy_joules"].iloc[t_lcpfn])
            feas_lcpfn = bool(df["B_t"].iloc[t_lcpfn] >= q_max - EPS)

            records.append({
                "dataset": name, "seed": seed, "oracle_epoch": t_oracle, "oracle_energy_j": e_oracle,
                "calibrated_epoch": t_cal, "calibrated_energy_j": e_cal,
                "calibrated_feasible": feas_cal,
                "calibrated_energy_saved_pct": 100 * (1 - e_cal / full_e),
                "patience_epoch": t_patience, "patience_energy_j": e_patience,
                "patience_energy_saved_pct": 100 * (1 - e_patience / full_e),
                "lcpfn_epoch": t_lcpfn, "lcpfn_energy_j": e_lcpfn,
                "lcpfn_feasible": feas_lcpfn,
                "lcpfn_energy_saved_pct": 100 * (1 - e_lcpfn / full_e),
                "full_energy_j": full_e,
            })

    df_out = pd.DataFrame(records)
    df_out.to_csv("figures/nlp_evaluation.csv", index=False)
    print(df_out.groupby("dataset")[[
        "calibrated_energy_saved_pct", "calibrated_feasible",
        "patience_energy_saved_pct", "lcpfn_energy_saved_pct", "lcpfn_feasible",
    ]].mean().to_string())
    print(f"\nMean energy saved: oracle {100*(1-df_out.oracle_energy_j/df_out.full_energy_j).mean():.1f}% | "
          f"calibrated {df_out.calibrated_energy_saved_pct.mean():.1f}% | "
          f"patience {df_out.patience_energy_saved_pct.mean():.1f}% | "
          f"lcpfn {df_out.lcpfn_energy_saved_pct.mean():.1f}%")
    print(f"Feasibility: calibrated {df_out.calibrated_feasible.mean()*100:.0f}% | "
          f"lcpfn {df_out.lcpfn_feasible.mean()*100:.0f}%")


if __name__ == "__main__":
    main()
