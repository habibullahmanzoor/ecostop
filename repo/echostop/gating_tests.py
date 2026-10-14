"""Plan §40.1 -- the two tests that must run BEFORE any EcoStop code exists.

Test A: is there real headroom between well-tuned patience and the
        epsilon-tolerance oracle?
Test B (RQ0): does energy carry information beyond epoch count / wall-clock
        time, or is it just a rescaling of one of them?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from echostop.analyze_pilot import discover_runs, EPS_LADDER

PATIENCE_GRID = [3, 5, 10, 15, 20]
LAMBDA_STEPS = 40  # sweep density for the cost-axis oracle comparison (Test B)


# ---------------------------------------------------------------------------
# Test A: oracle vs. patience
# ---------------------------------------------------------------------------

def eps_tolerance_oracle(df: pd.DataFrame, eps: float) -> tuple[int, float]:
    """plan §13.3: t*_eps = min{t : B_t >= Q_max - eps}."""
    q_max = df["val_acc"].max()
    reached = df.index[df["B_t"] >= q_max - eps]
    t_star = int(reached[0])
    e_star = float(df["cumulative_energy_joules"].iloc[t_star])
    return t_star, e_star


def patience_stop(df: pd.DataFrame, patience: int) -> tuple[int, float]:
    """Stop the first epoch t such that B_t hasn't improved for `patience`
    consecutive epochs (classical validation-accuracy early stopping, B3/B4)."""
    best_so_far = -np.inf
    epochs_since_improve = 0
    for t, val_acc in enumerate(df["val_acc"]):
        if val_acc > best_so_far:
            best_so_far = val_acc
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if epochs_since_improve >= patience:
            return t, float(df["cumulative_energy_joules"].iloc[t])
    last = len(df) - 1
    return last, float(df["cumulative_energy_joules"].iloc[last])


def meta_fraction_stop(df: pd.DataFrame, fraction: float) -> tuple[int, float]:
    """EcoStop-Meta (session addition): stop at a fixed FRACTION of the
    run's own budget, where the fraction itself is meta-derived from other
    jobs' observed oracle-stop points (never the target job's own data) --
    a simple, honest stand-in for "uses historical cross-job information,"
    the direct warm-start counterpart to EcoStop-Online's cold-start-only
    rule. Candidate fractions and their calibration live in
    run_calibration_meta.py, mirroring run_calibration_patience.py exactly
    so cold-start vs. warm-start is a fair, like-for-like comparison."""
    t = max(0, min(round(fraction * (len(df) - 1)), len(df) - 1))
    return t, float(df["cumulative_energy_joules"].iloc[t])


def test_a(runs: dict) -> pd.DataFrame:
    records = []
    for (dataset, model), dfs in runs.items():
        for eps in EPS_LADDER:
            for i, df in enumerate(dfs):
                t_oracle, e_oracle = eps_tolerance_oracle(df, eps)
                q_max = df["val_acc"].max()

                # sweep patience; among FEASIBLE settings (checkpoint within
                # eps of Q_max), take the one using least energy -- this is
                # the "oracle-tuned patience envelope" (hindsight-best),
                # plan §19.1 -- an upper bound on what patience can do, not
                # a deployable baseline.
                feasible = []
                for p in PATIENCE_GRID:
                    t_p, e_p = patience_stop(df, p)
                    b_t_p = df["B_t"].iloc[t_p]
                    is_feasible = b_t_p >= q_max - eps
                    feasible.append((p, t_p, e_p, is_feasible))

                feasible_only = [f for f in feasible if f[3]]
                if feasible_only:
                    best_patience_energy = min(f[2] for f in feasible_only)
                    regret = best_patience_energy - e_oracle
                else:
                    best_patience_energy = float("nan")
                    regret = float("nan")

                records.append(
                    {
                        "dataset": dataset, "model": model, "seed_idx": i, "eps": eps,
                        "oracle_epoch": t_oracle, "oracle_energy_j": e_oracle,
                        "best_feasible_patience_energy_j": best_patience_energy,
                        "regret_j": regret,
                        "regret_pct_of_oracle": (
                            100 * regret / e_oracle if not np.isnan(regret) else np.nan
                        ),
                        "n_feasible_patience_settings": len(feasible_only),
                    }
                )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Test B (RQ0): does energy differ from epoch/time?
# ---------------------------------------------------------------------------

def per_epoch_diagnostics(df: pd.DataFrame) -> dict:
    e_t = df["energy_joules"].values
    r_t = df["duration_seconds"].values
    cv_e = np.std(e_t) / np.mean(e_t)
    cv_r = np.std(r_t) / np.mean(r_t)
    # R^2 of e_t ~ r_t (does runtime explain energy?)
    if np.std(r_t) > 0:
        corr = np.corrcoef(e_t, r_t)[0, 1]
        r2 = corr ** 2
    else:
        r2 = float("nan")
    return {"cv_energy": cv_e, "cv_runtime": cv_r, "r2_energy_vs_runtime": r2}


def cost_axis_oracle_epochs(df: pd.DataFrame, lam: float) -> tuple[int, int, int]:
    """plan §13.4: same utility oracle, three normalized cost terms."""
    e_norm = df["cumulative_energy_joules"] / df["cumulative_energy_joules"].iloc[-1]
    time_norm = df["cumulative_duration_seconds"] / df["cumulative_duration_seconds"].iloc[-1]
    epoch_norm = (df.index + 1) / len(df)

    t_energy = int((df["B_t"] - lam * e_norm).idxmax())
    t_time = int((df["B_t"] - lam * time_norm).idxmax())
    t_epoch = int((df["B_t"] - lam * epoch_norm).idxmax())
    return t_energy, t_epoch, t_time


def test_b(runs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    diag_records = []
    for (dataset, model), dfs in runs.items():
        for i, df in enumerate(dfs):
            d = per_epoch_diagnostics(df)
            d.update({"dataset": dataset, "model": model, "seed_idx": i})
            diag_records.append(d)
    diag_df = pd.DataFrame(diag_records)

    oracle_records = []
    lambdas = np.linspace(0.01, 5.0, LAMBDA_STEPS)
    for (dataset, model), dfs in runs.items():
        for i, df in enumerate(dfs):
            for lam in lambdas:
                t_e, t_ep, t_ti = cost_axis_oracle_epochs(df, lam)
                oracle_records.append(
                    {
                        "dataset": dataset, "model": model, "seed_idx": i, "lambda": lam,
                        "t_energy": t_e, "t_epoch": t_ep, "t_time": t_ti,
                        "abs_diff_energy_epoch": abs(t_e - t_ep),
                        "abs_diff_energy_time": abs(t_e - t_ti),
                    }
                )
    oracle_df = pd.DataFrame(oracle_records)
    return diag_df, oracle_df


# ---------------------------------------------------------------------------

def main() -> None:
    runs = discover_runs()

    print("=" * 70)
    print("TEST A -- oracle vs. well-tuned patience (plan §40.1)")
    print("=" * 70)
    a_df = test_a(runs)
    summary_a = (
        a_df.groupby(["dataset", "model", "eps"])
        .agg(
            mean_regret_pct=("regret_pct_of_oracle", "mean"),
            mean_oracle_energy_j=("oracle_energy_j", "mean"),
            n_feasible=("n_feasible_patience_settings", "mean"),
        )
        .reset_index()
    )
    print(summary_a.to_string(index=False))
    a_df.to_csv("figures/test_a_full.csv", index=False)
    summary_a.to_csv("figures/test_a_summary.csv", index=False)

    print()
    print("=" * 70)
    print("TEST B (RQ0) -- does energy differ from epoch/time? (plan §40.1)")
    print("=" * 70)
    diag_df, oracle_df = test_b(runs)
    print("\nPer-epoch energy/runtime diagnostics (mean over runs per cell):")
    print(
        diag_df.groupby(["dataset", "model"])[
            ["cv_energy", "cv_runtime", "r2_energy_vs_runtime"]
        ].mean().to_string()
    )

    print("\nCost-axis oracle selected-epoch agreement (mean |diff| across lambda sweep):")
    print(
        oracle_df.groupby(["dataset", "model"])[
            ["abs_diff_energy_epoch", "abs_diff_energy_time"]
        ].mean().to_string()
    )
    diag_df.to_csv("figures/test_b_diagnostics.csv", index=False)
    oracle_df.to_csv("figures/test_b_oracle_epochs.csv", index=False)


if __name__ == "__main__":
    main()
