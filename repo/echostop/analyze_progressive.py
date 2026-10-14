"""Deciding RQ0 test (plan §16 REQUIRED, §40.1 Test B) on the progressive-
resizing CIFAR-10 data -- does energy diverge from epoch/wall-clock time
once it's forced to vary independently of epoch count?
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from echostop.gating_tests import per_epoch_diagnostics, cost_axis_oracle_epochs
import numpy as np

PROG_DIR = Path("logs/pilot_progressive")
FIG_DIR = Path("figures")


def load_progressive_runs() -> list[pd.DataFrame]:
    dfs = []
    for csv_path in sorted(PROG_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if len(df) < 40:
            print(f"skipping incomplete: {csv_path.name}")
            continue
        df["B_t"] = df["val_acc"].cummax()
        dfs.append(df)
    return dfs


def figure_energy_profile(dfs: list[pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, df in enumerate(dfs):
        axes[0].plot(df["epoch"], df["energy_joules"], color="#0072B2", alpha=0.7,
                     label="per-epoch energy" if i == 0 else None)
        axes[1].plot(df["epoch"], df["B_t"], color="#0072B2", alpha=0.7,
                     label="$B_t$" if i == 0 else None)

    # mark phase boundaries
    for ax in axes:
        for boundary in [10, 20, 30]:
            ax.axvline(boundary, color="#999999", linestyle="--", linewidth=1)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Per-epoch energy (J)")
    axes[0].set_title("Energy profile (dashed = resolution change: 16→20→24→32px)")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Best-checkpoint accuracy $B_t$ (%)")
    axes[1].set_title("Accuracy profile (same phase boundaries)")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Progressive-resizing CIFAR-10 + ResNet-9 (n=3 seeds)", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "figure_progressive_energy_profile.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    dfs = load_progressive_runs()
    print(f"loaded {len(dfs)} complete progressive-resizing runs\n")

    figure_energy_profile(dfs)

    print("Per-epoch energy/runtime diagnostics (progressive resizing):")
    diags = [per_epoch_diagnostics(df) for df in dfs]
    diag_df = pd.DataFrame(diags)
    print(diag_df.to_string(index=False))
    print(f"\nmean cv_energy={diag_df.cv_energy.mean():.3f}  "
          f"mean cv_runtime={diag_df.cv_runtime.mean():.3f}  "
          f"mean r2(energy~runtime)={diag_df.r2_energy_vs_runtime.mean():.3f}")

    print("\nCost-axis oracle selected-epoch agreement (progressive resizing):")
    lambdas = np.linspace(0.01, 5.0, 40)
    records = []
    for i, df in enumerate(dfs):
        for lam in lambdas:
            t_e, t_ep, t_ti = cost_axis_oracle_epochs(df, lam)
            records.append({
                "seed_idx": i, "lambda": lam,
                "t_energy": t_e, "t_epoch": t_ep, "t_time": t_ti,
                "abs_diff_energy_epoch": abs(t_e - t_ep),
                "abs_diff_energy_time": abs(t_e - t_ti),
            })
    oracle_df = pd.DataFrame(records)
    print(oracle_df[["abs_diff_energy_epoch", "abs_diff_energy_time"]].mean().to_string())
    oracle_df.to_csv(FIG_DIR / "test_b_progressive_oracle_epochs.csv", index=False)

    # distribution of where they diverge, not just the mean
    print("\nDistribution of |t_energy - t_epoch| (progressive resizing):")
    print(oracle_df["abs_diff_energy_epoch"].describe().to_string())


if __name__ == "__main__":
    main()
