"""RQ0 headline figure (session addition, prompted by a figure-set review:
the paper's abstract leads with the RQ0 finding -- energy only diverges
from epoch-count as a stopping signal under variable-per-epoch-cost
training -- yet no figure in the paper showed the divergence itself, only
the mechanism (figure_progressive_energy_profile.png, never cited) or raw
per-cell curves (figure1/figure2, never cited either).

Reuses gating_tests.test_b(), unchanged, on two real run sets: the
original 4-cell fixed-resolution pilot (cifar10/fashion_mnist x
resnet9/resnet18) and the progressive-resizing CIFAR-10 runs -- the exact
same oracle-agreement methodology already used to produce the numbers
quoted in §6.1 ("mean |epoch difference| 0.01-0.42" / "0.66... 0.53"),
just not previously turned into a figure. Both |t_energy - t_epoch| and
|t_energy - t_time| are heavily zero-inflated (97%/80% exact agreement) --
a box plot would be as degenerate here as it was for the accuracy-gap
figure earlier this session, so this uses the same ECDF fix.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from echostop.analyze_pilot import discover_runs
from echostop.gating_tests import test_b

REGIME_COLOR = {"fixed": "#009E73", "variable": "#CC79A7"}
REGIME_LABEL = {"fixed": "Fixed-resolution (4-cell pilot, n=480 draws)",
                "variable": "Progressive resizing (n=120 draws)"}


def load_progressive() -> pd.DataFrame:
    from pathlib import Path
    dfs = []
    for csv_path in sorted(Path("logs/pilot_progressive").glob("*.csv")):
        df = pd.read_csv(csv_path)
        if len(df) < 40:
            continue
        df["B_t"] = df["val_acc"].cummax()
        dfs.append(df)
    _, oracle_df = test_b({("progressive_cifar10", "resnet9"): dfs})
    return oracle_df


def ecdf_panel(ax, fixed_vals, var_vals, xlabel: str, title: str) -> None:
    for key, vals in [("fixed", fixed_vals), ("variable", var_vals)]:
        v = np.sort(vals)
        n = len(v)
        y = np.arange(1, n + 1) / n
        x_max = 6
        x_plot = np.concatenate([v, [x_max]])
        y_plot = np.concatenate([y, [y[-1]]])
        ax.step(x_plot, y_plot, where="post", color=REGIME_COLOR[key], linewidth=2.2,
                label=REGIME_LABEL[key], zorder=3)
    ax.set_xlim(-0.3, 6)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel("Fraction of draws with agreement ≤ x", fontsize=15)
    ax.set_title(title, fontsize=17)
    ax.tick_params(labelsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)


def main() -> None:
    runs = discover_runs()
    pilot4 = {(d, m): dfs for (d, m), dfs in runs.items()
              if d in ("cifar10", "fashion_mnist") and m in ("resnet9", "resnet18")}
    _, fixed_df = test_b(pilot4)
    fixed_df.to_csv("figures/test_a_fixed_cost_oracle_epochs.csv", index=False)

    var_df = load_progressive()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ecdf_panel(axes[0], fixed_df["abs_diff_energy_epoch"], var_df["abs_diff_energy_epoch"],
               "Stopping-epoch difference",
               "Energy vs. epoch-count oracle agreement")
    ecdf_panel(axes[1], fixed_df["abs_diff_energy_time"], var_df["abs_diff_energy_time"],
               "Stopping-epoch difference",
               "Energy vs. wall-clock oracle agreement")
    axes[0].legend(frameon=False, fontsize=13, loc="lower right")

    fig.suptitle("Stopping-epoch agreement: fixed-resolution vs. progressive-resizing training", fontsize=19, y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    out = "figures/figure1.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out}")

    print(f"\nFixed-resolution regime: mean |diff epoch| vs epoch-oracle = "
          f"{fixed_df['abs_diff_energy_epoch'].mean():.3f}, vs time-oracle = "
          f"{fixed_df['abs_diff_energy_time'].mean():.3f}")
    print(f"Variable-cost regime:    mean |diff epoch| vs epoch-oracle = "
          f"{var_df['abs_diff_energy_epoch'].mean():.3f}, vs time-oracle = "
          f"{var_df['abs_diff_energy_time'].mean():.3f}")


if __name__ == "__main__":
    main()
