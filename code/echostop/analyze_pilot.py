"""First-look analysis of the pilot study (plan §12, §33 Figure 1/2).

Figure 1: best-checkpoint performance B_t vs cumulative energy.
Figure 2: Tail Energy Ratio TER_eps for eps in {0.1, 0.2, 0.5, 1.0} pp
          (plan §11.1 -- checkpoint-aware definition, not the discarded
          fraction-of-final-performance version).

Uses the Okabe-Ito colorblind-safe palette (two hues here: model identity),
fixed line style for dataset identity -- avoids a rainbow/cycled palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PILOT_DIR = Path("logs/pilot")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Okabe-Ito colorblind-safe pair, assigned by model identity (fixed order)
MODEL_COLOR = {"resnet9": "#0072B2", "resnet18": "#E69F00"}
DATASET_STYLE = {"fashion_mnist": "-", "cifar10": "-"}
EPS_LADDER = [0.1, 0.2, 0.5, 1.0]  # percentage points, plan §11.1


def load_run(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["B_t"] = df["val_acc"].cummax()  # running-best checkpoint, plan §6.2
    return df


KNOWN_MODELS = ["resnet9", "resnet18"]  # checked longest-suffix-first isn't needed: no model name is a suffix of another here


def discover_runs() -> dict[tuple[str, str], list[pd.DataFrame]]:
    """Parses "{dataset}_{model}_seed{N}.csv" by matching a KNOWN model
    suffix, not a hardcoded dataset-name allowlist -- the previous version
    silently dropped any dataset other than fashion_mnist/cifar10 (the same
    class of bug found and fixed once already this session for a dataset
    literally named "seeds" colliding with a naive "_seed" string split;
    dataset names containing underscores, e.g. "emnist_letters", have the
    identical failure mode here)."""
    runs: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for csv_path in sorted(PILOT_DIR.glob("*.csv")):
        stem = csv_path.stem  # e.g. fashion_mnist_resnet9_seed0
        parts = stem.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        dataset_model, seed = parts[0], parts[1]
        model = next((m for m in KNOWN_MODELS if dataset_model.endswith(f"_{m}")), None)
        if model is None:
            print(f"skipping unrecognized run (no known model suffix): {csv_path.name}")
            continue
        dataset = dataset_model[: -(len(model) + 1)]
        df = load_run(csv_path)
        n_epochs = len(df)
        if n_epochs < 40:
            print(f"skipping incomplete run: {csv_path.name} ({n_epochs}/40 epochs)")
            continue
        runs.setdefault((dataset, model), []).append(df)
    return runs


def _interpolate_mean_band(dfs: list[pd.DataFrame], n_grid: int = 200):
    """Interpolate each seed's B_t(cumulative_energy) onto a common energy
    grid (seeds don't share an energy x-axis, since per-epoch energy varies
    run to run), then return grid, mean, std across seeds."""
    max_e = min(df["cumulative_energy_joules"].iloc[-1] for df in dfs)
    grid = np.linspace(0, max_e, n_grid)
    interped = np.stack(
        [
            np.interp(grid, df["cumulative_energy_joules"], df["B_t"])
            for df in dfs
        ]
    )
    return grid, interped.mean(axis=0), interped.std(axis=0)


def figure1_energy_vs_performance(runs: dict) -> None:
    datasets = ["fashion_mnist", "cifar10"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    for ax, dataset in zip(axes, datasets):
        for model in ["resnet9", "resnet18"]:
            key = (dataset, model)
            if key not in runs:
                continue
            n_seeds = len(runs[key])
            grid, mean, std = _interpolate_mean_band(runs[key])
            grid_kj = grid / 1000.0
            ax.plot(
                grid_kj, mean,
                color=MODEL_COLOR[model], linewidth=1.8,
                label=f"{model} (n={n_seeds})",
            )
            ax.fill_between(
                grid_kj, mean - std, mean + std,
                color=MODEL_COLOR[model], alpha=0.18, linewidth=0,
            )
        ax.set_title(dataset.replace("_", "-"), fontsize=11)
        ax.set_xlabel("Cumulative energy (kJ)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)

    axes[0].set_ylabel("Best-checkpoint validation accuracy $B_t$ (%)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9, title="mean ± 1 std across seeds")
    fig.suptitle(
        "Figure 1 -- Best-checkpoint performance vs cumulative energy (plan §12/§33)",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIG_DIR / "retired_pilot_energy_vs_performance.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def compute_ter(df: pd.DataFrame, eps: float) -> float:
    """Checkpoint-aware TER_eps (plan §11.1): fraction of total energy spent
    after B_t first came within eps of the run's best checkpoint Q_max."""
    q_max = df["val_acc"].max()
    e_total = df["cumulative_energy_joules"].iloc[-1]
    reached = df.index[df["B_t"] >= q_max - eps]
    if len(reached) == 0:
        return float("nan")
    t_eps_idx = reached[0]
    e_at_t_eps = df["cumulative_energy_joules"].iloc[t_eps_idx]
    return (e_total - e_at_t_eps) / e_total


def figure2_tail_energy_ratio(runs: dict) -> pd.DataFrame:
    records = []
    for (dataset, model), dfs in runs.items():
        for eps in EPS_LADDER:
            ters = [compute_ter(df, eps) for df in dfs]
            records.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "eps": eps,
                    "ter_mean": np.nanmean(ters),
                    "ter_std": np.nanstd(ters),
                    "n_seeds": len(dfs),
                }
            )
    ter_df = pd.DataFrame(records)

    datasets = ["fashion_mnist", "cifar10"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    width = 0.35
    x = np.arange(len(EPS_LADDER))
    for ax, dataset in zip(axes, datasets):
        for j, model in enumerate(["resnet9", "resnet18"]):
            sub = ter_df[(ter_df.dataset == dataset) & (ter_df.model == model)]
            sub = sub.set_index("eps").reindex(EPS_LADDER)
            offset = (j - 0.5) * width
            ax.bar(
                x + offset,
                sub["ter_mean"] * 100,
                width=width,
                yerr=sub["ter_std"] * 100,
                color=MODEL_COLOR[model],
                label=model,
                capsize=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e:g}pp" for e in EPS_LADDER])
        ax.set_title(dataset.replace("_", "-"), fontsize=11)
        ax.set_xlabel("Tolerance ε")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)

    axes[0].set_ylabel("TER$_\\varepsilon$ (%) -- energy spent after reaching ε of best")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Figure 2 -- Tail Energy Ratio, T=40 epochs pre-specified (plan §11.1/§33)",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIG_DIR / "retired_pilot_tail_energy_ratio.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return ter_df


def main() -> None:
    runs = discover_runs()
    print("Complete (40-epoch) runs found:")
    for (dataset, model), dfs in runs.items():
        print(f"  {dataset:14s} {model:9s} n_seeds={len(dfs)}")

    figure1_energy_vs_performance(runs)
    ter_df = figure2_tail_energy_ratio(runs)

    print("\nTER_eps summary (T=40 epochs, pre-specified per plan §11.1):")
    print(ter_df.to_string(index=False))
    ter_df.to_csv(FIG_DIR / "ter_summary.csv", index=False)


if __name__ == "__main__":
    main()
