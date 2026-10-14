"""RQ0 mechanism figure (§6.1 companion to Figure 2): Figure 2 shows the
*consequence* of variable per-epoch cost -- energy and epoch/wall-clock
oracles disagreeing more under progressive resizing. It never shows the
*cause*. This plots the actual measured per-epoch energy profile for the
progressive-resizing CIFAR-10 runs, with the real resolution-change phase
boundaries marked, next to the accuracy profile at those same boundaries --
real data grounding the premise the RQ0 finding depends on, not an assumed
mechanism asserted in prose.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROG_DIR = Path("logs/pilot_progressive")
PHASE_BOUNDARIES = [10, 20, 30]  # epochs at which resolution steps up: 16->20->24->32px


def load_progressive_runs() -> list[pd.DataFrame]:
    dfs = []
    for csv_path in sorted(PROG_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if len(df) < 40:
            continue
        df["B_t"] = df["val_acc"].cummax()
        dfs.append(df)
    return dfs


def main() -> None:
    dfs = load_progressive_runs()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    for i, df in enumerate(dfs):
        axes[0].plot(df["epoch"], df["energy_joules"], color="#0072B2", linewidth=1.8,
                     alpha=0.75, label="individual seed runs (n=3)" if i == 0 else None)
        axes[1].plot(df["epoch"], df["B_t"], color="#0072B2", linewidth=1.8,
                     alpha=0.75, label="individual seed runs (n=3)" if i == 0 else None)

    for ax in axes:
        for j, boundary in enumerate(PHASE_BOUNDARIES):
            ax.axvline(boundary, color="#999999", linestyle="--", linewidth=1.2, zorder=1,
                       label="resolution step" if j == 0 else None)
        ax.set_xlabel("Epoch", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#eeeeee", linewidth=0.6, zorder=0)
        ax.legend(frameon=False, fontsize=13, loc="best")

    axes[0].set_ylabel("Per-epoch energy (J)", fontsize=16)
    axes[0].set_title("Energy profile\n(dashed = resolution step: 16→20→24→32px)", fontsize=17)

    axes[1].set_ylabel("Best-checkpoint validation accuracy (%)", fontsize=16)
    axes[1].set_title("Accuracy profile\n(same resolution-step boundaries)", fontsize=17)

    fig.suptitle("CIFAR-10 progressive resizing (n=3 seeds): energy and accuracy by resolution step",
                 fontsize=19, y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    out = "figures/figure2.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
