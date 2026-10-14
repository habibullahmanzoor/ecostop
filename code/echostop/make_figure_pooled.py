"""Headline figure: EcoStop framework minus single-rule baselines at the same rho, pooled out-of-fold over 259 jobs.
Source: figures/tierA/c1_framework_oof_diffs.csv (echostop/c1_framework_oof.py). Two baselines per rule: standard Learn-then-Test
(Hoeffding-Bentkus) and the betting certificate alone (isolates what rule selection adds). Filled marker: 95% interval above 0;
hollow: includes 0. Palette: reference categorical slots 3 (aqua) and 7 (violet), distinct from the regime figure."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

AQUA, VIOLET, INK, MUTED, GRID = "#1baf7a", "#6250d6", "#0b0b0b", "#52514e", "#e4e3df"
RULES = [("EcoStop", "curve rule"), ("calibrated patience", "calibrated patience"), ("LC-PFN + gates", "LC-PFN + gates")]


def main(out="figures/fig_pooled.pdf"):
    d = pd.read_csv("figures/tierA/c1_framework_oof_diffs.csv")
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(3.45, 3.6))
    rows, y, ticks, labels = [], 0, [], []
    for rho in (0.10, 0.15, 0.20):
        for key, lab in RULES:
            for kind, col, off in (("LTT-HB", AQUA, 0.15), ("betting only", VIOLET, -0.15)):
                r = d[(d.rho == rho) & (d.baseline == f"{kind}, {key}")].iloc[0]
                ax.plot([r.lo, r.hi], [y + off, y + off], color=col, lw=1.1, zorder=2)
                ax.plot(r["diff"], y + off, "o", ms=5, color=col, mfc=col if r.lo > 0 else "white", mec=col, mew=1.2, zorder=3)
            ticks.append(y); labels.append(lab); y -= 1
        y -= 0.45
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.2)
    for i, rho in enumerate((0.10, 0.15, 0.20)):
        yy = ticks[3 * i] + 0.62
        ax.text(ax.get_xlim()[0] if False else -4.4, yy, f"$\\rho={rho:.2f}$", fontsize=8, color=INK, va="center", ha="left", fontweight="bold")
    ax.set_xlim(-5, 14)
    ax.set_ylim(ticks[-1] - 0.7, ticks[0] + 1.0)
    ax.set_xlabel("framework minus baseline (EW pp)")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h = [Line2D([], [], color=AQUA, lw=1.4, marker="o", ms=5, label="vs standard certification (LTT-HB)"),
         Line2D([], [], color=VIOLET, lw=1.4, marker="o", ms=5, label="vs betting certificate alone"),
         Line2D([], [], color=MUTED, lw=0, marker="o", ms=5, label="interval above 0"),
         Line2D([], [], color=MUTED, lw=0, marker="o", ms=5, mfc="white", mew=1.2, label="includes 0")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=170)
    print("saved", out)


if __name__ == "__main__":
    main()
