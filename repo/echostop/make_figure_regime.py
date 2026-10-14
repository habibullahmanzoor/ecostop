"""Figure: where EcoStop wins. Forest plot of energy-weighted saving differences versus calibrated patience (both certified with the
same Hoeffding-Bentkus bound, same rho), pooled out-of-fold over 259 jobs, by regime. Source: figures/tierA/subgroup_wins.csv
(echostop/subgroup_wins.py). Series: EcoStop framework (blue) and EcoStop's curve rule (orange). Outcome is encoded by marker
shape as well as position: filled = interval above 0 (win), hollow = includes 0 (tie), cross = below 0 (loss).
Palette: reference categorical slots 1-2 (adjacent-pair CVD safe)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
ROWS = [("ALL", "All 259 jobs", 0), ("domain: vision", "Vision CNNs", 0), ("arch: resnet9", "ResNet-9, cosine", 1),
        ("domain: tabular", "Tabular MLPs", 0), ("arch: MLP", "MLP, cosine", 1), ("arch: MLP (wide)", "wide MLP, AdamW", 1),
        ("domain: nlp", "NLP fine-tuning", 0), ("arch: DistilBERT", "DistilBERT, 10 ep", 1), ("arch: BERT-small", "BERT-small, 15 ep", 1),
        ("arch: ELECTRA-small", "ELECTRA-small, 8 ep", 1), ("arch: BERT-L6", "BERT-L6, 12 ep", 1)]


def main(out="figures/fig_regime.pdf"):
    df = pd.read_csv("figures/tierA/subgroup_wins.csv")
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.1), sharey=True)
    n = len(ROWS)
    for ax, rho in zip(axes, (0.15, 0.20)):
        x = df[df.rho == rho].set_index("subgroup")
        for i, (key, label, indent) in enumerate(ROWS):
            if key not in x.index:
                continue
            r = x.loc[key]
            y = n - 1 - i
            for off, col, d, lo, hi in ((0.17, BLUE, r.fw_minus_pat, r.lo, r.hi), (-0.17, ORANGE, r.rule_minus_pat, r.rlo, r.rhi)):
                if pd.isna(lo):
                    continue
                kind = "win" if lo > 0 else ("loss" if hi < 0 else "tie")
                ax.plot([lo, hi], [y + off, y + off], color=col, lw=1.1, solid_capstyle="butt", zorder=2)
                if kind == "win":
                    ax.plot(d, y + off, "o", ms=5.5, color=col, mec=col, zorder=3)
                elif kind == "tie":
                    ax.plot(d, y + off, "o", ms=5.5, mfc="white", mec=col, mew=1.2, zorder=3)
                else:
                    ax.plot(d, y + off, marker="X", ms=6, color=col, mec="white", mew=0.4, zorder=3)
        ax.axvline(0, color=INK, lw=0.8, zorder=1)
        ax.set_xlim(-30, 30)
        ax.set_title(f"$\\rho={rho:.2f}$", loc="left", fontsize=9, color=INK)
        
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    labels = []
    xs = df[df.rho == 0.15].set_index("subgroup")
    for key, label, indent in ROWS:
        share = xs.loc[key, "energy_share"] if key in xs.index else float("nan")
        jobs = int(xs.loc[key, "jobs"]) if key in xs.index else 0
        labels.append(("  " if indent else "") + f"{label}  ({jobs} jobs, {share:.1f}% of energy)" if key != "ALL" else f"{label}")
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(labels[::-1], fontsize=7.2)
    h = [Line2D([], [], color=BLUE, lw=1.4, marker="o", ms=5, label="EcoStop framework"),
         Line2D([], [], color=ORANGE, lw=1.4, marker="o", ms=5, label="EcoStop curve rule"),
         Line2D([], [], color=MUTED, lw=0, marker="o", ms=5, label="interval above 0 (win)"),
         Line2D([], [], color=MUTED, lw=0, marker="o", ms=5, mfc="white", mew=1.2, label="includes 0 (tie)"),
         Line2D([], [], color=MUTED, lw=0, marker="X", ms=6, label="below 0 (loss)")]
    fig.supxlabel("energy-weighted saving minus calibrated patience (percentage points)", fontsize=8, y=0.095)
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=7.4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
