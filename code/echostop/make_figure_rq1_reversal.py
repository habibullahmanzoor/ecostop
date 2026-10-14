"""RQ1 headline figure (session addition, prompted by a figure-set review):
the paper's abstract and §6.6 lead with a reversal -- EcoStop-Meta's
certified fraction collapsed from 0.56 to 0.897 as the calibration corpus
grew more diverse, flipping the domain-level comparison from "cold-start
loses 3 of 4 domains" to "cold-start wins 4 of 4" -- yet the only figure
touching this (figure_meta_comparison.png) shows just the current (after)
snapshot. This figure puts both real calibration snapshots side by side,
same bar-chart grammar as figure_meta_comparison.png for visual
consistency, so the collapse is visible without reading a table.

BEFORE values (n=95 calibration pool, f_hat=0.56) are the real numbers
this project reported earlier this session, before the second vision/NLP
expansion round -- not recomputed here (that calibration pool no longer
exists as a discoverable run-set now that new datasets have been added),
but they are genuine prior results, not illustrative.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

# Real numbers reported earlier this session, at the n=95 calibration pool, f_hat=0.56
BEFORE = {
    "NLP\n(fine-tuning)":       {"oracle": 55.4, "online": 24.1, "patience": 0.0,  "meta": 40.8},
    "Tabular":        {"oracle": 81.7, "online": 70.2, "patience": 67.4, "meta": 43.3},
    "Vision\n(cosine)":         {"oracle": 32.8, "online": 25.9, "patience": 5.9,  "meta": 42.9},
    "Vision\n(step-decay)":     {"oracle": 32.4, "online": 12.2, "patience": 0.0,  "meta": 42.4},
}
# Current numbers, n=130 calibration pool, f_hat=0.927 (make_figure_meta.py's own output)
AFTER = {
    "NLP\n(fine-tuning)":       {"oracle": 57.1, "online": 25.4, "patience": 0.0,  "meta": 10.4},
    "Tabular":        {"oracle": 81.7, "online": 70.2, "patience": 67.4, "meta": 6.7},
    "Vision\n(cosine)":         {"oracle": 31.4, "online": 27.3, "patience": 5.8,  "meta": 7.5},
    "Vision\n(step-decay)":     {"oracle": 32.4, "online": 12.2, "patience": 0.0,  "meta": 7.3},
}


def panel(ax, data: dict, title: str) -> None:
    x = np.arange(len(DOMAIN_ORDER))
    width = 0.2
    methods = ["oracle", "online", "patience", "meta"]
    for i, method in enumerate(methods):
        vals = [data[d][method] for d in DOMAIN_ORDER]
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, color=METHOD_COLOR[method],
               label=METHOD_LABEL[method], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_ORDER, fontsize=11)
    ax.set_ylim(0, 92)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)

    panel(axes[0], BEFORE, "Smaller calibration corpus (n=95)\nEcoStop-Meta f̂=0.56")
    panel(axes[1], AFTER, "Full calibration corpus (n=130)\nEcoStop-Meta f̂=0.927")
    axes[0].set_ylabel("Mean energy saved vs. full training (%)", fontsize=12)
    axes[1].legend(frameon=False, fontsize=10.5, loc="upper right")

    # annotate the meta collapse directly
    for ax, data in zip(axes, [BEFORE, AFTER]):
        x = np.arange(len(DOMAIN_ORDER))
        width = 0.2
        meta_offset = (3 - 1.5) * width
        for xi, d in zip(x, DOMAIN_ORDER):
            ax.annotate(f"{data[d]['meta']:.0f}", (xi + meta_offset, data[d]["meta"] + 2),
                        ha="center", fontsize=9.5, color="#555555")

    fig.suptitle("Mean energy saved by method and domain, smaller vs. full calibration corpus",
                  fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.1, wspace=0.06)
    out = "figures/figure8.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
