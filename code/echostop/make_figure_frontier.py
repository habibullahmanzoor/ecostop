"""Energy-saved vs. accuracy-sacrificed frontier (paper §6.7/§6.6 companion,
session addition prompted by external review feedback): the single figure
that visually answers "how much training compute can I throw away before
model quality changes," faceted by domain -- which is also the figure that
carries the regime-map story (which method sits closest to the top-left
ideal corner varies systematically by domain).

Reuses the per-job saved_pct/gap_pp fields already computed for the energy
report dashboard (figures/energy_report_data.json) -- no new evaluation,
just a different view of numbers already audited elsewhere in the project.
Same conventions as every other figure here: colorblind-safe fixed
categorical colors (Okabe-Ito derived, matching figure_meta_comparison.png
and figure_accuracy_gap.png exactly), single axis per panel, no rainbow.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPS = 0.5
X_MAX = 3.5  # a handful of tiny-dataset outliers (hayes-roth, parkinsons, up
             # to 9.4pp) are clipped here for legibility -- see the caption
             # and §6.7's existing prose, which already discloses them.

METHOD_COLOR = {"oracle": "#000000", "calibrated": "#D55E00", "patience": "#0072B2", "meta": "#999999"}
METHOD_LABEL = {"oracle": "Oracle", "calibrated": "EcoStop (Online)", "patience": "Patience (fair)", "meta": "EcoStop-Meta"}
METHOD_ORDER = ["oracle", "patience", "meta", "calibrated"]  # draw EcoStop last so it's on top

DOMAIN_PANELS = [
    ("NLP (fine-tuning)", "NLP (fine-tuning)"),
    ("Vision (cosine, from-scratch)", "Vision (cosine)"),
    ("Tabular / OpenML", "Tabular"),
    ("Vision (step-decay, from-scratch)", "Vision (step-decay)"),
]


def main() -> None:
    with open("figures/energy_report_data.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), sharex=True, sharey=True)
    n_clipped = 0

    for ax, (domain_key, domain_title) in zip(axes.flat, DOMAIN_PANELS):
        rows = [r for r in data if r["domain"] == domain_key]
        ax.axvline(EPS, color="#D55E00", linestyle="--", linewidth=1.1, alpha=0.6, zorder=1)
        ax.axvline(0, color="#bbbbbb", linewidth=0.8, zorder=1)

        for method in METHOD_ORDER:
            xs, ys = [], []
            for r in rows:
                xs.append(min(r[f"{method}_gap_pp"], X_MAX))
                ys.append(r[f"{method}_saved_pct"])
            xs = np.array(xs)
            ys = np.array(ys)
            clip_mask = np.array([r[f"{method}_gap_pp"] > X_MAX for r in rows])
            n_clipped += int(clip_mask.sum())
            ax.scatter(xs[~clip_mask], ys[~clip_mask], s=34, color=METHOD_COLOR[method],
                       alpha=0.65, linewidth=0, label=METHOD_LABEL[method], zorder=3)
            if clip_mask.any():
                ax.scatter(xs[clip_mask], ys[clip_mask], s=60, color=METHOD_COLOR[method],
                           marker=">", alpha=0.85, linewidth=0, zorder=4)

        ax.set_title(f"{domain_title}  (n={len(rows)})", fontsize=12.5, fontweight="bold")
        ax.set_xlim(-1.8, X_MAX)
        ax.set_ylim(-3, 103)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e5e5e5", linewidth=0.7, zorder=0)

    for ax in axes[-1, :]:
        ax.set_xlabel("Accuracy gap vs. full training (pp), higher = worse", fontsize=11.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Energy saved (%)", fontsize=11.5)
    for ax in axes.flat:
        ax.tick_params(labelsize=10)

    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLOR[m],
                          markersize=8, alpha=0.8, label=METHOD_LABEL[m]) for m in METHOD_ORDER]
    handles.append(plt.Line2D([0], [0], color="#D55E00", linestyle="--", linewidth=1.1,
                              alpha=0.6, label=f"tolerance ε={EPS}pp"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Energy saved vs. accuracy gap, by domain and method",
                  fontsize=14.5, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.11, hspace=0.32, wspace=0.06)
    out = "figures/figure10.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out}")
    print(f"points clipped at x={X_MAX}pp (arrow markers): {n_clipped}")


if __name__ == "__main__":
    main()
