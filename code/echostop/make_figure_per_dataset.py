"""Per-dataset results figure (session addition, prompted directly by
feedback that no figure in the paper lets a reader see individual
per-dataset results -- every other figure aggregates to a domain mean or a
pooled distribution. This plots EcoStop's own energy-saved value for every
one of the 143 individual datasets, sorted within domain, against the
oracle ceiling for that same dataset -- the figure a reviewer would need
to visually confirm "yes, all 143 were actually run," not just count rows
in a text table (energy_report.md).

Oracle ceiling is drawn as a pale "headroom" bar behind each solid EcoStop
bar (same domain hue, lower alpha), not a tick mark -- at this many bars
(143), and sorted by EcoStop's own value rather than oracle's, individual
tick marks scatter into unreadable clutter. A bar-within-a-bar reads
directly: the solid portion is what EcoStop captured, the pale portion
above it is the ceiling left unclaimed.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOMAIN_COLOR = {
    "Tabular / OpenML": "#009E73",
    "NLP (fine-tuning)": "#D55E00",
    "Vision (cosine, from-scratch)": "#0072B2",
    "Vision (step-decay, from-scratch)": "#56B4E9",
}
DOMAIN_ORDER = ["Tabular / OpenML", "NLP (fine-tuning)", "Vision (cosine, from-scratch)", "Vision (step-decay, from-scratch)"]


def main() -> None:
    with open("figures/energy_report_data.json") as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(15, 6.2))

    SHORT_LABEL = {
        "Tabular / OpenML": "Tabular",
        "NLP (fine-tuning)": "NLP\n(fine-tuning)",
        "Vision (cosine, from-scratch)": "Vision\n(cosine)",
        "Vision (step-decay, from-scratch)": "Vision\n(step-decay)",
    }

    LEGEND_LABEL = {
        "Tabular / OpenML": "Tabular",
        "NLP (fine-tuning)": "NLP (fine-tuning)",
        "Vision (cosine, from-scratch)": "Vision (cosine)",
        "Vision (step-decay, from-scratch)": "Vision (step-decay)",
    }

    x0 = 0
    boundaries = []
    for domain in DOMAIN_ORDER:
        rows = [r for r in data if r["domain"] == domain]
        rows.sort(key=lambda r: r["calibrated_saved_pct"])
        n = len(rows)
        xs = np.arange(x0, x0 + n)
        online = [r["calibrated_saved_pct"] for r in rows]
        oracle = [r["oracle_saved_pct"] for r in rows]
        # pale grey headroom bar first (full oracle height), solid EcoStop bar on top --
        # grey rather than domain-tinted so it matches the single legend swatch exactly
        ax.bar(xs, oracle, width=0.85, color="#888888", alpha=0.35,
               linewidth=0, zorder=2)
        ax.bar(xs, online, width=0.85, color=DOMAIN_COLOR[domain], zorder=3,
               label=f"{LEGEND_LABEL[domain]} (n={n})")
        boundaries.append((domain, xs.mean()))
        x0 += n + 6

    ax.set_xticks([b[1] for b in boundaries])
    ax.set_xticklabels([f"{SHORT_LABEL[b[0]]}\n(n={sum(1 for r in data if r['domain']==b[0])})"
                         for b in boundaries], fontsize=14.5)
    ax.set_xlim(-2, x0 - 6 + 2)
    ax.set_ylabel("Energy saved vs. full training (%)", fontsize=16.5)
    ax.tick_params(axis="y", labelsize=14.5)
    ax.set_title("Per-dataset energy saved vs. full training,\n"
                  "sorted within domain, with each dataset's own oracle ceiling",
                  fontsize=18.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8, zorder=0)

    ceiling_handle = plt.Rectangle((0, 0), 1, 1, color="#888888", alpha=0.35,
                                    label="Oracle ceiling (unclaimed headroom)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(ceiling_handle)
    labels.append(ceiling_handle.get_label())
    ax.legend(handles=handles, labels=labels, frameon=False, fontsize=14.5, loc="upper left", ncol=1)

    fig.tight_layout()
    out = "figures/figure5.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".png", ".pdf"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
