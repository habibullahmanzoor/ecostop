"""Framework overview figure (Figure 1 in the journal version of the paper).

A schematic of the EcoStop pipeline: the online per-run decision loop
(top) and the offline calibration procedure that produces the deployed
threshold (bottom). This is a design/architecture figure, not a results
figure -- no numbers on it are measured, it is a diagram of the method
described in the Method section.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ONLINE_COLOR = "#0072B2"
OFFLINE_COLOR = "#D55E00"
NEUTRAL_COLOR = "#555555"
BOX_FACE = "#FFFFFF"


def box(ax, xy, w, h, text, edgecolor, fontsize=10.5, fontweight="normal", zorder=3):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=edgecolor, facecolor=BOX_FACE, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, color="#1a1a1a", zorder=zorder + 1,
             linespacing=1.35)
    return patch


def arrow(ax, start, end, color, style="-|>", lw=1.8, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color, zorder=2,
                         connectionstyle=connectionstyle, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    # ---- ONLINE per-run loop (top) ----
    ax.text(0.15, 7.15, "Online, per training run (deployed policy)", fontsize=13,
            fontweight="bold", color=ONLINE_COLOR)

    b1 = box(ax, (0.3, 5.55), 2.15, 1.15,
             "Training run\nepoch $t$:\nvalidation $Q_t$,\nenergy draw", ONLINE_COLOR)
    b2 = box(ax, (2.95, 5.55), 2.15, 1.15,
             "Running best\n$B_t=\\max_{s\\leq t}Q_s$", ONLINE_COLOR)
    b3 = box(ax, (5.6, 5.55), 2.35, 1.15,
             "Curve-fit predictor\n(§4.2–4.3):\ntrailing window,\nsaturation model", ONLINE_COLOR)
    b4 = box(ax, (8.35, 5.55), 2.15, 1.15,
             "Continuation score\n$s_t=\\hat P(G_t>\\epsilon\\mid H_t)$", ONLINE_COLOR)
    b5 = box(ax, (5.6, 3.85), 2.35, 1.1,
             "Schedule-relative\nplateau safeguard\n(§4.4)", ONLINE_COLOR)
    decision = box(ax, (8.35, 3.85), 2.15, 1.1,
                    "$s_t<\\hat\\gamma$ ?", ONLINE_COLOR, fontweight="bold")

    arrow(ax, (2.45, 6.12), (2.95, 6.12), ONLINE_COLOR)
    arrow(ax, (5.1, 6.12), (5.6, 6.12), ONLINE_COLOR)
    arrow(ax, (7.95, 6.12), (8.35, 6.12), ONLINE_COLOR)
    arrow(ax, (6.775, 5.55), (6.775, 4.95), ONLINE_COLOR)
    arrow(ax, (7.95, 4.4), (8.35, 4.4), ONLINE_COLOR)
    arrow(ax, (9.425, 5.55), (9.425, 4.95), ONLINE_COLOR)

    ax.text(11.15, 4.95, "continue\n(next epoch)", fontsize=9.5, color=NEUTRAL_COLOR,
            ha="center", va="center", style="italic")
    arrow(ax, (10.5, 4.4), (11.9, 4.4), ONLINE_COLOR)
    arrow(ax, (11.9, 4.4), (11.9, 6.12), ONLINE_COLOR, connectionstyle="arc3,rad=0.0")
    arrow(ax, (11.9, 6.12), (10.5, 6.12), ONLINE_COLOR)

    stop_box = box(ax, (8.85, 2.55), 1.15, 0.75, "stop", "#000000",
                    fontsize=11, fontweight="bold")
    arrow(ax, (9.425, 3.85), (9.425, 3.3), "#000000", lw=2.2)
    ax.text(9.9, 3.55, "yes", fontsize=9, color=NEUTRAL_COLOR, style="italic")

    # ---- OFFLINE calibration (bottom) ----
    ax.text(0.15, 3.15, "Offline, once per deployment (§3.1, §6.3)", fontsize=13,
            fontweight="bold", color=OFFLINE_COLOR)

    c1 = box(ax, (0.3, 1.15), 2.35, 1.35,
             "Calibration corpus\n(130 jobs, 3\ndomains, §5)", OFFLINE_COLOR)
    c2 = box(ax, (3.05, 1.15), 2.55, 1.35,
             "Simulate $\\tau_\\gamma(r)$ and\naudited loss $L_\\gamma(r)$\nfor each candidate $\\gamma$", OFFLINE_COLOR)
    c3 = box(ax, (6.0, 1.15), 2.75, 1.35,
             "Hoeffding UCB +\nBonferroni across grid;\nanytime-valid correction\n(§6.3.2)", OFFLINE_COLOR)
    c4 = box(ax, (9.15, 1.15), 2.35, 1.35,
             "Certify largest $\\gamma$\nwith UCB $\\leq\\rho$\n$\\Rightarrow\\ \\hat\\gamma$", OFFLINE_COLOR,
             fontweight="bold")

    arrow(ax, (2.65, 1.825), (3.05, 1.825), OFFLINE_COLOR)
    arrow(ax, (5.6, 1.825), (6.0, 1.825), OFFLINE_COLOR)
    arrow(ax, (8.75, 1.825), (9.15, 1.825), OFFLINE_COLOR)

    # threshold feeds into the online decision box
    arrow(ax, (10.325, 2.5), (9.6, 3.85), OFFLINE_COLOR, lw=2.0,
          connectionstyle="arc3,rad=-0.15")
    ax.text(10.55, 3.15, "$\\hat\\gamma$ deployed\nunchanged", fontsize=9, color=OFFLINE_COLOR,
            ha="left", va="center", style="italic")

    legend_handles = [
        Line2D([0], [0], color=ONLINE_COLOR, lw=2.2, label="Online decision loop (every run)"),
        Line2D([0], [0], color=OFFLINE_COLOR, lw=2.2, label="Offline calibration (once, before deployment)"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
              ncol=2, frameon=False, fontsize=10.5)

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig("figures/figure_framework.png", dpi=150)
    fig.savefig("figures/figure_framework.pdf")
    print("wrote figures/figure_framework.png and .pdf")


if __name__ == "__main__":
    main()
