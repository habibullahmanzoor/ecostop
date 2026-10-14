"""Framework overview (v3, revised): offline certification (rule families, betting bound, Bonferroni selection, scope card) and the
online per-run decision with abstention. Diagram of the method; no measured numbers. Right-angle connectors, Arial-class font."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

have = {f.name for f in fm.fontManager.ttflist}
FONT = next((n for n in ("Arial", "Segoe UI", "Liberation Sans", "Helvetica", "DejaVu Sans") if n in have), "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic",
                     "mathtext.bf": FONT + ":bold", "pdf.fonttype": 42})
OFF, ON, WARN, INK, MUTED = "#1f5fbf", "#d9531e", "#8a2a1f", "#0b0b0b", "#4a4a47"
DASH = (0, (5, 3))


def box(ax, x, y, w, h, text, edge, fs=9.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.09", lw=1.6, ec=edge, fc="white", zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal",
            linespacing=1.35, zorder=4)


def line(ax, pts, color, lw=1.8, ls="-"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=2)


def head(ax, a, b, color, lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15, lw=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=2))


def main(out="figures/figure_framework_v3.pdf"):
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    ax.set_xlim(0, 10.65)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    ax.text(0.1, 5.78, "Offline, once per calibration population", fontsize=12.5, fontweight="bold", color=OFF)
    y, h, w = 4.3, 1.25, 1.84
    xs = [0.1, 2.24, 4.38, 6.52, 8.66]
    texts = [
        "Calibration jobs\n(dataset, model,\nrecipe); seeds\nnested in jobs",
        "Rule families, each\nwith one ordered\nparameter: curve score,\npatience ($p = fT$),\nLC-PFN",
        "Replay every setting\non every job:\nloss $= \\mathbf{1}[A_T - A_\\tau > \\epsilon]$,\njob = share of seeds",
        "Certify by fixed-\nsequence testing with a\nbetting bound; choose a\nrule per domain",
        "Freeze: rule and\nsetting per domain,\nscope card (schedule,\nwarm-up, horizon)",
    ]
    for x, tx in zip(xs, texts):
        box(ax, x, y, w, h, tx, OFF)
    for i in range(4):
        head(ax, (xs[i] + w + 0.02, y + h / 2), (xs[i + 1] - 0.02, y + h / 2), OFF)
    ax.text(0.1, 4.03, "guarantee: with probability $\\geq 1-\\beta$ over the calibration jobs, a new independent job "
            "violates the tolerance with probability $\\leq \\rho$", ha="left", fontsize=9.2, color=MUTED, style="italic")

    # frozen artefacts: one vertical drop from the Freeze box, a horizontal bus, two branches
    bus_y = 3.5
    line(ax, [(xs[4] + w / 2, y - 0.02), (xs[4] + w / 2, bus_y), (3.5, bus_y)], OFF, ls=DASH)
    ax.text(7.0, bus_y + 0.1, "frozen rule, setting and scope card", ha="center", fontsize=9.2, color=OFF)
    head(ax, (3.5, bus_y), (3.5, 2.58), OFF, ls=DASH)
    head(ax, (6.3, bus_y), (6.3, 2.88), OFF, ls=DASH)
    ax.plot([6.3], [bus_y], "o", ms=4.5, color=OFF, zorder=3)

    ax.text(0.1, 3.0, "Online, per training run", fontsize=12.5, fontweight="bold", color=ON)
    box(ax, 0.1, 1.45, 1.95, 1.1, "New training job:\nvalidation curve\nobserved epoch\nby epoch", ON)
    box(ax, 2.55, 1.45, 1.95, 1.1, "Scope card:\nis the recipe inside the\ncalibration support?", ON, bold=True)
    box(ax, 5.2, 1.9, 2.2, 0.95, "Run the frozen rule\nbehind the schedule-\nrelative safeguard", ON)
    box(ax, 8.0, 1.9, 2.2, 0.95, "Stop; keep the best\nvalidation checkpoint", ON)
    box(ax, 5.2, 0.45, 2.2, 0.95, "Abstain: run to\ncompletion, no claim", WARN)
    head(ax, (2.07, 2.0), (2.53, 2.0), ON)
    # yes: elbow up-right into the run box
    line(ax, [(4.52, 2.15), (4.85, 2.15), (4.85, 2.37)], ON)
    head(ax, (4.85, 2.37), (5.18, 2.37), ON)
    ax.text(4.62, 2.5, "yes", fontsize=9.6, color=INK)
    # no: elbow down-right into the abstain box
    line(ax, [(4.52, 1.75), (4.85, 1.75), (4.85, 0.92)], WARN)
    head(ax, (4.85, 0.92), (5.18, 0.92), WARN)
    ax.text(4.62, 1.2, "no", fontsize=9.6, color=INK)
    head(ax, (7.42, 2.37), (7.98, 2.37), ON)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight", pad_inches=0.1)
    print("saved", out, "| font:", FONT)


if __name__ == "__main__":
    main()
