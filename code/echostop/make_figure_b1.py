"""Fresh-test figure: realised per-run risk of the frozen EcoStop thresholds by domain and by vision
schedule family, against the two targets. Palette validated (light surface); markers + labels as
secondary encoding."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
C15, C20 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})

fresh = pd.read_csv("figures/tierA/b1_fresh_test.csv")
sched = pd.read_csv("figures/tierA/b1_schedule_breakdown.csv")
eco = fresh[(fresh.method == "EcoStop") & (fresh.scheme == "pooled")].set_index("rho")
sch = sched[sched.rule == "EcoStop"]

rows = [  # label, n jobs, risk at .15, risk at .20
    ("Tabular", 11, eco.loc[0.15, "risk_tabular"], eco.loc[0.20, "risk_tabular"]),
    ("NLP", 8, eco.loc[0.15, "risk_nlp"], eco.loc[0.20, "risk_nlp"]),
]
for fam, lab in (("cosine + warm-up", "Vision, cosine + warm-up"), ("step decay", "Vision, step decay"),
                 ("one-cycle", "Vision, one-cycle")):
    r15 = sch[(sch.rho == 0.15) & (sch.family == fam)].iloc[0]
    r20 = sch[(sch.rho == 0.20) & (sch.family == fam)].iloc[0]
    rows.append((lab, int(r15.jobs), r15.risk, r20.risk))
rows.append(("All vision", 16, eco.loc[0.15, "risk_vision"], eco.loc[0.20, "risk_vision"]))

fig, ax = plt.subplots(figsize=(3.6, 2.9))
ys = list(range(len(rows)))[::-1]
for y, (lab, n, a, b) in zip(ys, rows):
    ax.plot([a, b], [y, y], color=GRID, lw=2, zorder=1)
    ax.scatter(a, y, s=34, color=C15, marker="o", zorder=3, edgecolor=SURF, linewidth=0.7)
    ax.scatter(b, y, s=34, color=C20, marker="s", zorder=3, edgecolor=SURF, linewidth=0.7)
for x, c in ((0.15, C15), (0.20, C20)):
    ax.axvline(x, color=c, lw=1.0, ls="--", alpha=0.8, zorder=0)
ax.set_yticks(ys)
ax.set_yticklabels([f"{lab} (n={n})" for lab, n, _, _ in rows], fontsize=7.5)
ax.set_xlim(-0.02, 0.62)
ax.set_xlabel("Realised per-run risk")
ax.scatter([], [], color=C15, marker="o", s=34, label=r"frozen target $\rho$=0.15")
ax.scatter([], [], color=C20, marker="s", s=34, label=r"frozen target $\rho$=0.20")
ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2,
          handletextpad=0.2, columnspacing=1.0, markerscale=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(True, axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/fig_b1_fresh_risk.pdf", bbox_inches="tight")
print(pd.DataFrame(rows, columns=["group", "n", "risk@.15", "risk@.20"]).round(3).to_string(index=False))
