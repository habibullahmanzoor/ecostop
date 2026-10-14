"""Pooled versus per-domain certification as a function of the calibration budget (echostop/pool_vs_domain.py, 300 random splits per
point, 259-job pool, calibrated patience, Hoeffding-Bentkus, rho = 0.20). Equal total budget: n jobs per domain certified separately
(per-domain) against the same 3n jobs certified together (pooled). Panels a-c: energy-weighted saving on held-out jobs of one domain.
Panel d: share of splits whose held-out risk in the vision domain exceeds rho.
Violet: per-domain; aqua: pooled. Reference categorical slots 7 and 3."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

VIOLET, AQUA, INK, MUTED, GRID = "#6250d6", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"


def main(out="figures/fig_pool_domain.pdf"):
    d = pd.read_csv("figures/tierA/pool_vs_domain.csv")
    rho = 0.20
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK})
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.35), gridspec_kw={"wspace": 0.42})
    pat = d[(d.rule == "calibrated patience") & (d.rho == rho)]
    for a, dom, title in zip(ax[:3], ("vision", "nlp", "tabular"), ("(a) vision", "(b) NLP", "(c) tabular")):
        for m, col in (("pooled", AQUA), ("per-domain", VIOLET)):
            x = pat[(pat.domain == dom) & (pat.method == m)].sort_values("n_per_domain")
            a.plot(x.n_per_domain, x.EW, "o-", color=col, lw=1.4, ms=3.6, label=m)
        a.set_title(title, loc="left", fontsize=8.2)
        a.set_xlabel("jobs per domain")
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_ylabel("held-out EW saving (%)")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    a = ax[3]
    v = d[(d.rho == rho) & (d.domain == "vision")]
    for rule, ls in (("calibrated patience", "-"), ("EcoStop", "--")):
        for m, col in (("pooled", AQUA), ("per-domain", VIOLET)):
            x = v[(v.rule == rule) & (v.method == m)].sort_values("n_per_domain")
            a.plot(x.n_per_domain, 100 * x.exceed, ls, marker="o", ms=3.4, color=col, lw=1.3,
                   label=f"{m}, {'patience' if rule.startswith('cal') else 'curve rule'}")
    a.axhline(5, color=INK, ls=":", lw=0.9)
    a.set_title("(d) vision risk > $\\rho$", loc="left", fontsize=8.2)
    a.set_xlabel("jobs per domain"); a.set_ylabel("% of splits")
    a.legend(frameon=False, fontsize=5.8, loc="upper left")
    a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
