"""Paper figures from the Tier A analysis: (a) matched-risk energy frontier, (b) validity over
random calibration/test splits. Out-of-fold throughout. Light surface, validated palette,
distinct markers + direct labels as secondary encoding."""
from __future__ import annotations

import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import tierA
from echostop.ablation import sim
from echostop.lcpfn_real import PFN_GRID

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
STYLE = {  # method -> (colour, marker, label)
    "EcoStop": ("#2a78d6", "o", "EcoStop"),
    "LC-PFN + EcoStop gates": ("#4a3aa7", "P", "LC-PFN (+gates)"),
    "Patience (p=f*T, min-delta 0.1)": ("#eb6834", "s", "Patience p=fT, min-delta"),
    "Patience (fixed p)": ("#1baf7a", "^", "Patience, fixed p"),
    "Fixed budget fraction": ("#e87ba4", "D", "Fixed budget fraction"),
}
RHOS = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def get_mats():
    cache = "cache/tierA_mats_all.pkl"
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    d = pickle.load(open("cache/tierA_mats.pkl", "rb"))
    mats, names, dom = d["mats"], d["names"], d["dom"]
    jobs, _, _ = tierA.load()
    pf = pickle.load(open("cache/lcpfn_traces.pkl", "rb"))
    pl = [(f"g={g}", (lambda df, tr, g=g: sim(tr, g, len(df), max(2, round(0.125 * len(df))), 2))) for g in PFN_GRID]
    mats["LC-PFN + EcoStop gates"] = tierA.evaluate_family(jobs, pf, pl)
    pickle.dump((mats, names, dom), open(cache, "wb"))
    return mats, names, dom


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def frontier(mats, dom):
    rows = []
    for m in STYLE:
        for rho in RHOS:
            r = tierA.crossfit(*mats[m], dom, rho, 0.05, K=5, R=20)
            rows.append({"method": m, "rho": rho, "oof_risk": r["risk"], "mean_saved": r["saved"], "ew": r["ew"]})
    df = pd.DataFrame(rows)
    df.to_csv("figures/tierA/frontier_curves.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, col, yl in ((axs[0], "mean_saved", "Mean energy saved per job (%)"),
                        (axs[1], "ew", "Energy-weighted saving (%)")):
        for m, (c, mk, lab) in STYLE.items():
            d = df[df.method == m].sort_values("oof_risk")
            ax.plot(d.oof_risk, d[col], color=c, lw=1.4, marker=mk, ms=4.5, mec=SURF, mew=0.6, label=lab)
        style_ax(ax)
        ax.set_xlabel("Realised out-of-fold violation risk")
        ax.set_ylabel(yl)
    axs[0].legend(frameon=False, fontsize=7, loc="lower right")
    axs[0].set_title("(a) Per-job mean", fontsize=9, color=INK, loc="left")
    axs[1].set_title("(b) Energy-weighted", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig("figures/fig_frontier_oof.pdf", bbox_inches="tight")
    plt.close(fig)
    return df


def validity(mats, dom, rho=0.20, beta=0.05, n_splits=1000, frac_test=0.3, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for m in ("EcoStop", "Patience (p=f*T, min-delta 0.1)"):
        loss, saved, e_s, e_t = mats[m]
        J = loss.shape[1]
        risks, savs = [], []
        for _ in range(n_splits):
            te = np.zeros(J, bool)
            for d in np.unique(dom):
                idx = np.where(dom == d)[0]
                te[rng.choice(idx, max(1, int(round(frac_test * len(idx)))), replace=False)] = True
            i = tierA.select(loss[:, ~te], rho, beta)
            risks.append(0.0 if i is None else loss[i, te].mean())
            savs.append(0.0 if i is None else saved[i, te].mean())
        out[m] = (np.array(risks), np.array(savs))
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    bins = np.linspace(0, 0.30, 31)
    for m, (c, _, lab) in ((k, STYLE[k]) for k in out):
        ax.hist(out[m][0], bins=bins, color=c, alpha=0.55, edgecolor=SURF, lw=0.5, label=lab)
    ax.axvline(rho, color=INK, lw=1.2, ls="--")
    ax.text(rho + 0.003, ax.get_ylim()[1] * 0.92, r"target $\rho$=0.20", fontsize=7, color=INK)
    style_ax(ax)
    ax.set_xlabel("Realised test risk on held-out 30% of jobs")
    ax.set_ylabel("Random splits (of 1000)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.85))
    fig.tight_layout()
    fig.savefig("figures/fig_validity_splits.pdf", bbox_inches="tight")
    plt.close(fig)
    summ = {m: {"mean_risk": float(r.mean()), "p95_risk": float(np.percentile(r, 95)),
                "frac_splits_over_rho": float((r > rho).mean()), "mean_saved": float(s.mean())}
            for m, (r, s) in out.items()}
    pd.DataFrame(summ).T.to_csv("figures/tierA/validity_splits.csv")
    return summ


if __name__ == "__main__":
    os.makedirs("figures/tierA", exist_ok=True)
    mats, names, dom = get_mats()
    f = frontier(mats, dom)
    print(f.round(3).to_string(index=False))
    print(validity(mats, dom))
