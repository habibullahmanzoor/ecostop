"""Figure for the C1 confirmatory study (82 fine-tuning jobs, 41 datasets).
(a) risk-saving frontiers: energy-weighted saving of every setting of every rule against realised job-level risk; dotted line = the
    pre-registered primary risk cap 0.10. (b) per dataset, energy saved by EcoStop-SR minus calibrated patience at their primary
    operating points (risk cap 0.10); positive = EcoStop-SR saves more. Palette: reference categorical slots.
usage: python -m echostop.make_figure_c1 [--out figures/fig_c1.pdf]"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import c1_eval as CE
from echostop import c1_rules as CR

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SER = {CR.PAT_MD: ("#2a78d6", "calibrated patience"), CR.ECO: ("#eb6834", "EcoStop-SR"), CR.ECO_P: ("#1baf7a", "EcoStop (paper gates)"),
       CR.PFN_SR: ("#6250d6", "LC-PFN + SR gates"), CR.PAT_FIX: ("#9a9892", "patience fixed in epochs")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/fig_c1.pdf")
    a = ap.parse_args()
    jobs, meta, _ = CE.load_logs("logs/c1/nlp")
    names = sorted(jobs)
    ds = np.array([k.split("|")[0] for k in names])
    M = CE.matrices({k: jobs[k] for k in names}, True, "cache/c1_lcpfn_traces.pkl")
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.0), gridspec_kw={"width_ratios": [1.0, 1.05]})
    for f, (col, lab) in SER.items():
        m = M[f]
        risk, ew = m["L"].mean(1), 100 * m["ES"].sum(1) / m["et"].sum()
        # upper-left envelope: best saving at or below each risk level
        o = np.argsort(risk)
        r_s, e_s = risk[o], np.maximum.accumulate(ew[o])
        ax[0].plot(r_s, e_s, "-", color=col, lw=1.4 if f in (CR.ECO, CR.PAT_MD) else 1.0, label=lab, zorder=3 if f in (CR.ECO, CR.PAT_MD) else 2)
    ax[0].axvline(CR.PRIMARY_RISK, color=INK, ls=":", lw=0.9)
    ax[0].text(CR.PRIMARY_RISK + 0.004, 4, "primary\nrisk cap", fontsize=7, color=MUTED)
    ax[0].set_xlim(0, 0.4); ax[0].set_ylim(0, 72)
    ax[0].set_xlabel("realised job-level risk"); ax[0].set_ylabel("energy-weighted saving (%)")
    ax[0].set_title("(a) in-sample risk-saving envelope", loc="left", fontsize=8.5)
    ax[0].grid(color=GRID, lw=0.6); ax[0].set_axisbelow(True)
    ax[0].legend(frameon=False, fontsize=7, loc="lower right")
    sel = {}
    for f in (CR.ECO, CR.PAT_MD):
        _, ix = CE.front(np.ones((1, len(names))), M[f]["L"], M[f]["ES"], M[f]["et"], CR.PRIMARY_RISK)
        sel[f] = int(ix[0])
    cs = sorted(set(ds))
    diff = []
    for c in cs:
        m_ = ds == c
        e = M[CR.ECO]["et"][m_].sum()
        diff.append(100 * (M[CR.ECO]["ES"][sel[CR.ECO]][m_].sum() - M[CR.PAT_MD]["ES"][sel[CR.PAT_MD]][m_].sum()) / e)
    o = np.argsort(diff)
    d = np.array(diff)[o]
    pos, neg = "#2a78d6", "#eb6834"
    ax[1].barh(range(len(cs)), d, color=[pos if v > 0 else neg for v in d], height=0.68)
    ax[1].set_yticks(range(len(cs)))
    ax[1].set_yticklabels([cs[i] for i in o], fontsize=5.6)
    ax[1].axvline(0, color=INK, lw=0.8)
    nb = int((d > 1e-9).sum()); nw = int((d < -1e-9).sum())
    ax[1].set_title("(b) per dataset, risk cap 0.10", loc="left", fontsize=8.5)
    note = "\n".join([f"EcoStop-SR better: {nb}", f"patience better: {nw}", f"ties: {len(cs) - nb - nw}"])
    ax[1].text(0.02, 0.97, note, transform=ax[1].transAxes,
               ha="left", va="top", fontsize=7.5, color=INK, bbox=dict(fc="white", ec=GRID, lw=0.6))
    ax[1].set_xlabel("EcoStop-SR minus patience (pp)")
    ax[1].grid(axis="x", color=GRID, lw=0.6); ax[1].set_axisbelow(True)
    for s in ("top", "right"):
        ax[0].spines[s].set_visible(False); ax[1].spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out); fig.savefig(a.out.replace(".pdf", ".png"), dpi=150)
    print("saved", a.out, "| datasets favouring SR", nb, "patience", nw)


if __name__ == "__main__":
    main()
