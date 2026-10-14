"""Restored paper material, all out-of-fold: accuracy-cost table, per-domain table, trajectories figure."""
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
from echostop import b1_eval as E
from echostop import tierA
from echostop.ablation import sim
from echostop.calibration import precompute_score_trace, simulate_from_trace
from echostop.lcpfn_real import PFN_GRID
from echostop.tierA import patience_stop_md

NL, BS = chr(10), chr(92)
METHODS = ["EcoStop", "LC-PFN + EcoStop gates", "Patience (p=f*T, min-delta 0.1)", "Patience (fixed p)", "Fixed budget fraction"]
NAME = {"EcoStop": "EcoStop", "LC-PFN + EcoStop gates": "LC-PFN + gates",
        "Patience (p=f*T, min-delta 0.1)": "Patience $p{=}fT$, min-$" + BS + "delta$",
        "Patience (fixed p)": "Patience, fixed $p$", "Fixed budget fraction": "Fixed budget fraction"}


def families(jobs):
    fam = tierA.build_policies(None)
    pf = pickle.load(open("cache/lcpfn_traces.pkl", "rb"))
    pl = [(f"g={g}", (lambda df, tr, g=g: sim(tr, g, len(df), max(2, round(0.125 * len(df))), 2))) for g in PFN_GRID]
    return fam, pl, pf


def tables():
    jobs, traces, dom = tierA.load()
    names = list(jobs)
    dom_arr = np.array([dom[k] for k in names])
    mats, _, _ = pickle.load(open("cache/tierA_mats_all.pkl", "rb"))
    fam, pfn_pl, pfn_tr = families(jobs)
    gaps = {m: tierA.evaluate_gap(jobs, traces, fam[m]) for m in METHODS if m in fam}
    gaps["LC-PFN + EcoStop gates"] = tierA.evaluate_gap(jobs, pfn_tr, pfn_pl)
    e_t = mats["EcoStop"][3]

    gap_rows, dom_rows = [], []
    for rho in (0.15, 0.20):
        for m in METHODS:
            r = tierA.crossfit(*mats[m], dom_arr, rho, 0.05, extra=gaps[m])
            g = r["job_extra"]
            gap_rows.append((rho, m, g.mean(), np.percentile(g, 90), 100 * np.mean(g <= 0.0), 100 * np.mean(g > 0.5), r["risk"]))
            for d in ("vision", "nlp", "tabular"):
                idx = dom_arr == d
                ew = 100 * np.sum(r["job_saved"][idx] / 100 * e_t[idx]) / np.sum(e_t[idx])
                dom_rows.append((rho, m, d, r["risk_by_dom"][d], r["job_saved"][idx].mean(), ew))
    pd.DataFrame(gap_rows, columns=["rho", "method", "mean_gap_pp", "p90_gap_pp", "pct_jobs_gap_le_0", "pct_jobs_gap_gt_eps", "risk"]).to_csv("figures/tierA/accuracy_cost.csv", index=False)
    pd.DataFrame(dom_rows, columns=["rho", "method", "domain", "risk", "mean_saved", "ew_saved"]).to_csv("figures/tierA/domain_oof.csv", index=False)

    # --- accuracy-cost table
    lines = []
    for rho in (0.15, 0.20):
        lines.append(BS + "multicolumn{5}{l}{" + BS + "emph{Target $" + BS + "rho=%.2f$}}" % rho + BS + BS)
        for r in [x for x in gap_rows if x[0] == rho]:
            lines.append("%s & %+.2f & %.2f & %.0f & %.0f " % (NAME[r[1]], r[2], r[3], r[4], r[5]) + BS + BS)
        lines.append(BS + "midrule")
    head = (BS + "begin{tabular}{p{3.6cm}rrrr}" + NL + BS + "toprule" + NL +
            "Rule & Mean gap (pp) & 90th pct.\\ gap (pp) & Jobs with gap $" + BS + "le0$ (" + BS + "%) & Jobs with gap $>" + BS + "epsilon$ (" + BS + "%) " + BS + BS + NL + BS + "midrule" + NL)
    open("tables/accuracy_cost.tex", "w", encoding="utf-8").write(head + NL.join(lines[:-1]) + NL + BS + "bottomrule" + NL + BS + "end{tabular}" + NL)

    # --- per-domain table at rho = 0.20
    lines = []
    for m in METHODS:
        cells = []
        for d in ("vision", "nlp", "tabular"):
            x = [r for r in dom_rows if r[0] == 0.20 and r[1] == m and r[2] == d][0]
            cells.append("%.2f & %.1f & %.1f" % (x[3], x[4], x[5]))
        lines.append("%s & %s " % (NAME[m], " & ".join(cells)) + BS + BS)
    head = (BS + "begin{tabular}{p{3.0cm}rrrrrrrrr}" + NL + BS + "toprule" + NL +
            " & " + BS + "multicolumn{3}{c}{Vision} & " + BS + "multicolumn{3}{c}{NLP} & " + BS + "multicolumn{3}{c}{Tabular} " + BS + BS + NL +
            BS + "cmidrule(lr){2-4}" + BS + "cmidrule(lr){5-7}" + BS + "cmidrule(lr){8-10}" + NL +
            "Rule & Risk & Saved & EW & Risk & Saved & EW & Risk & Saved & EW " + BS + BS + NL + BS + "midrule" + NL)
    open("tables/domain.tex", "w", encoding="utf-8").write(head + NL.join(lines) + NL + BS + "bottomrule" + NL + BS + "end{tabular}" + NL)
    print(pd.DataFrame(gap_rows, columns=["rho", "method", "mean_gap", "p90", "%<=0", "%>eps", "risk"]).round(2).to_string(index=False))


def trajectories():
    dev, dtr, _ = tierA.load()
    fresh, _ = E.load_b1()
    picks = [("Dev: SST-2, DistilBERT", dev["sst2_distilbert"][0]), ("Dev: CIFAR-10, ResNet-9", dev["cifar10_resnet9"][0]),
             ("Dev: tabular, segment", dev["segment"][0]),
             ("Fresh: CIFAR-10, MobileNetV3", fresh["cifar10_mobilenetv3"][0]),
             ("Fresh: F-MNIST, ResNet-18 step", fresh["fashion_mnist_resnet18_step"][0]),
             ("Fresh: MNIST, VGG one-cycle", fresh["mnist_vgg_small"][0])]
    rules = [("EcoStop", "#2a78d6", "-", lambda df, tr: simulate_from_trace(tr, 0.46, len(df))),
             ("Patience p=fT", "#eb6834", "--", lambda df, tr: patience_stop_md(df, max(1, round(0.175 * len(df))), 0.1)),
             ("Patience fixed p=8", "#1baf7a", ":", lambda df, tr: patience_stop_md(df, 8, 0.0))]
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.8))
    for ax, (title, df) in zip(axs.ravel(), picks):
        tr = precompute_score_trace(df, 0.5)
        q = df["val_acc"].values
        ax.plot(np.arange(1, len(q) + 1), q, color="#0b0b0b", lw=1.1)
        for lab, c, ls, fn in rules:
            t = fn(df, tr)
            ax.axvline(t + 1, color=c, ls=ls, lw=1.4, label=lab)
        ax.set_title(title, fontsize=7, loc="left")
        ax.set_xlabel("Epoch")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("Validation acc. (%)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig("figures/fig_trajectories.pdf", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("tables", exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("tables", "all"):
        tables()
    if which in ("fig", "all"):
        trajectories()
