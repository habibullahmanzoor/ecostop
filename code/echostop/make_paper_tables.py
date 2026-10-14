"""Generate LaTeX tables for the paper directly from result CSVs (no hand-copied numbers).
Writes tables/*.tex. Tables whose inputs do not exist yet are skipped with a message."""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

OUT = "tables"
os.makedirs(OUT, exist_ok=True)
NICE = {
    "LC-PFN + EcoStop gates": "LC-PFN + EcoStop gates",
    "EcoStop": "EcoStop (curve score + gates)",
    "Patience (fixed p)": "Patience, fixed $p$",
    "Patience (p=f*T)": "Patience, $p=\\lceil fT\\rfloor$",
    "Patience (p=f*T, min-delta 0.1)": "Patience, $p=\\lceil fT\\rfloor$, min-$\\delta$",
    "Fixed budget fraction": "Fixed budget fraction",
    "LC-PFN (original rule: score only, k=1)": "LC-PFN, original rule",
    "LC-PFN + EcoStop gates (flat-gate, k=2)": "LC-PFN + EcoStop gates",
}


def w(name, text):
    open(f"{OUT}/{name}.tex", "w", encoding="utf-8").write(text)
    print("wrote", name)


def f1(x, best=False):
    s = f"{x:.1f}"
    return f"\\textbf{{{s}}}" if best else s


def tab_oof():
    d = pd.read_csv("figures/tierA/crossfit_frontier.csv")
    lp = pd.read_csv("figures/tierA/lcpfn_real.csv")
    lp = lp[lp.method.str.contains("gates")].copy()
    lp["method"] = "LC-PFN + EcoStop gates (flat-gate, k=2)"
    d = pd.concat([d, lp[d.columns.intersection(lp.columns)]], ignore_index=True)
    order = ["EcoStop", "LC-PFN + EcoStop gates (flat-gate, k=2)", "Patience (p=f*T, min-delta 0.1)",
             "Patience (p=f*T)", "Patience (fixed p)", "Fixed budget fraction"]
    lines = []
    for rho in (0.15, 0.20):
        sub = d[(d.rho == rho) & (~d.per_domain)]
        best_ew = sub.energy_weighted_saved.max()
        best_ms = sub.mean_saved.max()
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{Target $\\rho={rho:.2f}$, $\\beta=0.05$}}}}\\\\")
        for m in order:
            r = sub[sub.method == m]
            if r.empty:
                continue
            r = r.iloc[0]
            pdm = d[(d.rho == rho) & (d.per_domain) & (d.method == m)]
            pdv = f1(pdm.iloc[0].energy_weighted_saved) if not pdm.empty else "--"
            lines.append(f"{NICE[m]} & {r.oof_risk:.3f} & {r.risk_vision:.3f} & "
                         f"{f1(r.mean_saved, r.mean_saved == best_ms)} & "
                         f"{f1(r.energy_weighted_saved, r.energy_weighted_saved == best_ew)} & {pdv} \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    body = "\n".join(lines)
    w("oof", "\\begin{tabular}{p{4.5cm}rrrrr}\n\\toprule\n"
      "Rule & Risk & Vision risk & Mean saved (\\%) & Energy-wtd.\\ (\\%) & Energy-wtd., per-domain (\\%) \\\\\n\\midrule\n"
      + body + "\n\\bottomrule\n\\end{tabular}\n")


def tab_ablation():
    d = pd.read_csv("figures/tierA/ablation_sensitivity.csv")
    a5 = d[(d.study == "A5") & (d.rho == 0.20)]
    rows = []
    for _, r in a5.iterrows():
        rows.append(f"{r.variant} & {r.oof_risk:.3f} & {r.risk_vision:.3f} & {r.mean_saved:.1f} & {r.energy_weighted_saved:.1f} \\\\")
    lp = pd.read_csv("figures/tierA/lcpfn_real.csv")
    lp = lp[(lp.rho == 0.20) & (~lp.per_domain)]
    for _, r in lp.iterrows():
        rows.append(f"Real LC-PFN: {r.method.split('(')[0].strip() if 'original' in r.method else 'with EcoStop gates'}"
                    f"{' (score only, $k{=}1$)' if 'original' in r.method else ''} & {r.oof_risk:.3f} & {r.risk_vision:.3f} & "
                    f"{r.mean_saved:.1f} & {r.energy_weighted_saved:.1f} \\\\")
    w("ablation", "\\begin{tabular}{p{6.3cm}rrrr}\n\\toprule\nVariant & Risk & Vision risk & Mean saved (\\%) & Energy-wtd.\\ (\\%) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_sens():
    d = pd.read_csv("figures/tierA/ablation_sensitivity.csv")
    s = d[(d.study == "A6") & (d.rho == 0.20)].copy()
    s["group"] = s.variant.str.split("=").str[0]
    names = {"eps": "$\\epsilon$ (pp)", "window": "window $w$", "safeguard_frac": "safeguard fraction",
             "flat_tol": "flat tolerance (pp)", "k": "consecutive signals $k$"}
    rows = []
    for g, sub in s.groupby("group", sort=False):
        for _, r in sub.iterrows():
            rows.append(f"{names[g]} = {r.variant.split('=')[1]} & {r.oof_risk:.3f} & {r.mean_saved:.1f} & {r.energy_weighted_saved:.1f} \\\\")
        rows.append("\\midrule")
    w("sensitivity", "\\begin{tabular}{p{4.2cm}rrr}\n\\toprule\nSetting & Risk & Mean saved (\\%) & Energy-wtd.\\ (\\%) \\\\\n\\midrule\n"
      + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")


def tab_paired():
    d = pd.read_csv("figures/tierA/paired_ci.csv")
    d = d[d.metric == "mean_saved"]
    nm = {"Patience (p=f*T, min-delta 0.1)": r"Patience $p{=}fT$, min-$\delta$", "Patience (fixed p)": r"Patience, fixed $p$",
          "LC-PFN + EcoStop gates": "LC-PFN + gates", "Fixed budget fraction": "Fixed budget fraction"}
    rows = []
    for _, r in d.iterrows():
        rows.append("%.2f & vs. %s & %+.1f & [%+.1f, %+.1f] \\\\" % (r.rho, nm[r.EcoStop_minus], r["diff"], r.ci_lo, r.ci_hi))
    NL = chr(10)
    BS = chr(92)
    top = BS + 'begin{tabular}{llrr}' + NL + BS + 'toprule' + NL + '$' + BS + 'rho$ & Comparison (EcoStop minus ...) & $' + BS + 'Delta$ mean saved (pp) & 95' + BS + '% job-bootstrap CI ' + BS + BS + NL + BS + 'midrule' + NL
    w('paired', top + NL.join(rows) + NL + BS + 'bottomrule' + NL + BS + 'end{tabular}' + NL)


def tab_b1():
    d = pd.read_csv("figures/tierA/b1_fresh_test.csv")
    order = ["EcoStop", "LC-PFN + EcoStop gates", "Patience (p=f*T, min-delta 0.1)", "Patience (p=f*T)",
             "Patience (fixed p)", "Fixed budget fraction"]
    NL, BS = chr(10), chr(92)
    lines = []
    for rho in sorted(d.rho.unique()):
        lines.append(BS + "multicolumn{8}{l}{" + BS + "emph{Frozen target $" + BS + "rho=" + ("%.2f" % rho) + "$, $" + BS + "beta=0.05$}}" + BS + BS)
        for m in order:
            r = d[(d.rho == rho) & (d.method == m) & (d.scheme == "pooled")]
            if r.empty:
                continue
            r = r.iloc[0]
            pdm = d[(d.rho == rho) & (d.method == m) & (d.scheme == "per_domain")]
            pdv = "%.1f" % pdm.iloc[0].energy_weighted_saved if not pdm.empty else "--"
            f = lambda x: "--" if pd.isna(x) else "%.3f" % x
            lines.append("%s & %.3f [%.3f, %.3f] & %s & %s & %s & %.1f & %.1f & %s " % (
                NICE.get(m, m), r.risk, r.risk_ci_lo, r.risk_ci_hi, f(r.risk_vision), f(r.risk_nlp), f(r.risk_tabular),
                r.mean_saved, r.energy_weighted_saved, pdv) + BS + BS)
        lines.append(BS + "midrule")
    head = (BS + "begin{tabular}{p{3.6cm}rrrrrrr}" + NL + BS + "toprule" + NL +
            "Rule & Risk [95" + BS + "% CI] & Vision & NLP & Tabular & Mean saved (" + BS + "%) & Energy-wtd. (" + BS + "%) & Per-domain EW (" + BS + "%) " + BS + BS + NL + BS + "midrule" + NL)
    w("b1_fresh", head + NL.join(lines[:-1]) + NL + BS + "bottomrule" + NL + BS + "end{tabular}" + NL)


def tab_b2():
    d = pd.read_csv("figures/tierA/b2_results.csv")
    d = d[d.scheme == "pooled"]
    NL, BS = chr(10), chr(92)
    nm = {"EcoStop": "EcoStop", "Patience (p=f*T, min-delta 0.1)": "Patience, $p=" + BS + "lceil fT" + BS + "rfloor$, min-$" + BS + "delta$",
          "Patience (p=f*T)": "Patience, $p=" + BS + "lceil fT" + BS + "rfloor$", "Patience (fixed p)": "Patience, fixed $p$"}
    lines = []
    for rho in sorted(d.rho.unique()):
        lines.append(BS + "multicolumn{5}{l}{" + BS + "emph{Frozen target $" + BS + "rho=" + ("%.2f" % rho) + "$}}" + BS + BS)
        for r in nm:
            x = d[(d.rho == rho) & (d.rule == r)].iloc[0]
            lines.append("%s & %.3f & %.1f & %.1f & %.1f " % (nm[r], x.risk, x.epochs_saved, x.energy_saved_mean, x.energy_saved_weighted) + BS + BS)
        lines.append(BS + "midrule")
    head = (BS + "begin{tabular}{p{3.8cm}rrrr}" + NL + BS + "toprule" + NL + "Rule & Risk & Epochs saved (" + BS + "%) & Energy saved (" + BS + "%) & Energy-wtd. (" + BS + "%) " + BS + BS + NL + BS + "midrule" + NL)
    w("b2", head + NL.join(lines[:-1]) + NL + BS + "bottomrule" + NL + BS + "end{tabular}" + NL)


if __name__ == '__main__':
    for fn in (tab_oof, tab_ablation, tab_sens, tab_paired, tab_b1, tab_b2):
        try:
            fn()
        except FileNotFoundError as e:
            print('skipped', fn.__name__, e)
