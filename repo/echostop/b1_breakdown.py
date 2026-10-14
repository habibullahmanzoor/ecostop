"""B1 breakdowns of the frozen-protocol result: by schedule family, with/without non-learning jobs,
and the test-noise floor (violation rate of the hindsight oracle). Writes figures/tierA/b1_*.csv
and tables/b1_schedule.tex."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E
from echostop.b2_b4_analysis import oracle_stop
from echostop.calibration import forward_fill_audit, precompute_score_trace

FAMILY = {"mobilenetv3": "cosine + warm-up", "vit_tiny": "cosine + warm-up",
          "resnet18_step": "step decay", "vgg_small": "one-cycle"}
RULES = [("EcoStop", "EcoStop", {0.15: "g=0.14", 0.20: "g=0.46"}),
         ("Patience, p=fT, min-delta", "Patience (p=f*T, min-delta 0.1)", {0.15: "f=0.25", 0.20: "f=0.175"})]
NL, BS = chr(10), chr(92)


def main():
    jobs, dom = E.load_b1()
    V = {k: v for k, v in jobs.items() if dom[k] == "vision"}
    tr = {k: [precompute_score_trace(df, 0.5) for df in dfs] for k, dfs in V.items()}
    fam = {k: FAMILY[k.split("_", 1)[1] if k.split("_", 1)[1] in FAMILY else [a for a in FAMILY if k.endswith(a)][0]] for k in V}
    nonlearn = [k for k, dfs in V.items() if np.mean([df["val_acc"].max() for df in dfs]) < 3.0]
    rows, summ = [], {"non_learning_jobs": nonlearn}
    for rho in (0.15, 0.20):
        for nice, family, lab in [(a, b, c[rho]) for a, b, c in RULES]:
            fn = E.stop_fn(family, lab)
            l, s, ew = E.score(V, tr, fn)
            names = list(V)
            for f in sorted(set(fam.values())):
                idx = [i for i, k in enumerate(names) if fam[k] == f]
                rows.append({"rho": rho, "rule": nice, "family": f, "jobs": len(idx),
                             "risk": l[idx].mean(), "mean_saved": s[idx].mean()})
            keep = [i for i, k in enumerate(names) if k not in nonlearn]
            rows.append({"rho": rho, "rule": nice, "family": "all vision", "jobs": len(names), "risk": l.mean(), "mean_saved": s.mean()})
            rows.append({"rho": rho, "rule": nice, "family": "all vision, excl. non-learning", "jobs": len(keep),
                         "risk": l[keep].mean(), "mean_saved": s[keep].mean()})
    # test-noise floor: hindsight oracle violation rate
    orc = {}
    for k, dfs in V.items():
        v = []
        for df in dfs:
            a = forward_fill_audit(df)
            v.append(float(a[-1] - a[oracle_stop(df)] > 0.5))
        orc[k] = np.mean(v)
    floor_by_fam = {f: float(np.mean([orc[k] for k in V if fam[k] == f])) for f in sorted(set(fam.values()))}
    summ["oracle_violation_all_vision"] = float(np.mean(list(orc.values())))
    summ["oracle_violation_by_family"] = floor_by_fam
    d = pd.DataFrame(rows)
    d.to_csv("figures/tierA/b1_schedule_breakdown.csv", index=False)
    pd.Series(summ).to_json("figures/tierA/b1_vision_summary.json")
    pd.set_option("display.width", 200)
    print(d.round(3).to_string(index=False)); print(summ)

    # LaTeX table at rho = 0.20
    sub = d[(d.rho == 0.20)]
    lines = []
    for f in ["cosine + warm-up", "step decay", "one-cycle", "all vision", "all vision, excl. non-learning"]:
        cells = []
        for nice, _, _ in RULES:
            r = sub[(sub.rule == nice) & (sub.family == f)].iloc[0]
            cells.append("%.2f & %.1f" % (r.risk, r.mean_saved))
        fl = floor_by_fam.get(f, summ["oracle_violation_all_vision"] if f == "all vision" else float("nan"))
        n = int(sub[(sub.rule == RULES[0][0]) & (sub.family == f)].iloc[0].jobs)
        lines.append("%s & %d & %s & %s " % (f, n, " & ".join(cells), "--" if np.isnan(fl) else "%.2f" % fl) + BS + BS)
    head = (BS + "begin{tabular}{p{3.6cm}rrrrrr}" + NL + BS + "toprule" + NL +
            "Schedule family & Jobs & EcoStop risk & EcoStop saved (" + BS + "%) & Patience risk & Patience saved (" + BS + "%) & Oracle risk " + BS + BS + NL + BS + "midrule" + NL)
    open("tables/b1_schedule.tex", "w", encoding="utf-8").write(head + NL.join(lines) + NL + BS + "bottomrule" + NL + BS + "end{tabular}" + NL)


if __name__ == "__main__":
    main()
