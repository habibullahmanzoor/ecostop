"""Pre-registered P0 pilot gate (protocols/DEVIATIONS_AND_DECISIONS.md). Written BEFORE any pilot data exists.

For every finished (dataset, seed):
  reference  = branch from stable epoch 32 (stable 32 + cooldown 8); outcome = test accuracy at the val-best epoch of the branch
  branch(t)  = same for the branch from checkpoint t; energy(t) = stable energy through t epochs + that branch's energy
  GATE 1: |mean(reference - cosine corpus final accuracy)| <= 0.5 pp (reference is credible; same datasets, model, seeds).
  GATE 2: for >= 70% of job-seeds some trigger t <= 24 gives branch(t) >= reference - 0.5 pp (there is energy to save).
  GATE 3: at matched empirical violation rate, EcoStop-triggered cooldown saves >= 10 pp more energy than stop-only
          EcoStop on the corpus' cosine runs of the same datasets/seeds (descriptive; n is small).
GATE 3b (added 2026-10-09 after viewing ONE job-seed, fashion_mnist seed 0; it makes the test harder): the adaptive trigger must also
          beat a NON-ADAPTIVE fixed trigger epoch (the budgeted-training baseline inside the same framework) by >= 5 pp at violation 0.1.
All gates must pass to continue to S1. usage: python -m echostop.pilot_cool_analysis [seed0|all]"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ecostop_bench"))
import bench
from echostop import phase1_prep as P
from echostop.pilot_cool import CKPTS, DATASETS

EPS = 0.5
GAMMAS = [round(x, 2) for x in np.linspace(0.02, 0.98, 25)]


def load_pair(path):
    df = pd.read_csv(path)
    st = df[df.phase == "stable"].sort_values("epoch")
    out = {"val": st.val_acc.values, "test": st.test_acc.values, "e": st.energy_joules.values}
    br = {}
    for t, g in df[df.phase == "branch"].groupby("branch_t"):
        g = g.sort_values("epoch")
        v = g.val_acc.values
        k = len(v) - 1 - int(np.argmax(v[::-1]))       # val-best, ties -> latest
        br[int(t)] = {"acc": g.test_acc.values[k], "e": g.energy_joules.sum()}
    out["branch"] = br
    return out


def analyse(seeds):
    pairs = {}
    for p in sorted(glob.glob("logs/pilot_cool/*_seed*.csv")):
        if not os.path.exists(p.replace(".csv", ".meta.json")):
            continue
        ds, sd = os.path.basename(p)[:-4].rsplit("_seed", 1)
        if int(sd) in seeds:
            pairs[(ds, int(sd))] = load_pair(p)
    if not pairs:
        print("no finished job-seeds yet"); return None
    print(f"finished job-seeds: {len(pairs)}  {sorted(pairs)}")
    corpus_all = {(r["job"].split(":", 1)[1], r["seed"]): r for r in P.load_runs() if r["set"] == "dev"}
    corpus = {}
    n_matched = 0
    for (ds, sd) in pairs:
        pc = f"logs/pilot_cosine/{ds}_seed{sd}.csv"
        if os.path.exists(pc.replace(".csv", ".meta.json")):
            c = pd.read_csv(pc).sort_values("epoch")
            q = c.val_acc.values.astype(float); tst = c.test_acc.values.astype(float)
            a = np.empty(len(q)); best, cur = -1.0, np.nan
            for i in range(len(q)):
                if q[i] >= best:
                    best, cur = q[i], tst[i]
                a[i] = cur
            corpus[(f"{ds}_resnet9", sd)] = {"T": len(q), "q": q, "b": np.maximum.accumulate(q), "a": a,
                                             "e": np.cumsum(c.energy_joules.values)}
            n_matched += 1
        elif (f"{ds}_resnet9", sd) in corpus_all:
            corpus[(f"{ds}_resnet9", sd)] = corpus_all[(f"{ds}_resnet9", sd)]
    print(f"cosine comparator: {n_matched} recipe-matched control runs, {len(corpus) - n_matched} corpus-recipe runs "
          "(corpus recipe: lr 0.1, no warm-up; pilot recipe: lr 0.05 + warm-up)")
    # ---------- per job-seed quantities
    rows = []
    for (ds, sd), d in pairs.items():
        ref = d["branch"][32]
        e_stable = np.cumsum(d["e"])
        e_ref = e_stable[31] + ref["e"]
        cos = corpus.get((f"{ds}_resnet9", sd))
        row = {"dataset": ds, "seed": sd, "ref_acc": ref["acc"], "cosine_acc": (cos["a"][-1] if cos is not None else np.nan),
               "e_ref": e_ref}
        ok = [t for t in CKPTS if t <= 24 and d["branch"][t]["acc"] >= ref["acc"] - EPS]
        row["feasible_trigger"] = bool(ok)
        row["earliest_feasible"] = min(ok) if ok else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))
    g1 = float((df.ref_acc - df.cosine_acc).mean())
    g1_by = df.assign(d=df.ref_acc - df.cosine_acc).groupby("dataset").d.mean().round(2).to_dict()
    g2 = float(df.feasible_trigger.mean())
    # ---------- gate 3: cooldown triggered by the EcoStop score vs stop-only EcoStop on the cosine corpus runs
    cool = {}
    for gm in GAMMAS:
        v, sv = [], []
        for (ds, sd), d in pairs.items():
            ref = d["branch"][32]
            run = {"T": 32, "q": d["val"], "b": np.maximum.accumulate(d["val"])}
            s = bench.ecostop_scores(run)
            trig = next((t for t in CKPTS[:-1] if s[t - 1] == s[t - 1] and s[t - 1] < gm), 32)
            acc = ref["acc"] if trig == 32 else d["branch"][trig]["acc"]
            # COST IN EPOCHS (energy readings of this pilot are contaminated by a neighbouring GPU job):
            # cooldown run = trig stable epochs + 8 cooldown epochs; reference = 32 + 8
            v.append(float(acc < ref["acc"] - EPS)); sv.append(100 * (1 - (trig + 8) / 40.0))
        cool[gm] = (np.mean(v), np.mean(sv))
    stop = {}
    for gm in GAMMAS:
        v, sv = [], []
        for (ds, sd) in pairs:
            cos = corpus.get((f"{ds}_resnet9", sd))
            if cos is None:
                continue
            run = {"T": cos["T"], "q": cos["q"], "b": cos["b"], "a": cos["a"], "e": cos["e"]}
            t = bench.ecostop(run, gm, k=1)
            v.append(float(run["a"][-1] - run["a"][t] > EPS)); sv.append(100 * (1 - (t + 1) / run["T"]))
        stop[gm] = (np.mean(v), np.mean(sv))

    # non-adaptive baseline: trigger the cooldown at the same epoch t for every job-seed
    fixed = {}
    for t in CKPTS:
        v, sv = [], []
        for (ds, sd), d in pairs.items():
            ref = d["branch"][32]
            acc = ref["acc"] if t == 32 else d["branch"][t]["acc"]
            v.append(float(acc < ref["acc"] - EPS)); sv.append(100 * (1 - (t + 8) / 40.0))
        fixed[t] = (np.mean(v), np.mean(sv))

    def at(curve, viol):
        # best compute saving achievable at an empirical violation rate <= viol (no interpolation: n is small)
        ys = [c[1] for c in curve.values() if c[0] <= viol + 1e-9] + [0.0]   # 0.0: the rule that never stops is always available
        return float(max(ys)) if ys else float("nan")
    print("\nGATE 1: mean(reference - cosine) =", round(g1, 2), "pp  by dataset:", g1_by)
    print("GATE 2: fraction of job-seeds with a feasible trigger t<=24 =", round(g2, 2))
    print("GATE 3: compute saved (% of epochs; energy not used, pilot energy is contaminated) at matched violation rate")
    for viol in (0.0, 0.1, 0.2):
        c, s = at(cool, viol), at(stop, viol)
        print(f"   violation {viol:.1f}:  cooldown {c:6.1f}   stop-only {s:6.1f}   diff {c - s:+6.1f} pp")
    d10 = at(cool, 0.1) - at(stop, 0.1)
    print("GATE 3b: adaptive trigger vs NON-ADAPTIVE fixed trigger epoch (budgeted-training baseline, same framework):")
    print("   fixed trigger t -> (violation rate, compute saved %):", {t: (round(v[0], 2), round(v[1], 1)) for t, v in fixed.items()})
    for viol in (0.0, 0.1, 0.2):
        c, f = at(cool, viol), at(fixed, viol)
        print(f"   violation {viol:.1f}:  adaptive {c:6.1f}   fixed-trigger {f:6.1f}   diff {c - f:+6.1f} pp")
    d3b = at(cool, 0.1) - at(fixed, 0.1)
    # Gate 1 with the recipe-matched control (rule fixed 2026-10-09 ~18:10, BEFORE any control result existed):
    # one-sided, mean(reference - matched cosine) >= -0.5 pp (the cooldown reference must not be worse than standard training;
    # being better is harmless to the claim). Without the control it stays two-sided |.| <= 0.5.
    g1_ok = (g1 >= -0.5) if n_matched == len(pairs) else (abs(g1) <= 0.5)
    verdict = (g1_ok, g2 >= 0.70, d10 >= 10.0, d3b >= 5.0)
    print("\nPASS flags (gate1, gate2, gate3, gate3b @ violation 0.1):", verdict, "->", "CONTINUE" if all(verdict) else "STOP")
    return verdict


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    analyse({0} if which == "seed0" else {0, 1})
