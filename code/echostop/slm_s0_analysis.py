"""SLM S0 pilot: gate analysis exactly as defined in SLM_PLAN.md section 5 (written while the pilot was running).

Evaluation points are logged every 1/8 epoch (24 per run). Granularity g in {1, 2, 4, 8} keeps every g-th point (1/8, 1/4,
1/2, 1 epoch). For granularity g and a run:
  A_t   test accuracy of the validation-selected checkpoint among the g-points <= t (first maximum), A_T at the last point
  loss  1[A_T - A_t > 0.5 pp]; oracle stop = earliest g-point with loss 0
  E_t   training energy up to t + validation-evaluation energy of the g-points <= t (test evaluation = audit, excluded)
  base  the epoch-level full run (g = 8): all training + 3 validation evaluations  -> net saving = 1 - E_t / base
Gates:
  (a) deployed duration of one epoch-level run <= 15 min (training + 3 validation evaluations)
  (b) validation-evaluation energy <= 10% of the run's deployed energy at g = 2 (1/4 epoch)
  (c) oracle stop at g = 2 occurs at <= 60% of the training steps (median over runs)
  (d) energy-weighted net oracle saving at g = 2 exceeds g = 8 by >= 5 pp
Informational: practical patience (counted in evaluation points) and Hugging Face-style patience at epoch level, with
violation rate and energy-weighted net saving, per granularity."""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

EPS = 0.5
G = (1, 2, 4, 8)
DIR = sys.argv[1] if len(sys.argv) > 1 else "logs/slm/s0"


def load():
    runs = {}
    for p in sorted(glob.glob(f"{DIR}/*.csv")):
        if not os.path.exists(p.replace(".csv", ".meta.json")):
            continue
        d = pd.read_csv(p)
        m = re.match(r"^(.*)_([a-z0-9]+)_seed(\d+)\.csv$", os.path.basename(p))
        runs[(m.group(1), int(m.group(3)))] = d
    return runs


def per_gran(d, g):
    idx = [j for j in range(len(d)) if (j + 1) % g == 0]
    sub = d.iloc[idx].reset_index(drop=True)
    val, test = sub.val_acc.values, sub.test_acc.values
    best = np.zeros(len(sub), int)
    cur = 0
    for i in range(len(sub)):
        if val[i] > val[cur]:
            cur = i
        best[i] = cur
    A = test[best]
    cum_train = d.e_train.cumsum().values[idx]
    cum_val = np.cumsum(sub.e_val.values)
    return dict(A=A, E=cum_train + cum_val, val_only=cum_val, steps=sub.step.values, val=val, dur=d.d_train.sum() + sub.d_val.sum())


def main():
    runs = load()
    print(f"{len(runs)} completed runs\n")
    rows, base = [], {}
    for key, d in runs.items():
        p8 = per_gran(d, 8)
        base[key] = p8["E"][-1]
    out = {g: [] for g in G}
    for key, d in runs.items():
        for g in G:
            p = per_gran(d, g)
            A, E = p["A"], p["E"]
            t = int(np.argmax(A[-1] - A <= EPS))
            out[g].append(dict(key=key, frac=p["steps"][t] / p["steps"][-1], net=1 - E[t] / base[key],
                               evalfrac=p["val_only"][-1] / E[-1], base=base[key], saved=base[key] - E[t], dur=p["dur"]))
    print("=== oracle (hindsight) stop, net of validation-evaluation energy, vs the epoch-level full run ===")
    summ = {}
    for g in G:
        o = pd.DataFrame(out[g])
        summ[g] = dict(ew=100 * o.saved.sum() / o.base.sum(), med_frac=o.frac.median(), evalfrac=100 * o.evalfrac.mean(), dur=o.dur.mean() / 60)
        print(f"  g={g} ({8 // g if g else 0}: every {g}/8 epoch): net EW saving {summ[g]['ew']:5.1f}% | oracle stop at "
              f"{100 * summ[g]['med_frac']:4.0f}% of steps (median) | validation-eval share of deployed energy {summ[g]['evalfrac']:4.1f}% "
              f"| mean deployed duration {summ[g]['dur']:4.1f} min")
    gate_a = summ[8]["dur"] <= 15
    gate_b = summ[2]["evalfrac"] <= 10
    gate_c = summ[2]["med_frac"] <= 0.6
    gate_d = summ[2]["ew"] - summ[8]["ew"] >= 5
    print("\n=== GATES ===")
    print(f"  (a) epoch-level deployed run <= 15 min        : {summ[8]['dur']:.1f} min -> {'PASS' if gate_a else 'FAIL'}")
    print(f"  (b) eval energy <= 10% of run at 1/4 epoch    : {summ[2]['evalfrac']:.1f}% -> {'PASS' if gate_b else 'FAIL'}")
    print(f"  (c) oracle stops at <= 60% of steps (median)  : {100 * summ[2]['med_frac']:.0f}% -> {'PASS' if gate_c else 'FAIL'}")
    print(f"  (d) net oracle saving g=2 minus g=8 >= 5 pp   : {summ[2]['ew'] - summ[8]['ew']:+.1f} pp -> {'PASS' if gate_d else 'FAIL'}")
    print("  overall:", "ALL PASS" if all((gate_a, gate_b, gate_c, gate_d)) else "NOT ALL PASS")

    print("\n=== informational: practical patience, counted in evaluation points (violation rate, EW net saving vs epoch-level full run) ===")
    for g, plist in ((8, (1, 2)), (4, (1, 2, 3)), (2, (2, 3, 4, 6)), (1, (4, 6, 8, 12))):
        for pat in plist:
            v, sv, bs = [], 0.0, 0.0
            for key, d in runs.items():
                p = per_gran(d, g)
                val, best, since, stop = p["val"], -np.inf, 0, len(p["val"]) - 1
                for i, x in enumerate(val):
                    if x > best:
                        best, since = x, 0
                    else:
                        since += 1
                    if since >= pat:
                        stop = i
                        break
                v.append(p["A"][-1] - p["A"][stop] > EPS)
                sv += base[key] - p["E"][stop]
                bs += base[key]
            print(f"  g={g} patience {pat} points: violation rate {np.mean(v):.2f} | net EW saving {100 * sv / bs:5.1f}%")
    pd.set_option("display.width", 200)
    per = pd.DataFrame([{"task": k[0], "seed": k[1], **{f"net_g{g}": 100 * next(x for x in out[g] if x['key'] == k)["net"] for g in G},
                         "oracle_frac_g2": next(x for x in out[2] if x['key'] == k)["frac"]} for k in runs])
    print("\nper run (oracle net saving %):\n" + per.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
