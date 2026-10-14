"""Free confirmation: B2 (layer freeze/thaw, 6 datasets x 3 seeds) was NOT part of the 177-job development
set. Calibrate EcoStop k=1 and calibrated patience on the 177 dev jobs, apply to B2 unchanged."""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import phase1_gate as G
from echostop import phase1_prep as P
from echostop.calibration import forward_fill_audit
from echostop.tierA import select

M = G.build_matrices()          # dev matrices (also leaves FAMILIES = {eco, pat} with sd gates)
eco_cfgs, pat_cfgs = P.FAMILIES["eco"], P.FAMILIES["pat"]


def idx_sorted(fam, key, **kw):
    return sorted(G.idx_of(fam, **kw), key=key)


EC = idx_sorted("eco", lambda i: eco_cfgs[i]["g"], k=1, h="sd0")
EC2 = idx_sorted("eco", lambda i: eco_cfgs[i]["g"], k=2, h="sd0")
PT = idx_sorted("pat", lambda i: -pat_cfgs[i]["f"], h="sd0")

runs = {}
for p in sorted(glob.glob("logs/b2/*_resnet9freezethaw_seed*.csv")):
    if not os.path.exists(p.replace(".csv", ".meta.json")):
        continue
    df = pd.read_csv(p)
    key = re.sub(r"_resnet9freezethaw_seed\d+\.csv$", "", os.path.basename(p))
    runs.setdefault(key, []).append({"job": key, "domain": "vision", "family": "cosine", "warmup": 0, "seed": len(runs.get(key, [])),
                                     "T": len(df), "q": df["val_acc"].values.astype(float), "b": df["val_acc"].cummax().values,
                                     "a": forward_fill_audit(df), "e": df["cumulative_energy_joules"].values.astype(float),
                                     "loss_tr": df["train_loss"].values.astype(float)})
flat = [r for v in runs.values() for r in v]
flat = [P._trace(r) for r in flat]
for r in flat:
    G.sudden_blocks(r)
by = {}
for r in flat:
    by.setdefault(r["job"], []).append(r)
print("B2 jobs:", len(by), "runs:", len(flat))


def evaluate(fam, cfg):
    loss, saved, es, et = [], [], 0.0, 0.0
    for k, rs in by.items():
        l, s = [], []
        for r in rs:
            t = P.stop_score(r, cfg, "s") if fam == "eco" else P.stop_patience(r, cfg)
            l.append(float(r["a"][-1] - r["a"][t] > P.EPS)); s.append(100 * (1 - r["e"][t] / r["e"][-1]))
            es += r["e"][-1] - r["e"][t]; et += r["e"][-1]
        loss.append(np.mean(l)); saved.append(np.mean(s))
    return np.mean(loss), np.mean(saved), 100 * es / et


loss_e, _, _, _ = M["eco"]
loss_p, _, _, _ = M["pat"]
print(f"{'rule':34s} {'rho':>5s} {'param':>14s} {'B2 risk':>8s} {'B2 saved':>9s} {'B2 EW':>7s}")
for rho in (0.15, 0.20):
    for name, fam, idx, loss_dev, cfgs in (("Patience p=fT, min-delta", "pat", PT, loss_p, pat_cfgs),
                                           ("EcoStop paper (k=2)", "eco", EC2, loss_e, eco_cfgs),
                                           ("EcoStop k=1", "eco", EC, loss_e, eco_cfgs)):
        i = select(loss_dev[idx], rho, 0.05)
        if i is None:
            print(f"{name:34s} {rho:5.2f} {'not certified':>14s}")
            continue
        c = cfgs[idx[i]]
        r, s, ew = evaluate(fam, c)
        prm = f"g={c['g']}" if fam == "eco" else f"f={c['f']}"
        print(f"{name:34s} {rho:5.2f} {prm:>14s} {r:8.3f} {s:9.1f} {ew:7.1f}")
