"""Export every metered run of the EcoStop study to one long CSV (ecostop_bench/runs.csv.gz).

Sets: c1 (82-job confirmatory fine-tuning study), dev (142-job development corpus), dev_extra (step-decay CIFAR-10 job), fresh (pre-registered fresh test),
regime_freezethaw (layer freeze/thaw), repeat (5 identical repeats for energy repeatability).
Run from the project root: python ecostop_bench/export_data.py"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from echostop import phase1_prep as P
from echostop.calibration import forward_fill_audit

rows = []


def add(set_name, job, domain, family, seed, df, lr=None, extra=None):
    T = len(df)
    a = forward_fill_audit(df)
    q = df["val_acc"].values
    e = df["cumulative_energy_joules"].values
    d = df["cumulative_duration_seconds"].values if "cumulative_duration_seconds" in df else np.full(T, np.nan)
    tl = df["train_loss"].values if "train_loss" in df else np.full(T, np.nan)
    for t in range(T):
        rows.append({"set": set_name, "job": job, "domain": domain, "schedule_family": family, "seed": seed, "horizon_T": T,
                     "epoch": t, "val_acc": q[t], "test_acc_at_best_val": a[t], "train_loss": tl[t],
                     "lr_multiplier": (lr[t] if lr is not None else np.nan), "cumulative_energy_J": e[t],
                     "cumulative_time_s": d[t]})


# development + fresh (already assembled with learning-rate schedules in phase1_prep)
for r in P.load_runs():
    df = pd.DataFrame({"val_acc": r["q"], "cumulative_energy_joules": r["e"], "train_loss": r["loss_tr"],
                       "test_acc_at_checkpoint": np.nan})
    # forward-filled audit is already in r['a']; rebuild frame accordingly
    T = r["T"]
    q, e = r["q"], r["e"]
    for t in range(T):
        rows.append({"set": "dev" if r["set"] == "dev" else "fresh", "job": r["job"].split(":", 1)[1], "domain": r["domain"],
                     "schedule_family": r["family"], "seed": r["seed"], "horizon_T": T, "epoch": t, "val_acc": q[t],
                     "test_acc_at_best_val": r["a"][t], "train_loss": r["loss_tr"][t], "lr_multiplier": r["lr"][t],
                     "cumulative_energy_J": e[t], "cumulative_time_s": np.nan})

for p in sorted(glob.glob("logs/step_decay_test/*.csv")):
    df = pd.read_csv(p)
    seed = int(re.search(r"seed(\d+)", p).group(1))
    add("dev_extra", "cifar10_resnet9_stepdecay", "vision", "step", seed, df, lr=P.lr_curve("step", len(df)))
for p in sorted(glob.glob("logs/b2/*_resnet9freezethaw_seed*.csv")):
    if not os.path.exists(p.replace(".csv", ".meta.json")):
        continue
    df = pd.read_csv(p)
    job = re.sub(r"_resnet9freezethaw_seed\d+\.csv$", "", os.path.basename(p)) + "_resnet9_freezethaw"
    df["test_acc_at_checkpoint"] = df["test_acc_at_checkpoint"]
    add("regime_freezethaw", job, "vision", "cosine_freeze12", int(re.search(r"seed(\d+)", p).group(1)), df, lr=P.lr_curve("cosine", len(df)))
for p in sorted(glob.glob("logs/b4/semeion_resnet9_repeat*_seed0.csv")):
    if not os.path.exists(p.replace(".csv", ".meta.json")):
        continue
    df = pd.read_csv(p)
    add("repeat", "semeion_resnet9_" + re.search(r"repeat\d+", p).group(0), "vision", "cosine", 0, df, lr=P.lr_curve("cosine", len(df)))

# C1: confirmatory short-schedule fine-tuning study (valid runs only; quarantined runs are NOT exported)
for p in sorted(glob.glob("logs/c1/nlp/*.csv")):
    if not os.path.exists(p.replace(".csv", ".meta.json")):
        continue
    m = re.match(r"^(.*)_([AB])_seed(\d+)\.csv$", os.path.basename(p))
    ds, arm, seed = m.groups()
    df = pd.read_csv(p)
    df["test_acc_at_checkpoint"] = pd.to_numeric(df["test_acc_at_checkpoint"], errors="coerce")
    add("c1", f"{ds}_{'electra_small' if arm == 'A' else 'bert_l6'}", "nlp", "linear_decay", int(seed), df)

out = pd.DataFrame(rows)
out["val_acc"] = out["val_acc"].round(4); out["test_acc_at_best_val"] = out["test_acc_at_best_val"].round(4)
out["cumulative_energy_J"] = out["cumulative_energy_J"].round(3)
os.makedirs("ecostop_bench", exist_ok=True)
out.to_csv("ecostop_bench/runs.csv.gz", index=False, compression="gzip")
n_runs = out.groupby(["set", "job", "seed"]).ngroups
print("rows:", len(out), "| runs:", n_runs, "| jobs:", out.groupby(["set", "job"]).ngroups)
print(out.groupby("set").apply(lambda d: pd.Series({"jobs": d.job.nunique(), "runs": d.groupby(["job", "seed"]).ngroups})).to_string())
print("size KB:", os.path.getsize("ecostop_bench/runs.csv.gz") // 1024)
