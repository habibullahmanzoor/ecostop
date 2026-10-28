import sys, glob, os, csv, pandas as pd, numpy as np
sys.path.insert(0, ".")
from echostop.calibration import precompute_score_trace, audited_loss
from echostop.methods import schedule_relative_safeguard

EVENTS = [15, 27]; LAST = max(EVENTS)

def stop_with(trace, gamma, n, policy, k=2):
    gate = schedule_relative_safeguard(n); consec = 0
    for row in trace:
        t = row["t"]
        blk = (policy == "window" and any(e-2 <= t <= e for e in EVENTS))
        blk = blk or (policy == "global" and t < LAST)
        fire = (row["s_t"] is not None and row["s_t"] < gamma
                and row["epochs_flat"] >= gate and not blk)
        if fire:
            consec += 1
            if consec >= k: return t
        else: consec = 0
    return n - 1

rows = []
for f in sorted(glob.glob("logs/b1/vision/*resnet18_step*.csv")):
    df = pd.read_csv(f); df["B_t"] = np.maximum.accumulate(df["val_acc"].values)
    n = len(df); tr = precompute_score_trace(df, 0.5)
    cum = df["cumulative_energy_joules"].values; tot = float(cum[-1])
    for gamma in (0.46, 0.14):
        for policy in ("none", "window", "global"):
            st = stop_with(tr, gamma, n, policy)
            rows.append(dict(run=os.path.basename(f), gamma=gamma, policy=policy, stop=st,
                             loss=audited_loss(df, st, 0.5), saved=float(tot-cum[st]), total=tot))
r = pd.DataFrame(rows); r.to_csv("figures/tierA/gate_schedule.csv", index=False)
g = r.groupby(["gamma","policy"]).agg(runs=("loss","size"), violations=("loss","sum"),
                                      risk=("loss","mean"), saved=("saved","sum"), total=("total","sum"))
g["EW_%"] = 100*g["saved"]/g["total"]
print(g[["runs","violations","risk","EW_%"]].to_string())
