"""Evaluate B5 against its frozen protocol (b5_protocol.json).

For each run: replay the paper's EcoStop rule at the two frozen thresholds, compute
the audited loss and the energy saved, with the gate off and on. The gate blocks a
stop at epoch t when a *declared* schedule event t_d satisfies t_d - h <= t <= t_d,
with h fixed in the protocol before the runs.

The gated replay is checked against calibration.simulate_from_trace when the gate is
off, so a discrepancy in the re-implementation would be caught rather than reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echostop.calibration import audited_loss, precompute_score_trace, simulate_from_trace
from echostop.methods import schedule_relative_safeguard

ROOT = Path(__file__).resolve().parents[1]
PROTO = json.loads((ROOT / "b5_protocol.json").read_text(encoding="utf-8"))
EPS = PROTO["evaluation"]["tolerance_eps_pp"]
GAMMAS = PROTO["evaluation"]["gamma_frozen"]
K = PROTO["evaluation"]["patience_k"]
H = PROTO["evaluation"]["gate"]["h_epochs"]
EVENTS = {s: a["scheduled_events"] for s, a in PROTO["arms"].items()}


def gated_stop(trace, gamma, n_epochs, schedule, h=H, k=K):
    """Same fixed-sequence replay as simulate_from_trace, with the declared-event gate."""
    gate = schedule_relative_safeguard(n_epochs)
    events = EVENTS[schedule]
    consecutive = 0
    for row in trace:
        t = row["t"]
        blocked = any(td - h <= t <= td for td in events)
        fire = (row["s_t"] is not None and row["s_t"] < gamma
                and row["epochs_flat"] >= gate and not blocked)
        if fire:
            consecutive += 1
            if consecutive >= k:
                return t
        else:
            consecutive = 0
    return n_epochs - 1


def main() -> None:
    files = sorted((ROOT / "logs/b5").glob("*.csv"))
    # a run is complete only once the trainer has written its meta (it does so at the end)
    missing = [f.name for f in files if not f.with_suffix(".meta.json").exists()]
    files = [f for f in files if f.with_suffix(".meta.json").exists()]
    if missing:
        print(f"skipping {len(missing)} incomplete run(s): {', '.join(missing)}")
    if not files:
        sys.exit("no complete B5 runs found")
    rows = []
    for f in files:
        meta = json.loads(f.with_suffix(".meta.json").read_text(encoding="utf-8"))
        sched, seed = meta["schedule"], meta["seed"]
        df = pd.read_csv(f)
        df["B_t"] = np.maximum.accumulate(df["val_acc"].values)
        n = len(df)
        tr = precompute_score_trace(df, EPS)
        cum_e = df["cumulative_energy_joules"].values
        total_e = cum_e[-1]
        for gname, gamma in GAMMAS.items():
            for gated in (False, True):
                stop = (gated_stop(tr, gamma, n, sched) if gated
                        else simulate_from_trace(tr, gamma, n))
                # the gated path must reduce to the frozen rule when nothing is blocked
                if (not gated) or (sched == "cosine"):
                    ref = simulate_from_trace(tr, gamma, n)
                    if stop != ref:
                        sys.exit(f"gate changed a schedule with no events: {f.name} {gname}")
                rows.append({"schedule": sched, "seed": seed, "gamma": gname,
                             "gated": gated, "stop": int(stop), "epochs": n,
                             "loss": audited_loss(df, stop, EPS),
                             "saved_j": float(total_e - cum_e[stop]),
                             "total_j": float(total_e)})
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "figures/tierA/b5_confound.csv", index=False)

    print(f"runs: {r[['schedule', 'seed']].drop_duplicates().shape[0]}")
    for gamma in r.gamma.unique():
        print(f"\n=== gamma {gamma} ({'rho=0.15' if gamma == 'rho_0.15' else 'rho=0.20'}) ===")
        for gated in (False, True):
            sub = r[(r.gamma == gamma) & (r.gated == gated)]
            if sub.empty:
                continue
            agg = (sub.groupby("schedule")
                      .agg(runs=("loss", "size"), violations=("loss", "sum"),
                           risk=("loss", "mean"), saved=("saved_j", "sum"),
                           total=("total_j", "sum")))
            agg["EW_%"] = 100 * agg["saved"] / agg["total"]
            agg["gate"] = "on" if gated else "off"
            print(agg[["runs", "violations", "risk", "EW_%", "gate"]].to_string())


if __name__ == "__main__":
    main()
