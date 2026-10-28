"""The gate, evaluated on all three B1 schedule families at once.

The gate is the one Corollary 3 licenses: do not stop while a *declared discontinuity* still lies
ahead. Discontinuities are read from the recipe (the registry), not discovered in the data:

  resnet18_step   MultiStepLR milestones [15, 27] on a 40-epoch budget -> a drop at each milestone
  vgg_small       OneCycleLR over 30 epochs, peak at pct_start=0.3 -> the LR falls every epoch after
                  epoch 9, so no single step is a declared discontinuity
  mobilenetv3/vit_tiny  cosine with warm-up -> no discontinuity

So the gate should fire only for step decay. That is the point: it is free on continuous schedules
(inert, so no cost) and inert on one-cycle too, where the decay is gradual and the theorem's route
does not apply -- the family-level certificate is the instrument there.

Writes figures/tierA/gate_families.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echostop import b1_eval as E
from echostop.calibration import audited_loss, precompute_score_trace
from echostop.methods import schedule_relative_safeguard

ROOT = Path(__file__).resolve().parents[1]
GAMMAS = (0.46, 0.14)

# declared discontinuities, from the registry (b1_registry.py), not from the logs
DECLARED = {"resnet18_step": [15, 27], "vgg_small": [], "mobilenetv3": [], "vit_tiny": []}


def family_of(job: str) -> str:
    if "resnet18_step" in job:
        return "step"
    if "vgg_small" in job:
        return "onecycle"
    return "continuous"


def arch_of(job: str) -> str:
    for a in DECLARED:
        if a in job:
            return a
    return "unknown"


def gated_stop(trace, gamma, n, events, gated: bool, k: int = 2) -> int:
    gate = schedule_relative_safeguard(n)
    last_event = max(events) if events else 0
    consec = 0
    for row in trace:
        t = row["t"]
        blocked = gated and t < last_event          # a declared drop still lies ahead
        fire = (row["s_t"] is not None and row["s_t"] < gamma
                and row["epochs_flat"] >= gate and not blocked)
        if fire:
            consec += 1
            if consec >= k:
                return t
        else:
            consec = 0
    return n - 1


def main() -> None:
    jobs, dom = E.load_b1()
    rows = []
    for job, traces in sorted(jobs.items()):
        fam = family_of(job)
        events = DECLARED.get(arch_of(job), [])
        for seed, df in enumerate(traces):
            d = df.copy()
            d["B_t"] = np.maximum.accumulate(d["val_acc"].values)
            n = len(d)
            tr = precompute_score_trace(d, 0.5)
            cum = d["cumulative_energy_joules"].values
            total = float(cum[-1])
            for gamma in GAMMAS:
                for gated in (False, True):
                    stop = gated_stop(tr, gamma, n, events, gated)
                    rows.append(dict(job=job, seed=seed, family=fam, gamma=gamma, gated=gated,
                                     stop=int(stop), epochs=n, loss=audited_loss(d, stop, 0.5),
                                     saved=float(total - cum[stop]), total=total))
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "figures/tierA/gate_families.csv", index=False)

    g = (r.groupby(["family", "gamma", "gated"])
           .agg(runs=("loss", "size"), violations=("loss", "sum"), risk=("loss", "mean"),
                saved=("saved", "sum"), total=("total", "sum")))
    g["EW_%"] = 100 * g["saved"] / g["total"]
    print(g[["runs", "violations", "risk", "EW_%"]].round(3).to_string())
    print("\nnote: 'gated' rows for cosine/one-cycle are inert by construction (no declared drop)")


if __name__ == "__main__":
    main()

