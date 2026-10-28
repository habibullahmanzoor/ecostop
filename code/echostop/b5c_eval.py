"""B5c: the confound test, at matched architecture.

For each of the four datasets that carry a B1 step-decay job, compare the step-decay arm with the
cosine arm under otherwise identical settings. Three of the cosine arms were trained for this test
(logs/b5c, protocol b5c_protocol.json); the fourth, fashion_mnist, already existed in the development
corpus with the same loader, seed, architecture and hyperparameters, and is labelled as such.

The endpoint is the violation rate at the threshold frozen on the development corpus. The declared-drop
gate is inert on cosine by construction, so only the ungated numbers are reported here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echostop.calibration import audited_loss, precompute_score_trace, simulate_from_trace

ROOT = Path(__file__).resolve().parents[1]
GAMMAS = {"rho_0.20": 0.46, "rho_0.15": 0.14}
NEW = ("dtd", "oxford_pet", "semeion")


def runs_for(arm: str, ds: str) -> list[Path]:
    if arm == "step":
        return sorted((ROOT / "logs/b1/vision").glob(f"{ds}_resnet18_step_seed*.csv"))
    if ds == "fashion_mnist":
        return sorted((ROOT / "logs/pilot").glob(f"{ds}_resnet18_seed*.csv"))
    return sorted((ROOT / "logs/b5c").glob(f"cosine_{ds}_resnet18_seed*.csv"))


def score(path: Path, gamma: float) -> dict | None:
    df = pd.read_csv(path)
    if len(df) < 20:
        return None                      # incomplete run
    df["B_t"] = np.maximum.accumulate(df["val_acc"].values)
    if "cumulative_energy_joules" not in df.columns or df["cumulative_energy_joules"].isna().all():
        return None
    tr = precompute_score_trace(df, 0.5)
    stop = simulate_from_trace(tr, gamma, len(df))
    cum = df["cumulative_energy_joules"].values
    return {"stop": int(stop), "loss": audited_loss(df, stop, 0.5),
            "saved": float(cum[-1] - cum[stop]), "total": float(cum[-1])}


def main() -> None:
    rows = []
    for ds in ("fashion_mnist",) + NEW:
        for arm in ("step", "cosine"):
            files = runs_for(arm, ds)
            if not files:
                print(f"  (no {arm} runs for {ds})")
                continue
            for f in files:
                for gname, gamma in GAMMAS.items():
                    r = score(f, gamma)
                    if r is None:
                        continue
                    rows.append(dict(dataset=ds, arm=arm, seed=f.stem.split("seed")[-1],
                                     gamma=gname, **r))
    if not rows:
        sys.exit("no complete runs yet")

    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "figures/tierA/b5c_confound.csv", index=False)

    for gname in GAMMAS:
        sub = r[r.gamma == gname]
        counts = sub.groupby(["dataset", "arm"]).agg(
            runs=("loss", "size"), violations=("loss", "sum"), risk=("loss", "mean"),
            saved=("saved", "sum"), total=("total", "sum"))
        counts["EW_%"] = 100 * counts["saved"] / counts["total"]
        print(f"\n=== {gname} ===")
        print(counts[["runs", "violations", "risk", "EW_%"]].round(3).to_string())

    g = "rho_0.20"
    sub = r[r.gamma == g]
    wide = sub.pivot_table(index="dataset", columns="arm", values="loss", aggfunc="mean")
    if {"step", "cosine"} <= set(wide.columns):
        wide["cosine minus step"] = wide["cosine"] - wide["step"]
        print(f"\n=== paired comparison at {g} ===")
        print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
