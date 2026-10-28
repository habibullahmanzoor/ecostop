"""Figure for B5: the mechanism, from the runs themselves.

Two panels. (a) the declared learning-rate trajectories of the three arms -- what a
schedule-declared rule can see at epoch 0 and a curve-only rule cannot. (b) validation
accuracy per arm, mean over seeds with the seed range shaded, with each arm's scheduled
event marked; the plateau that precedes each event is the window the theorem is about.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/fig_b5_confound.pdf"
COLOURS = {"cosine": "#1f4e79", "step": "#c0392b", "onecycle": "#6c3483"}
LABELS = {"cosine": "cosine (no declared event)", "step": "step decay (drops at 20, 30)",
          "onecycle": "one-cycle (peak at 12)"}


def main() -> None:
    proto = json.loads((ROOT / "b5_protocol.json").read_text(encoding="utf-8"))
    events = {s: a["scheduled_events"] for s, a in proto["arms"].items()}

    fig, (ax_lr, ax_acc) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    for sched in ("cosine", "step", "onecycle"):
        runs = sorted((ROOT / "logs/b5").glob(f"{sched}_*.csv"))
        if not runs:
            continue
        accs, lrs = [], []
        for f in runs:
            df = pd.read_csv(f)
            accs.append(df["val_acc"].values)
            lrs.append(df["lr"].values)
        n = min(len(a) for a in accs)
        acc = np.vstack([a[:n] for a in accs])
        lr = np.vstack([a[:n] for a in lrs])
        e = np.arange(n)
        c = COLOURS[sched]
        ax_lr.plot(e, lr.mean(0), color=c, lw=1.4, label=LABELS[sched])
        ax_acc.plot(e, acc.mean(0), color=c, lw=1.4, label=LABELS[sched])
        if len(acc) > 1:
            ax_acc.fill_between(e, acc.min(0), acc.max(0), color=c, alpha=0.18, lw=0)
        for td in events[sched]:
            for ax in (ax_lr, ax_acc):
                ax.axvline(td, color=c, ls=":", lw=1.0, alpha=0.9)

    ax_lr.set_xlabel("epoch")
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_yscale("log")
    ax_lr.set_title("(a) the declared schedule", fontsize=9)
    ax_lr.legend(fontsize=6.5, frameon=False, loc="lower left")

    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("validation accuracy (%)")
    ax_acc.set_title("(b) what the rule observes", fontsize=9)
    ax_acc.legend(fontsize=6.5, frameon=False, loc="lower right")

    for ax in (ax_lr, ax_acc):
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
