"""Recipe-matched cosine control for the cooldown pilot (added 2026-10-09 after the seed-0 gate showed that the
corpus' cosine recipe (lr 0.1, no warm-up) is weaker than the pilot's (lr 0.05 + 2-epoch warm-up) on harder datasets).

Same model, optimiser, batch size, warm-up and peak LR as pilot_cool; a standard 40-epoch cosine decay from the
peak to ~0. Same logging as the pilot (val and test accuracy and energy every epoch).
usage: python -m echostop.pilot_cosine"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")

import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from echostop.data import get_dataset
from echostop.energy import EnergyMeter
from echostop.models import build_model
from echostop.pilot_cool import DATASETS, FIELDS, LR, SEEDS, WARM, one_epoch
from echostop.train import set_seed

T = 40
OUT = Path("logs/pilot_cosine")


def cosine_lr(e: int) -> float:
    if e < WARM:
        return LR * (e + 1) / (WARM + 1)
    return 0.5 * LR * (1 + math.cos(math.pi * (e - WARM) / (T - WARM)))


def run(dataset, seed):
    run_id = f"{dataset}_seed{seed}"
    if (OUT / f"{run_id}.meta.json").exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    set_seed(seed)
    bundle = get_dataset(dataset, seed=seed, batch_size=256)
    model = build_model("resnet9", bundle.in_channels, bundle.num_classes).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    crit = nn.CrossEntropyLoss()
    meter = EnergyMeter(device_index=0, sample_interval_s=0.02)
    meter.start()
    t0 = time.time()
    with open(OUT / f"{run_id}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for e in range(T):
            lr = cosine_lr(e)
            loss, val, test, rec = one_epoch(model, opt, bundle, crit, lr, dev, meter, e)
            w.writerow({"phase": "cosine", "branch_t": -1, "epoch": e, "lr": lr, "train_loss": loss, "val_acc": val,
                        "test_acc": test, "energy_joules": rec.energy_joules, "duration_seconds": rec.duration_seconds,
                        "mean_power_watts": rec.mean_power_watts, "t_wall": time.time()})
            f.flush()
            print(f"[{run_id}] cosine ep {e:2d} lr {lr:.4f} val {val:6.2f} test {test:6.2f} E {rec.energy_joules:7.1f}J "
                  f"t {rec.duration_seconds:5.1f}s", flush=True)
    meter.stop()
    json.dump({"dataset": dataset, "seed": seed, "lr": LR, "warm": WARM, "T": T, "gpu": meter.device_name,
               "wall_s": time.time() - t0}, open(OUT / f"{run_id}.meta.json", "w"), indent=2)


def main():
    for seed in SEEDS:
        for ds in DATASETS:
            try:
                run(ds, seed)
            except Exception as e:
                print(f"FAILED {ds} seed {seed}: {type(e).__name__}: {e}", flush=True)
    print("COSINE CONTROL COMPLETE", flush=True)


if __name__ == "__main__":
    main()
