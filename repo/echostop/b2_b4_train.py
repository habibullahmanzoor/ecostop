"""B2 (variable-cost regime: layer freeze/thaw) and B4 (energy-measurement validation).
Run AFTER b1_train finishes (sequential GPU use keeps NVML energy clean).

B2: ResNet-9, 40 epochs cosine, SGD (corpus recipe), but the backbone (prep, layer1, res1, layer2)
    is frozen for epochs [0, 12): backward stops at layer3, so per-epoch energy steps UP at thaw.
    6 cheap datasets x 3 seeds -> logs/b2/.  Energy and epoch count then disagree by construction.
B4: (i) idle power for 60 s, (ii) NVML sampling-interval statistics, (iii) 5 identical repeats of
    one job (semeion/resnet9 seed 0, full recipe) to give run-to-run energy variation -> logs/b4/.

usage: python -m echostop.b2_b4_train [b2|b4|all]
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from echostop.b1_train import done, run_loop
from echostop.data import get_dataset
from echostop.energy import EnergyMeter
from echostop.models import build_model
from echostop.train import evaluate, set_seed

B2_DATASETS = ["fashion_mnist", "kmnist", "mnist", "semeion", "dtd", "oxford_pet"]
B2_FREEZE_EPOCHS = 12
EPOCHS = 40


def set_backbone_trainable(model, trainable: bool):
    for name in ("prep", "layer1", "res1", "layer2"):
        for p in getattr(model, name).parameters():
            p.requires_grad_(trainable)


def train_b2(dataset, seed, out=Path("logs/b2"), tag="freezethaw", epochs=EPOCHS, freeze=B2_FREEZE_EPOCHS):
    run_id = f"{dataset}_resnet9{tag}_seed{seed}"
    if done(out / f"{run_id}.csv", epochs):
        return
    set_seed(seed)
    dev = torch.device("cuda")
    b = get_dataset(dataset, seed=seed, batch_size=256)
    model = build_model("resnet9", b.in_channels, b.num_classes).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    def train_epoch(ep):
        set_backbone_trainable(model, ep >= freeze)
        model.train()
        tot, n = 0.0, 0
        for x, y in b.train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item(); n += 1
        sch.step()
        return tot / max(n, 1)

    run_loop(run_id, out, epochs, train_epoch, lambda: evaluate(model, b.val_loader, dev),
             lambda: evaluate(model, b.test_loader, dev),
             {"dataset": dataset, "arch": "resnet9", "regime": tag, "freeze_epochs": freeze, "seed": seed})


def b4_idle_and_sampling(out=Path("logs/b4")):
    out.mkdir(parents=True, exist_ok=True)
    m = EnergyMeter(device_index=0, sample_interval_s=0.02)
    m.start()
    idle = m.idle_power_watts(60.0)
    time.sleep(0.1)
    ts = np.array([t for t, _ in m._samples]) if m._samples else np.array([])
    m.stop()
    # separate sampling probe: raw NVML call rate and the power sensor's update granularity
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    stamps, vals = [], []
    t_end = time.monotonic() + 10
    while time.monotonic() < t_end:
        stamps.append(time.monotonic()); vals.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
        time.sleep(0.002)
    pynvml.nvmlShutdown()
    stamps, vals = np.array(stamps), np.array(vals)
    changes = np.where(np.diff(vals) != 0)[0]
    update_dt = np.diff(stamps[changes]) if len(changes) > 1 else np.array([np.nan])
    json.dump({"idle_power_w_60s": idle, "poll_calls_per_s": len(stamps) / 10.0,
               "sensor_value_change_interval_s_median": float(np.nanmedian(update_dt)),
               "sensor_value_change_interval_s_p90": float(np.nanpercentile(update_dt, 90))},
              open(out / "idle_and_sampling.json", "w"), indent=2)
    print(open(out / "idle_and_sampling.json").read())


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("b4", "all"):
        b4_idle_and_sampling()
        for rep in range(5):  # identical repeats, same seed => same training; spread is measurement + system noise
            train_b2("semeion", 0, out=Path("logs/b4"), tag=f"_repeat{rep}", epochs=EPOCHS, freeze=0)
    if which in ("b2", "all"):
        for seed in (0, 1, 2):
            for ds in B2_DATASETS:
                try:
                    train_b2(ds, seed)
                except Exception as e:
                    print(f"FAILED b2 {ds}: {type(e).__name__}: {e}", flush=True)
    print("B2/B4 complete", flush=True)


if __name__ == "__main__":
    main()
