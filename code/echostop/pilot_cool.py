"""P0 pilot (protocols/DEVIATIONS_AND_DECISIONS.md): certified early cooldown.

Per (dataset, seed): ResNet-9, SGD (momentum 0.9, nesterov, wd 5e-4), batch 256, STABLE lr 0.05 (2-epoch warm-up) for T_STABLE
epochs, with a checkpoint (model + optimizer) at epochs CKPTS. After the stable run, from every checkpoint a
COOLDOWN branch of C epochs is trained (lr decays linearly to ~0). The branch from epoch 32 is the reference
(stable 32 + cooldown 8 = 40 epochs, the same budget as the corpus' cosine runs).

Everything is metered the same way as the corpus: NVML power, per-epoch trapezoidal energy, validation accuracy
inside the metered window, test accuracy after it (unmetered, logged EVERY epoch for the counterfactual audit).
Branches run sequentially after the stable run, never concurrently.
Resumable: a (dataset, seed) counts as done only if its .meta.json exists.
usage: python -m echostop.pilot_cool [smoke]
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")

import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from echostop.data import get_dataset
from echostop.energy import EnergyMeter
from echostop.models import build_model
from echostop.train import evaluate, set_seed

LR = 0.05
WARM = 2   # epochs of linear warm-up at the start of the stable phase
T_STABLE = 32
CKPTS = [8, 12, 16, 20, 24, 28, 32]
C = 8
DATASETS = ["fashion_mnist", "kmnist", "mnist", "cifar10", "gtsrb", "imagenette"]
SEEDS = [0, 1]
FIELDS = ["phase", "branch_t", "epoch", "lr", "train_loss", "val_acc", "test_acc", "energy_joules",
          "duration_seconds", "mean_power_watts", "t_wall"]


def stable_lr(e: int) -> float:
    return LR * min(1.0, (e + 1) / (WARM + 1))


def cool_lr(e: int, c: int = C) -> float:
    return LR * (c - e - 0.5) / c      # linear decay, midpoint rule: never exactly zero, last epoch ~LR/(2c)


def one_epoch(model, opt, bundle, crit, lr, dev, meter, ep):
    for g in opt.param_groups:
        g["lr"] = lr
    meter.begin_epoch(ep)
    model.train()
    tot, n = 0.0, 0
    for x, y in bundle.train_loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        tot += loss.item(); n += 1
    val = evaluate(model, bundle.val_loader, dev)
    rec = meter.end_epoch()
    test = evaluate(model, bundle.test_loader, dev)          # unmetered
    return tot / max(n, 1), val, test, rec


def run_pair(dataset, seed, out_dir: Path, t_stable=T_STABLE, ckpts=CKPTS, c=C):
    run_id = f"{dataset}_seed{seed}"
    if (out_dir / f"{run_id}.meta.json").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    set_seed(seed)
    bundle = get_dataset(dataset, seed=seed, batch_size=256)
    model = build_model("resnet9", bundle.in_channels, bundle.num_classes).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    crit = nn.CrossEntropyLoss()
    meter = EnergyMeter(device_index=0, sample_interval_s=0.02)
    meter.start()
    saved = {}
    t0 = time.time()
    with open(out_dir / f"{run_id}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()

        def log(phase, bt, ep, lr, loss, val, test, rec):
            w.writerow({"phase": phase, "branch_t": bt, "epoch": ep, "lr": lr, "train_loss": loss, "val_acc": val,
                        "test_acc": test, "energy_joules": rec.energy_joules, "duration_seconds": rec.duration_seconds,
                        "mean_power_watts": rec.mean_power_watts, "t_wall": time.time()})
            f.flush()
            print(f"[{run_id}] {phase:6s} t={bt:>2} ep {ep:2d} lr {lr:.4f} val {val:6.2f} test {test:6.2f} "
                  f"E {rec.energy_joules:7.1f}J t {rec.duration_seconds:5.1f}s", flush=True)

        # ---- stable-LR run with checkpoints
        for e in range(t_stable):
            lr_e = stable_lr(e)
            loss, val, test, rec = one_epoch(model, opt, bundle, crit, lr_e, dev, meter, e)
            log("stable", -1, e, lr_e, loss, val, test, rec)
            if (e + 1) in ckpts:
                saved[e + 1] = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()))
        # ---- cooldown branches (sequential)
        for t in ckpts:
            set_seed(seed * 1000 + t)
            m = build_model("resnet9", bundle.in_channels, bundle.num_classes).to(dev)
            o = torch.optim.SGD(m.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
            m.load_state_dict(saved[t][0]); o.load_state_dict(saved[t][1])
            for e in range(c):
                lr = cool_lr(e, c)
                loss, val, test, rec = one_epoch(m, o, bundle, crit, lr, dev, meter, e)
                log("branch", t, e, lr, loss, val, test, rec)
            del m, o
            torch.cuda.empty_cache()
    meter.stop()
    json.dump({"dataset": dataset, "seed": seed, "lr": LR, "t_stable": t_stable, "ckpts": ckpts, "c": c,
               "gpu": meter.device_name, "wall_s": time.time() - t0}, open(out_dir / f"{run_id}.meta.json", "w"), indent=2)


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    if smoke:
        run_pair("fashion_mnist", 0, Path("logs/pilot_cool_smoke"), t_stable=4, ckpts=[2, 4], c=2)
        print("SMOKE OK")
        return
    out = Path("logs/pilot_cool")
    for seed in SEEDS:
        for ds in DATASETS:
            try:
                run_pair(ds, seed, out)
            except Exception as e:
                print(f"FAILED {ds} seed {seed}: {type(e).__name__}: {e}", flush=True)
    print("PILOT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
