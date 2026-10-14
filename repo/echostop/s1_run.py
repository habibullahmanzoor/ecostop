"""S1 trainer: cooldown branches + recipe-matched cosine control for every (job, seed).
usage: python -m echostop.s1_run cal | test      (test refuses to start without a matching s1_protocol.json)
Strictly sequential (one metered job at a time). Resumable: a run counts as done only if its .meta.json exists."""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")

import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from echostop import s1_registry as R
from echostop.data import get_dataset
from echostop.energy import EnergyMeter
from echostop.models import build_model
from echostop.pilot_cool import FIELDS, one_epoch
from echostop.train import set_seed

OUT = Path("logs/s1")


def job_id(ds, arch, lr):
    return f"{ds}_{arch}_lr{lr}"


def stable_lr(e, lr):
    return lr * min(1.0, (e + 1) / (R.WARM + 1))


def cool_lr(e, lr):
    return lr * (R.C - e - 0.5) / R.C


def cosine_lr(e, lr):
    if e < R.WARM:
        return lr * (e + 1) / (R.WARM + 1)
    return 0.5 * lr * (1 + math.cos(math.pi * (e - R.WARM) / (R.T_COSINE - R.WARM)))


def make(arch, bundle, lr, dev):
    m = build_model(arch, bundle.in_channels, bundle.num_classes).to(dev)
    o = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    return m, o


def run_cool(ds, arch, lr, seed, out: Path):
    rid = f"{job_id(ds, arch, lr)}_seed{seed}_cool"
    if (out / f"{rid}.meta.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    set_seed(seed)
    bundle = get_dataset(ds, seed=seed, batch_size=256)
    model, opt = make(arch, bundle, lr, dev)
    crit = nn.CrossEntropyLoss()
    meter = EnergyMeter(device_index=0, sample_interval_s=0.02)
    meter.start()
    saved, t0 = {}, time.time()
    with open(out / f"{rid}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()

        def log(phase, bt, ep, lr_e, loss, val, test, rec):
            w.writerow({"phase": phase, "branch_t": bt, "epoch": ep, "lr": lr_e, "train_loss": loss, "val_acc": val,
                        "test_acc": test, "energy_joules": rec.energy_joules, "duration_seconds": rec.duration_seconds,
                        "mean_power_watts": rec.mean_power_watts, "t_wall": time.time()})
            f.flush()
            print(f"[{rid}] {phase:6s} t={bt:>2} ep {ep:2d} val {val:6.2f} test {test:6.2f} E {rec.energy_joules:7.1f}J "
                  f"t {rec.duration_seconds:5.1f}s", flush=True)

        for e in range(R.T_STABLE):
            lr_e = stable_lr(e, lr)
            loss, val, test, rec = one_epoch(model, opt, bundle, crit, lr_e, dev, meter, e)
            log("stable", -1, e, lr_e, loss, val, test, rec)
            if (e + 1) in R.CKPTS:
                saved[e + 1] = (copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()))
        for t in R.CKPTS:
            set_seed(seed * 1000 + t)
            m, o = make(arch, bundle, lr, dev)
            m.load_state_dict(saved[t][0]); o.load_state_dict(saved[t][1])
            for e in range(R.C):
                lr_e = cool_lr(e, lr)
                loss, val, test, rec = one_epoch(m, o, bundle, crit, lr_e, dev, meter, e)
                log("branch", t, e, lr_e, loss, val, test, rec)
            del m, o
            torch.cuda.empty_cache()
    meter.stop()
    json.dump({"dataset": ds, "arch": arch, "lr": lr, "seed": seed, "kind": "cool", "gpu": meter.device_name,
               "wall_s": time.time() - t0}, open(out / f"{rid}.meta.json", "w"), indent=2)


def run_cosine(ds, arch, lr, seed, out: Path):
    rid = f"{job_id(ds, arch, lr)}_seed{seed}_cosine"
    if (out / f"{rid}.meta.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    set_seed(seed)
    bundle = get_dataset(ds, seed=seed, batch_size=256)
    model, opt = make(arch, bundle, lr, dev)
    crit = nn.CrossEntropyLoss()
    meter = EnergyMeter(device_index=0, sample_interval_s=0.02)
    meter.start()
    t0 = time.time()
    with open(out / f"{rid}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for e in range(R.T_COSINE):
            lr_e = cosine_lr(e, lr)
            loss, val, test, rec = one_epoch(model, opt, bundle, crit, lr_e, dev, meter, e)
            w.writerow({"phase": "cosine", "branch_t": -1, "epoch": e, "lr": lr_e, "train_loss": loss, "val_acc": val,
                        "test_acc": test, "energy_joules": rec.energy_joules, "duration_seconds": rec.duration_seconds,
                        "mean_power_watts": rec.mean_power_watts, "t_wall": time.time()})
            f.flush()
            print(f"[{rid}] cosine ep {e:2d} val {val:6.2f} test {test:6.2f} E {rec.energy_joules:7.1f}J t {rec.duration_seconds:5.1f}s", flush=True)
    meter.stop()
    json.dump({"dataset": ds, "arch": arch, "lr": lr, "seed": seed, "kind": "cosine", "gpu": meter.device_name,
               "wall_s": time.time() - t0}, open(out / f"{rid}.meta.json", "w"), indent=2)


def main():
    phase = sys.argv[1]
    if phase == "cal":
        jobs, seeds, out = R.CAL_NEW + R.CAL_VARIANT, R.CAL_SEEDS, OUT / "cal"
    elif phase == "test":
        proto = json.load(open("s1_protocol.json"))      # refuses to run without a freeze
        h = hashlib.sha256(open("echostop/s1_registry.py", "rb").read()).hexdigest()
        if h != proto["registry_sha256"]:
            raise SystemExit("s1_registry.py changed after the freeze; test set no longer pre-registered")
        jobs, seeds, out = R.TEST, R.TEST_SEEDS, OUT / "test"
    else:
        raise SystemExit("usage: s1_run cal|test")
    for seed in seeds:
        for ds, arch, lr in jobs:
            try:
                run_cool(ds, arch, lr, seed, out)
                run_cosine(ds, arch, lr, seed, out)
            except Exception as e:
                print(f"FAILED {ds} {arch} lr{lr} seed {seed}: {type(e).__name__}: {e}", flush=True)
    print(f"S1 {phase.upper()} COMPLETE", flush=True)


if __name__ == "__main__":
    main()
