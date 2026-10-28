"""B5: the schedule-confound experiment.

One architecture (ResNet-18), one dataset (CIFAR-10), one recipe -- exactly the
corpus recipe -- with only the learning-rate schedule varied: cosine, step
decay, one-cycle. This is the experiment that separates "the schedule family
matters" from "ResNet-18 vs VGG matters": in the fresh test set (B1) the
discontinuous families also used different architectures, so the two were
confounded.

Logs the same schema as echostop/train.py so that calibration.precompute_score_trace
and calibration.simulate_from_trace replay these runs unchanged.

Frozen protocol: b5_protocol.json (written before the first run; see b5_freeze.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echostop.data import get_dataset
from echostop.energy import EnergyMeter
from echostop.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def make_scheduler(name: str, optimizer, epochs: int, lr: float):
    """The only thing that differs between arms. All three step once per epoch."""
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        milestones = [int(round(0.5 * epochs)), int(round(0.75 * epochs))]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=epochs, pct_start=0.3,
            div_factor=10.0, final_div_factor=100.0)
    raise ValueError(f"unknown schedule {name}")


def train_run(schedule: str, dataset: str, model_name: str, seed: int, epochs: int,
              batch_size: int, lr: float, weight_decay: float, out_dir: Path,
              device_index: int = 0) -> Path:
    set_seed(seed)
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    bundle = get_dataset(dataset, seed=seed, batch_size=batch_size)
    model = build_model(model_name, bundle.in_channels, bundle.num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=weight_decay, nesterov=True)
    scheduler = make_scheduler(schedule, optimizer, epochs, lr)
    criterion = nn.CrossEntropyLoss()

    run_id = f"{schedule}_{dataset}_{model_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_id}.csv"

    meter = EnergyMeter(device_index=device_index)
    meter.start()

    fields = ["epoch", "train_loss", "val_acc", "is_new_best", "test_acc_at_checkpoint",
              "energy_joules", "duration_seconds", "mean_power_watts",
              "cumulative_energy_joules", "cumulative_duration_seconds", "lr"]
    best, cum_e, cum_t = -1.0, 0.0, 0.0

    with open(log_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for epoch in range(epochs):
            meter.begin_epoch(epoch)
            model.train()
            running, nb = 0.0, 0
            for x, y in bundle.train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                running += loss.item()
                nb += 1
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            val_acc = evaluate(model, bundle.val_loader, device)
            rec = meter.end_epoch()
            is_best = val_acc > best
            test_at = ""
            if is_best:
                best = val_acc
                test_at = evaluate(model, bundle.test_loader, device)
            cum_e += rec.energy_joules
            cum_t += rec.duration_seconds
            w.writerow({"epoch": epoch, "train_loss": running / max(nb, 1), "val_acc": val_acc,
                        "is_new_best": int(is_best), "test_acc_at_checkpoint": test_at,
                        "energy_joules": rec.energy_joules, "duration_seconds": rec.duration_seconds,
                        "mean_power_watts": rec.mean_power_watts,
                        "cumulative_energy_joules": cum_e,
                        "cumulative_duration_seconds": cum_t, "lr": current_lr})
            fh.flush()
            print(f"[{run_id}] ep {epoch:3d} lr {current_lr:.4f} val {val_acc:6.2f} "
                  f"E {rec.energy_joules:7.1f}J P {rec.mean_power_watts:5.1f}W", flush=True)

    meter.stop()
    (out_dir / f"{run_id}.meta.json").write_text(json.dumps(
        {"schedule": schedule, "dataset": dataset, "model": model_name, "seed": seed,
         "epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
         "gpu": meter.device_name, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2), encoding="utf-8")
    return log_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", required=True, choices=["cosine", "step", "onecycle"])
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--out-dir", default="logs/b5")
    a = p.parse_args()
    print(train_run(a.schedule, a.dataset, a.model, a.seed, a.epochs, a.batch_size,
                    a.lr, a.weight_decay, Path(a.out_dir)))
