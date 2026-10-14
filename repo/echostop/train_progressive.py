"""Variable-per-epoch-energy setting (plan §16, REQUIRED): progressive
resizing on CIFAR-10. Training resolution ramps 16 -> 20 -> 24 -> 32 px in
four phases across the 40-epoch budget; validation/test stay at native
32x32 throughout so B_t remains comparable across the whole run. This is
the deciding test for RQ0 (plan §2): §6.4/§7.2 predict energy can only
carry decision-relevant information beyond epoch count in a setting like
this one, since the fixed-resolution pilot cells showed the energy oracle
and epoch oracle picking essentially the same stopping point.
"""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from echostop.data import get_cifar10, get_cifar10_train_loader_at_size
from echostop.energy import EnergyMeter
from echostop.models import build_model
from echostop.train import evaluate, set_seed

# (start_epoch_inclusive, end_epoch_exclusive, image_size)
RESIZE_SCHEDULE = [
    (0, 10, 16),
    (10, 20, 20),
    (20, 30, 24),
    (30, 40, 32),
]


def size_for_epoch(epoch: int) -> int:
    for start, end, size in RESIZE_SCHEDULE:
        if start <= epoch < end:
            return size
    return RESIZE_SCHEDULE[-1][2]


def train_progressive_run(
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    out_dir: Path,
    device_index: int = 0,
    model_name: str = "resnet9",
) -> Path:
    set_seed(seed)
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

    # validation/test loaders stay fixed at native resolution (32x32)
    bundle = get_cifar10(seed=seed, batch_size=batch_size)
    model = build_model(model_name, bundle.in_channels, bundle.num_classes).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    run_id = f"cifar10prog_{model_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_id}.csv"
    meta_path = out_dir / f"{run_id}.meta.json"

    meter = EnergyMeter(device_index=device_index)
    meter.start()

    fieldnames = [
        "epoch", "image_size", "train_loss", "val_acc", "is_new_best",
        "test_acc_at_checkpoint", "energy_joules", "duration_seconds",
        "mean_power_watts", "cumulative_energy_joules", "cumulative_duration_seconds",
    ]

    best_val_acc = -1.0
    cumulative_energy = 0.0
    cumulative_duration = 0.0
    current_size = None
    train_loader = None

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(epochs):
            size_now = size_for_epoch(epoch)
            if size_now != current_size:
                train_loader = get_cifar10_train_loader_at_size(
                    size_now, seed=seed, batch_size=batch_size
                )
                current_size = size_now
                print(f"[{run_id}] epoch {epoch}: switching train resolution -> {size_now}px")

            meter.begin_epoch(epoch)

            model.train()
            running_loss, n_batches = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1
            scheduler.step()

            val_acc = evaluate(model, bundle.val_loader, device)
            rec = meter.end_epoch()

            is_new_best = val_acc > best_val_acc
            test_acc_at_checkpoint = ""
            if is_new_best:
                best_val_acc = val_acc
                test_acc_at_checkpoint = evaluate(model, bundle.test_loader, device)

            cumulative_energy += rec.energy_joules
            cumulative_duration += rec.duration_seconds

            row = {
                "epoch": epoch,
                "image_size": size_now,
                "train_loss": running_loss / max(n_batches, 1),
                "val_acc": val_acc,
                "is_new_best": int(is_new_best),
                "test_acc_at_checkpoint": test_acc_at_checkpoint,
                "energy_joules": rec.energy_joules,
                "duration_seconds": rec.duration_seconds,
                "mean_power_watts": rec.mean_power_watts,
                "cumulative_energy_joules": cumulative_energy,
                "cumulative_duration_seconds": cumulative_duration,
            }
            writer.writerow(row)
            f.flush()
            print(
                f"[{run_id}] epoch {epoch:3d} | sz {size_now:2d}px | loss {row['train_loss']:.4f} | "
                f"val_acc {val_acc:6.2f}% | best {'*' if is_new_best else ' '} | "
                f"E {rec.energy_joules:7.1f} J | P {rec.mean_power_watts:6.1f} W | "
                f"t {rec.duration_seconds:5.1f} s"
            )

    meter.stop()
    with open(meta_path, "w") as f:
        json.dump(
            {
                "run": run_id, "seed": seed, "epochs": epochs, "batch_size": batch_size,
                "lr": lr, "weight_decay": weight_decay, "gpu": meter.device_name,
                "resize_schedule": RESIZE_SCHEDULE,
            },
            f, indent=2,
        )
    return log_path


def main() -> None:
    for seed in [0, 1, 2]:
        train_progressive_run(
            seed=seed, epochs=40, batch_size=256, lr=0.1, weight_decay=5e-4,
            out_dir=Path("logs/pilot_progressive"),
        )


if __name__ == "__main__":
    main()
