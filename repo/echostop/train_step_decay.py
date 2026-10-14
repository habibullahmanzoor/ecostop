"""Bounded test (plan §24 A11): does a step-decay schedule produce
exploitable plateaus, unlike cosine annealing where every online method
(including plain patience) essentially never fires early on vision data
(session finding -- see figure_method_comparison.png and the vision
pilot's near-universal "stops at the last epoch" result)?

Step decay creates a qualitatively different curve shape from cosine: a
jump right after each LR drop, then a genuine flat plateau until the next
drop -- exactly the kind of signal patience-based and predictor-based
stopping need. Cosine anneals continuously, so small improvements keep
appearing sporadically all the way to epoch 39 with no genuine plateau to
detect. One cell (CIFAR-10 + ResNet-9), 3 seeds, directly comparable to
the existing cosine-decay pilot cell for the same architecture.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from echostop.data import get_dataset
from echostop.energy import EnergyMeter
from echostop.models import build_model
from echostop.train import evaluate, set_seed

MILESTONES = [15, 27]  # epoch boundaries: 0.1 -> 0.01 -> 0.001, for a 40-epoch budget
GAMMA = 0.1


def train_step_decay_run(
    dataset_name: str, model_name: str, seed: int, epochs: int,
    batch_size: int, lr: float, weight_decay: float, out_dir: Path,
    device_index: int = 0,
) -> Path:
    set_seed(seed)
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

    bundle = get_dataset(dataset_name, seed=seed, batch_size=batch_size)
    model = build_model(model_name, bundle.in_channels, bundle.num_classes).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
    criterion = nn.CrossEntropyLoss()

    run_id = f"stepdecay_{dataset_name}_{model_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_id}.csv"
    meta_path = out_dir / f"{run_id}.meta.json"

    meter = EnergyMeter(device_index=device_index)
    meter.start()

    fieldnames = [
        "epoch", "lr", "train_loss", "val_acc", "is_new_best", "test_acc_at_checkpoint",
        "energy_joules", "duration_seconds", "mean_power_watts",
        "cumulative_energy_joules", "cumulative_duration_seconds",
    ]
    best_val_acc = -1.0
    cumulative_energy = 0.0
    cumulative_duration = 0.0

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(epochs):
            meter.begin_epoch(epoch)
            model.train()
            running_loss, n_batches = 0.0, 0
            for x, y in bundle.train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1
            current_lr = optimizer.param_groups[0]["lr"]
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
                "epoch": epoch, "lr": current_lr,
                "train_loss": running_loss / max(n_batches, 1), "val_acc": val_acc,
                "is_new_best": int(is_new_best), "test_acc_at_checkpoint": test_acc_at_checkpoint,
                "energy_joules": rec.energy_joules, "duration_seconds": rec.duration_seconds,
                "mean_power_watts": rec.mean_power_watts,
                "cumulative_energy_joules": cumulative_energy,
                "cumulative_duration_seconds": cumulative_duration,
            }
            writer.writerow(row)
            f.flush()
            print(f"[{run_id}] epoch {epoch:3d} | lr {current_lr:.4f} | val_acc {val_acc:6.2f}% | "
                  f"best {'*' if is_new_best else ' '}")

    meter.stop()
    with open(meta_path, "w") as f:
        json.dump({"dataset": dataset_name, "model": model_name, "seed": seed, "epochs": epochs,
                    "schedule": "step_decay", "milestones": MILESTONES, "gamma": GAMMA,
                    "gpu": meter.device_name}, f, indent=2)
    return log_path


def main() -> None:
    t0 = time.time()
    for seed in [0, 1, 2]:
        train_step_decay_run(
            dataset_name="cifar10", model_name="resnet9", seed=seed, epochs=40,
            batch_size=256, lr=0.1, weight_decay=5e-4, out_dir=Path("logs/step_decay_test"),
        )
        print(f"seed {seed} done, elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
