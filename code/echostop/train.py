"""Training loop for the pilot study (plan §40, Phase 1 = §12).

Trains to completion (no early stopping — the pilot needs full curves),
logging per epoch: train loss, validation accuracy `Q_t`, whether this is a
new running-best checkpoint, that checkpoint's test-set accuracy (so
`A_t = Q_{j_t}^test`, plan §11.0.1, can be reconstructed by forward-filling
in analysis), plus per-epoch energy `e_t`, runtime `r_t`, and mean power
`P_bar_t` (plan §2 RQ0 — the incremental quantities, not just cumulative).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train_one_run(
    dataset_name: str,
    model_name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    out_dir: Path,
    device_index: int = 0,
) -> Path:
    set_seed(seed)
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

    bundle = get_dataset(dataset_name, seed=seed, batch_size=batch_size)
    model = build_model(model_name, bundle.in_channels, bundle.num_classes).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    run_id = f"{dataset_name}_{model_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_id}.csv"
    meta_path = out_dir / f"{run_id}.meta.json"

    meter = EnergyMeter(device_index=device_index)
    meter.start()

    fieldnames = [
        "epoch", "train_loss", "val_acc", "is_new_best", "test_acc_at_checkpoint",
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
                f"[{run_id}] epoch {epoch:3d} | loss {row['train_loss']:.4f} | "
                f"val_acc {val_acc:6.2f}% | best {'*' if is_new_best else ' '} | "
                f"E {rec.energy_joules:7.1f} J | P {rec.mean_power_watts:6.1f} W | "
                f"t {rec.duration_seconds:5.1f} s"
            )

    meter.stop()

    with open(meta_path, "w") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "model": model_name,
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "gpu": meter.device_name,
            },
            f,
            indent=2,
        )

    return log_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["fashion_mnist", "cifar10"])
    p.add_argument("--model", required=True, choices=["resnet9", "resnet18"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--out-dir", type=str, default="logs")
    p.add_argument("--device-index", type=int, default=0)
    args = p.parse_args()

    log_path = train_one_run(
        dataset_name=args.dataset,
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        out_dir=Path(args.out_dir),
        device_index=args.device_index,
    )
    print(f"done -> {log_path}")


if __name__ == "__main__":
    main()
