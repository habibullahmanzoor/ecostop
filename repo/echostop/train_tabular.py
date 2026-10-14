"""Training loop for the tabular calibration-corpus jobs (plan §16).
Same CSV schema as train.py so these plug into the same analysis/
calibration code as the vision runs.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from echostop.energy import EnergyMeter
from echostop.models import TabularMLP
from echostop.tabular_data import get_tabular_dataset
from echostop.train import evaluate, set_seed


def train_tabular_run(
    dataset_name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    out_dir: Path,
    device_index: int = 0,
    source: str = "sklearn",
    run_id_override: str | None = None,
) -> Path:
    set_seed(seed)
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

    if source == "sklearn":
        bundle = get_tabular_dataset(dataset_name, seed=seed, batch_size=batch_size)
    elif source == "openml":
        from echostop.openml_data import get_openml_dataset
        bundle = get_openml_dataset(dataset_name, seed=seed, batch_size=batch_size)
    else:
        raise ValueError(f"unknown source: {source!r}")
    model = TabularMLP(bundle.in_features, bundle.num_classes).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    run_id = run_id_override or f"tabular_{dataset_name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_id}.csv"
    meta_path = out_dir / f"{run_id}.meta.json"

    # Tighter polling interval than the vision pipeline's default 20ms --
    # tabular epochs on tiny MLPs/tiny data can complete in a few ms, faster
    # than the default interval can sample even once.
    meter = EnergyMeter(device_index=device_index, sample_interval_s=0.002)
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

        MIN_EPOCH_DURATION_S = 0.025  # see note below

        for epoch in range(epochs):
            meter.begin_epoch(epoch)
            epoch_start = time.monotonic()

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

            # Tabular epochs on tiny datasets (iris, wine) can complete in
            # under 1ms -- faster than the energy meter's polling thread can
            # reliably catch even once, giving spurious energy=0 readings
            # (verified empirically: digits at ~34 batches/epoch had zero
            # gaps; iris/wine at 2-4 batches/epoch had gaps in the majority
            # of epochs, and consolidating to full-batch made it WORSE, not
            # better, since fewer dispatch calls meant even less wall-clock
            # time). Pad to a minimum measurable duration rather than report
            # noise-dominated near-zero readings; this adds idle-GPU wait
            # time to the measurement window, which is a bounded, documented
            # distortion, not a silent one.
            elapsed = time.monotonic() - epoch_start
            if elapsed < MIN_EPOCH_DURATION_S:
                time.sleep(MIN_EPOCH_DURATION_S - elapsed)

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

    meter.stop()
    with open(meta_path, "w") as f:
        json.dump(
            {"dataset": dataset_name, "model": "tabular_mlp", "seed": seed, "epochs": epochs,
             "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
             "in_features": bundle.in_features, "num_classes": bundle.num_classes,
             "gpu": meter.device_name},
            f, indent=2,
        )
    return log_path


def main() -> None:
    datasets = ["digits", "breast_cancer", "wine", "iris"]
    seeds = list(range(8))  # 8 seeds x 4 datasets = 32 cheap jobs
    out_dir = Path("logs/calibration_tabular")
    t0 = time.time()
    total = len(datasets) * len(seeds)
    done = 0
    for name in datasets:
        for seed in seeds:
            done += 1
            log_path = train_tabular_run(
                dataset_name=name, seed=seed, epochs=60, batch_size=32,
                lr=0.05, weight_decay=1e-4, out_dir=out_dir,
            )
            print(f"[{done}/{total}] {log_path} (elapsed {time.time()-t0:.1f}s)")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
