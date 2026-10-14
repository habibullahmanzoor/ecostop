"""Training loop for the NLP leg (plan section 16/section 41): DistilBERT
fine-tuning on SST-2, rotten_tomatoes, or TREC, energy-logged with the same
CSV schema as the vision/tabular legs (epoch, val_acc, test_acc_at_checkpoint,
cumulative_energy_joules, ...) so it plugs directly into
calibration.py/methods.py/evaluate_*.py without any changes there --
this is the whole point of testing whether EcoStop generalizes to a new
domain rather than being vision-specific. Session addition: the original
SST-2-only leg was n=1 for the entire NLP modality; rotten_tomatoes and
TREC (already scoped in plan section 16's calibration-corpus pool) extend
that to n=3 datasets of deliberately different size/task shape.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from echostop.energy import EnergyMeter
from echostop.nlp_data import MODEL_NAME, get_nlp_dataset
from echostop.train import set_seed


@torch.no_grad()
def evaluate_nlp(model, loader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("labels")
        logits = model(**batch).logits
        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / total


def train_nlp_run(
    dataset_name: str, seed: int, epochs: int, batch_size: int, lr: float, out_dir: Path,
    device_index: int = 0,
) -> Path:
    set_seed(seed)
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

    bundle = get_nlp_dataset(dataset_name, seed=seed, batch_size=batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=bundle.num_classes
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(bundle.train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    run_id = f"{dataset_name}_distilbert_seed{seed}"
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
            for batch in bundle.train_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                out = model(**batch)
                loss = out.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                n_batches += 1
            current_lr = scheduler.get_last_lr()[0]

            val_acc = evaluate_nlp(model, bundle.val_loader, device)
            rec = meter.end_epoch()
            is_new_best = val_acc > best_val_acc
            test_acc_at_checkpoint = ""
            if is_new_best:
                best_val_acc = val_acc
                test_acc_at_checkpoint = evaluate_nlp(model, bundle.test_loader, device)
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
            print(f"[{run_id}] epoch {epoch:3d} | lr {current_lr:.2e} | "
                  f"loss {row['train_loss']:.4f} | val_acc {val_acc:6.2f}% | "
                  f"best {'*' if is_new_best else ' '} | E {rec.energy_joules:7.1f} J | "
                  f"t {rec.duration_seconds:5.1f}s")

    meter.stop()
    with open(meta_path, "w") as f:
        json.dump({"dataset": dataset_name, "model": "distilbert", "seed": seed, "epochs": epochs,
                    "batch_size": batch_size, "lr": lr, "gpu": meter.device_name}, f, indent=2)
    return log_path


def main() -> None:
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "sst2"
    t0 = time.time()
    for seed in [0, 1, 2]:
        train_nlp_run(dataset_name=dataset_name, seed=seed, epochs=10, batch_size=32, lr=2e-5,
                       out_dir=Path("logs/nlp_pilot"))
        print(f"seed {seed} done, elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
