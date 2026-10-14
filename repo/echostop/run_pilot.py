"""Orchestrates the minimal viable pilot study (plan §40).

Matrix: {fashion_mnist, cifar10} x {resnet9, resnet18} x 3 seeds, each
trained to completion (no early stopping) at a pre-specified epoch budget
(plan §11.1 -- T must be fixed before seeing curves, not chosen after).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from echostop.train import train_one_run

PILOT_EPOCHS = 40  # pre-specified T (plan §11.1) -- decided before this run, not after
SEEDS = [0, 1, 2]
MATRIX = [
    ("fashion_mnist", "resnet9"),
    ("fashion_mnist", "resnet18"),
    ("cifar10", "resnet9"),
    ("cifar10", "resnet18"),
]

OUT_DIR = Path("logs/pilot")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"
    manifest = {
        "epochs": PILOT_EPOCHS,
        "seeds": SEEDS,
        "matrix": MATRIX,
        "runs": [],
    }

    total = len(MATRIX) * len(SEEDS)
    done = 0
    t0 = time.time()

    for dataset, model in MATRIX:
        for seed in SEEDS:
            done += 1
            print(
                f"\n=== [{done}/{total}] {dataset} + {model}, seed {seed} "
                f"(elapsed {time.time() - t0:.0f}s) ===",
                flush=True,
            )
            log_path = train_one_run(
                dataset_name=dataset,
                model_name=model,
                seed=seed,
                epochs=PILOT_EPOCHS,
                batch_size=256,
                lr=0.1,
                weight_decay=5e-4,
                out_dir=OUT_DIR,
            )
            manifest["runs"].append(
                {"dataset": dataset, "model": model, "seed": seed, "log": str(log_path)}
            )
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

    print(f"\npilot matrix complete in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
