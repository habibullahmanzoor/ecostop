"""Vision domain expansion (session addition): grow the vision calibration
corpus beyond its original 2-dataset x 2-architecture pilot, directly
addressing the paper's own §6.3.1 finding that vision alone (5 jobs) can't
support any non-degenerate calibration bound. New real, torchvision-native
datasets from the plan's own §16 calibration-corpus job pool: MNIST, KMNIST,
EMNIST-letters -- same speed class as Fashion-MNIST, all real handwritten
digits/characters, no synthetic data. ResNet-9 only for this first pass
(ResNet-18 variants are a natural follow-up if GPU time allows).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from echostop.train import train_one_run

EXPANSION_EPOCHS = 40  # match the existing vision pilot's budget for comparability
SEEDS = [0, 1, 2]
MATRIX = [
    ("mnist", "resnet9"),
    ("kmnist", "resnet9"),
    ("emnist_letters", "resnet9"),
]

OUT_DIR = Path("logs/pilot")  # same directory as the original pilot -- these ARE vision-domain jobs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(MATRIX) * len(SEEDS)
    done = 0
    t0 = time.time()

    for dataset, model in MATRIX:
        for seed in SEEDS:
            done += 1
            print(f"\n=== [{done}/{total}] {dataset} + {model}, seed {seed} "
                  f"(elapsed {time.time() - t0:.0f}s) ===", flush=True)
            train_one_run(
                dataset_name=dataset, model_name=model, seed=seed,
                epochs=EXPANSION_EPOCHS, batch_size=256, lr=0.1, weight_decay=5e-4,
                out_dir=OUT_DIR,
            )
            print(f"  done (elapsed {time.time() - t0:.0f}s)")

    print(f"\nvision expansion complete in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
