"""Vision domain expansion, round 3, slow batch: STL-10 and Food-101, split
out from run_vision_expansion3.py because both are large, slow downloads
from their official academic hosts (STL-10 from cs.stanford.edu measured
at ~20MB/min on this connection; Food-101 was already the single largest
time cost in expansion round 2). Run as a separate process so these slow
downloads don't block the fast batch's much quicker jobs. Same recipe as
every other vision job in this project (ResNet-9, 40 epochs, 3 seeds).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from echostop.train import train_one_run

EXPANSION_EPOCHS = 40
SEEDS = [0, 1, 2]
MATRIX = [
    ("stl10", "resnet9"),
    ("food101", "resnet9"),
]

OUT_DIR = Path("logs/pilot")


def _already_done(dataset: str, model: str, seed: int) -> bool:
    path = OUT_DIR / f"{dataset}_{model}_seed{seed}.csv"
    if not path.exists():
        return False
    with open(path, newline="") as f:
        n_rows = sum(1 for _ in csv.reader(f)) - 1
    return n_rows >= EXPANSION_EPOCHS


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(MATRIX) * len(SEEDS)
    done = 0
    t0 = time.time()

    for dataset, model in MATRIX:
        for seed in SEEDS:
            done += 1
            if _already_done(dataset, model, seed):
                print(f"=== [{done}/{total}] {dataset} + {model}, seed {seed} -- already complete, skipping ===")
                continue
            print(f"\n=== [{done}/{total}] {dataset} + {model}, seed {seed} "
                  f"(elapsed {time.time() - t0:.0f}s) ===", flush=True)
            try:
                train_one_run(
                    dataset_name=dataset, model_name=model, seed=seed,
                    epochs=EXPANSION_EPOCHS, batch_size=256, lr=0.1, weight_decay=5e-4,
                    out_dir=OUT_DIR,
                )
                print(f"  done (elapsed {time.time() - t0:.0f}s)")
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e} -- skipping, continuing with remaining jobs")

    print(f"\nvision expansion round 3 (slow batch) complete in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
