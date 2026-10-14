"""Vision domain expansion, round 3, fast batch: directly targeting the
paper's own §6.3.1/§8 finding that vision alone still can't support a
non-degenerate domain-only calibration bound even after two expansion
rounds (n=20, best UCB 0.347). ResNet-9 only, same recipe as every other
vision job in this project.

Three real, reliable, fast-downloading datasets. STL-10 and Food-101 are
handled separately (run_vision_expansion3_slow.py) since both are
large/slow downloads from their official academic hosts and would block
this batch's much faster jobs if run in the same sequential process.

Country211 was probed and dropped entirely: its download alone reached
~6GB and was still growing (larger than Food-101) -- cut for
wall-clock/storage reasons before any training was attempted, not a
technical failure. Caltech-101 and PCAM were considered and dropped before
downloading anything: both require `gdown` (unauthenticated Google Drive
downloads), a materially less reliable dependency than every other loader
in this project, which all use direct HTTP from the dataset's own host.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from echostop.train import train_one_run

EXPANSION_EPOCHS = 40  # match the existing vision pilot's budget for comparability
SEEDS = [0, 1, 2]
MATRIX = [
    ("eurosat", "resnet9"),
    ("emnist_byclass", "resnet9"),
    ("imagenette", "resnet9"),
]

OUT_DIR = Path("logs/pilot")  # same directory as the original pilot -- these ARE vision-domain jobs


def _already_done(dataset: str, model: str, seed: int) -> bool:
    """Resumability: a run counts as done only if its CSV has exactly
    EXPANSION_EPOCHS data rows, not just if the file exists -- a killed
    process leaves a truncated CSV that must be redone, not skipped."""
    path = OUT_DIR / f"{dataset}_{model}_seed{seed}.csv"
    if not path.exists():
        return False
    with open(path, newline="") as f:
        n_rows = sum(1 for _ in csv.reader(f)) - 1  # minus header
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

    print(f"\nvision expansion round 3 complete in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
