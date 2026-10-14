"""Vision domain expansion, round 2 (session addition): a second batch of
real, torchvision-native datasets beyond the first expansion
(run_vision_expansion.py's mnist/kmnist/emnist_letters), directly targeting
the paper's own §6.3.1/§8 finding that vision alone still can't support a
non-degenerate domain-only calibration bound even after the first
expansion (n=8, best UCB 0.549) -- this batch aims for roughly n≈27.
ResNet-9 only, same recipe as every other vision job in this project.

USPS was in the original candidate list but its host fails SSL
certificate verification on this machine (external issue, not fixed here);
dropped rather than worked around with insecure downloads.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from echostop.train import train_one_run

EXPANSION_EPOCHS = 40  # match the existing vision pilot's budget for comparability
SEEDS = [0, 1, 2]
MATRIX = [
    ("qmnist", "resnet9"),
    ("emnist_balanced", "resnet9"),
    ("emnist_digits", "resnet9"),
    ("cifar100", "resnet9"),
    ("svhn", "resnet9"),
    ("gtsrb", "resnet9"),
    ("flowers102", "resnet9"),
    ("dtd", "resnet9"),
    ("fgvc_aircraft", "resnet9"),
    ("rendered_sst2", "resnet9"),
    ("oxford_pet", "resnet9"),
    ("semeion", "resnet9"),
]
# food101 and stl10 were in the original candidate list but dropped mid-run
# (session decision, time-vs-corpus-size tradeoff): food101 alone was the
# single largest time cost in this batch (~1.5-2h) and stl10 the second
# largest (~30-45min), together roughly half the batch's total runtime for
# 2 of 14 datasets. Cut to finish the remaining small/fast datasets quickly
# rather than let those two dominate the wall-clock.

OUT_DIR = Path("logs/pilot")  # same directory as the original pilot -- these ARE vision-domain jobs


def _already_done(dataset: str, model: str, seed: int) -> bool:
    """Resumability (session addition, after a background run got killed
    mid-job by a host interruption unrelated to this code): a run counts as
    done only if its CSV has exactly EXPANSION_EPOCHS data rows, not just if
    the file exists -- a killed process leaves a truncated CSV that must be
    redone, not skipped."""
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

    print(f"\nvision expansion round 2 complete in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
