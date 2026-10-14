"""NLP domain expansion, round 3: directly targeting the paper's own
§6.3.1/§8 finding that NLP alone still can't support a non-degenerate
domain-only calibration bound even after two expansion rounds (n=23, best
UCB 0.324). DistilBERT fine-tuning, same recipe as every other NLP job in
this project (10 epochs, 3 seeds).

Yelp Polarity and Amazon Polarity's official corpora are enormous (560k
and 3.6M train examples); both are capped to 60k train / 3k val / 5k test
in nlp_data.py to keep training time comparable to the corpus's other
large jobs (AG News, QNLI), documented there rather than silently truncated.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from echostop.train_nlp import train_nlp_run

EPOCHS = 10

DATASETS = [
    "yelp_polarity", "amazon_polarity", "clinc_oos", "sms_spam",
]
SEEDS = [0, 1, 2]
OUT_DIR = Path("logs/nlp_pilot")


def _already_done(dataset: str, seed: int) -> bool:
    """Resumability: a run counts as done only if its CSV has exactly
    EPOCHS data rows, not just if the file exists -- a killed process
    leaves a truncated CSV that must be redone, not skipped."""
    path = OUT_DIR / f"{dataset}_distilbert_seed{seed}.csv"
    if not path.exists():
        return False
    with open(path, newline="") as f:
        n_rows = sum(1 for _ in csv.reader(f)) - 1  # minus header
    return n_rows >= EPOCHS


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(DATASETS) * len(SEEDS)
    done = 0
    t0 = time.time()

    for dataset in DATASETS:
        for seed in SEEDS:
            done += 1
            if _already_done(dataset, seed):
                print(f"=== [{done}/{total}] {dataset} + distilbert, seed {seed} -- already complete, skipping ===")
                continue
            print(f"\n=== [{done}/{total}] {dataset} + distilbert, seed {seed} "
                  f"(elapsed {time.time() - t0:.0f}s) ===", flush=True)
            try:
                train_nlp_run(dataset_name=dataset, seed=seed, epochs=EPOCHS, batch_size=32, lr=2e-5,
                               out_dir=OUT_DIR)
                print(f"  done (elapsed {time.time() - t0:.0f}s)")
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e} -- skipping, continuing with remaining jobs")

    print(f"\nNLP expansion round 3 complete in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
