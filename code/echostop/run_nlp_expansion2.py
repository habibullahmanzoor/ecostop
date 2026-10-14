"""NLP domain expansion, round 2 (session addition): a second batch of real
Hugging Face text-classification datasets beyond the first expansion
(emotion/tweet_eval_sentiment/poem_sentiment/banking77), directly targeting
the paper's own §6.3.1/§8 finding that NLP alone still can't support a
non-degenerate domain-only calibration bound even after the first
expansion (n=7, best UCB 0.587) -- this batch aims for roughly n≈27.
DistilBERT fine-tuning, same recipe as every other NLP job in this project
(10 epochs, 3 seeds).

SetFit/sst5 was in the original candidate list but its `text` column has a
nested structure the shared tokenizer/collator path can't batch; dropped
rather than special-cased for one dataset.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from echostop.train_nlp import train_nlp_run

EPOCHS = 10

DATASETS = [
    "cola", "mrpc", "rte", "qnli", "wnli",
    "tweet_eval_irony", "tweet_eval_hate", "tweet_eval_offensive", "tweet_eval_emoji",
    "tweet_eval_stance_abortion", "tweet_eval_stance_atheism", "tweet_eval_stance_climate",
    "tweet_eval_stance_feminist", "tweet_eval_stance_hillary",
    "imdb", "ag_news",
]
SEEDS = [0, 1, 2]
OUT_DIR = Path("logs/nlp_pilot")


def _already_done(dataset: str, seed: int) -> bool:
    """Resumability (session addition, after a background run got killed
    mid-job by a host interruption unrelated to this code): a run counts as
    done only if its CSV has exactly EPOCHS data rows, not just if the file
    exists -- a killed process leaves a truncated CSV (e.g. qnli seed0 died
    at epoch 3/10) that must be redone, not skipped."""
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

    print(f"\nNLP expansion round 2 complete in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
