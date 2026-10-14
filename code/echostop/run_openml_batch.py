"""Fetch + train the OpenML candidate datasets (plan §16), skipping any
that fail to load cleanly (missing/flaky/too-large-after-encoding) rather
than blocking the whole batch on one bad dataset.
"""

from __future__ import annotations

import time
from pathlib import Path

import sys

from echostop.openml_data import (
    CANDIDATE_DATASETS, CANDIDATE_DATASETS_BATCH2, CANDIDATE_DATASETS_BATCH3,
    CANDIDATE_DATASETS_BATCH4, CANDIDATE_DATASETS_BATCH5, fetch_and_clean,
)
from echostop.train_tabular import train_tabular_run

SEEDS = [0, 1, 2, 3]
OUT_DIR = Path("logs/calibration_openml")

BATCHES = {
    "1": CANDIDATE_DATASETS, "2": CANDIDATE_DATASETS_BATCH2,
    "3": CANDIDATE_DATASETS_BATCH3, "4": CANDIDATE_DATASETS_BATCH4,
    "5": CANDIDATE_DATASETS_BATCH5,
}


def main() -> None:
    batch = sys.argv[1] if len(sys.argv) > 1 else "1"
    candidates = BATCHES[batch]

    t0 = time.time()
    succeeded, failed = [], []

    for name, version in candidates:
        print(f"probing {name}... ", end="", flush=True)
        try:
            x, y = fetch_and_clean(name, version)
            print(f"OK  (n={x.shape[0]}, d={x.shape[1]}, classes={len(set(y))})")
            succeeded.append((name, version))
        except Exception as e:
            print(f"SKIP  ({type(e).__name__}: {str(e)[:80]})")
            failed.append((name, str(e)[:120]))

    print(f"\n{len(succeeded)}/{len(candidates)} datasets usable, "
          f"probing took {time.time()-t0:.1f}s\n")

    print("Training...")
    for name, version in succeeded:
        for seed in SEEDS:
            run_id = f"openml_{name}_seed{seed}"
            try:
                train_tabular_run(
                    dataset_name=name, seed=seed, epochs=60, batch_size=32,
                    lr=0.05, weight_decay=1e-4, out_dir=OUT_DIR,
                    source="openml", run_id_override=run_id,
                )
                print(f"  {run_id} done (elapsed {time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  {run_id} FAILED during training: {type(e).__name__}: {str(e)[:100]}")

    print(f"\ndone in {time.time()-t0:.1f}s")
    print(f"succeeded datasets: {[n for n, v in succeeded]}")
    if failed:
        print(f"\nfailed datasets:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
