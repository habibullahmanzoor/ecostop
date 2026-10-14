"""C1 (confirmatory short-schedule fine-tuning study): outcome-blind harvest of candidate datasets.

Rule (fixed before any C1 training and before looking at any model output):
  pool       the 500 most-downloaded Hub datasets tagged task_categories:text-classification and language:en;
  keep       ungated, no loading script, a single default config, exactly one ClassLabel column with 2..50 classes,
             1..2 string feature columns, a train split with >= 1,500 rows and a labelled validation or test split;
  exclude    corpora already in the 142-job development corpus or the 35-job B1 fresh set.
Output: cache/c1_candidates.json (metadata only). Selection of the final list from these candidates is done in
c1_registry.py by manual review that is blind to any training result (there is none for these datasets)."""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("USE_TF", "0")
from datasets import load_dataset_builder
from datasets.features import ClassLabel, Value
from huggingface_hub import HfApi

USED = ("glue", "super_glue", "imdb", "ag_news", "rotten_tomatoes", "sst2", "sst5", "emotion", "clinc", "banking77", "trec",
        "tweet_eval", "poem_sentiment", "yelp_polarity", "amazon_polarity", "sms_spam", "enron_spam", "subj", "bbc",
        "hate_speech_offensive", "student", "/cr", "boolq")


def probe(ds_id):
    try:
        b = load_dataset_builder(ds_id)
    except Exception as e:                                              # scripts, gated, multi-config, broken metadata
        return {"id": ds_id, "ok": False, "why": type(e).__name__ + ": " + str(e)[:80]}
    info = b.info
    if not info.features or not info.splits:
        return {"id": ds_id, "ok": False, "why": "no features/splits in metadata"}
    labels = {k: v for k, v in info.features.items() if isinstance(v, ClassLabel)}
    strs = [k for k, v in info.features.items() if isinstance(v, Value) and v.dtype in ("string", "large_string")]
    sizes = {k: v.num_examples for k, v in info.splits.items()}
    return {"id": ds_id, "ok": True, "labels": {k: v.num_classes for k, v in labels.items()}, "strings": strs, "splits": sizes}


def main():
    api = HfApi()
    pool = list(api.list_datasets(filter=["task_categories:text-classification", "language:en"], sort="downloads",
                                  direction=-1, limit=500))
    ids = [d.id for d in pool if not d.gated and not any(u in d.id.lower() for u in USED)]
    print(len(ids), "ungated, not-obviously-used ids", flush=True)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(probe, ids))
    good = []
    for r in res:
        if not r["ok"] or len(r["labels"]) != 1 or not 1 <= len(r["strings"]) <= 2:
            continue
        nc = list(r["labels"].values())[0]
        sp = r["splits"]
        if not 2 <= nc <= 50 or sp.get("train", 0) < 1500 or not ({"validation", "test"} & set(sp)):
            continue
        good.append(r)
    os.makedirs("cache", exist_ok=True)
    json.dump({"all": res, "candidates": good}, open("cache/c1_candidates.json", "w"), indent=1)
    print(len(good), "candidates")
    for r in good:
        print(f"{r['id']:55s} classes={list(r['labels'].values())[0]:3d} text={r['strings']} splits={r['splits']}")


if __name__ == "__main__":
    main()
