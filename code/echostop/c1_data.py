"""C1 data layer: deterministic split carving + size rules. No model is involved anywhere in this file.

Split rules (c1_registry.SPLIT_SEED = 0 shuffles each split once; the run seed never touches the data split):
  test  official test split if it has labels (else the official validation split registered as test); if the spec has
        no evaluation split, 20% of the train rows; at most TEST_CAP rows (a seeded random subset).
  val   the official validation split if registered; else 10% of the train rows; at most VAL_CAP rows.
  train what remains, at most TRAIN_CAP rows (a seeded random subset).
Rows with a missing or negative label are removed first; validation/test rows whose label never occurs in train are
removed. Eligibility: train rows available >= MIN_TRAIN, test rows >= MIN_TEST, MIN_CLASSES <= classes <= MAX_CLASSES.
Output: cache/c1_data/<name>.pkl and c1_manifest.json (the frozen record of what was kept and why)."""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import c1_registry as R

CACHE = "cache/c1_data"


def _clean(ds, col):
    u = ds.unique(col)
    bad = [v for v in u if v is None or (isinstance(v, (int, float)) and (v != v or v < 0))]
    if not bad:
        return ds
    return ds.filter(lambda b: [(x is not None) and not (isinstance(x, (int, float)) and (x != x or x < 0)) for x in b[col]], batched=True)


def _rows(ds, spec):
    t = [[("" if x is None else str(x)) for x in ds[c]] for c in spec["text"]]
    y = list(ds[spec["label"]])
    return t, y


def prepare(name, force=False):
    path = f"{CACHE}/{name}.pkl"
    if os.path.exists(path) and not force:
        return pickle.load(open(path, "rb"))
    spec = R.DATASETS[name]
    raw = load_dataset(spec["id"], spec["cfg"]) if spec["cfg"] else load_dataset(spec["id"])
    lab = spec["label"]
    tr = _clean(raw[spec["train"]], lab).shuffle(seed=R.SPLIT_SEED)
    te = _clean(raw[spec["test"]], lab).shuffle(seed=R.SPLIT_SEED) if spec["test"] else None
    va = _clean(raw[spec["val"]], lab).shuffle(seed=R.SPLIT_SEED) if spec["val"] else None
    n_all = len(tr)
    pos = 0
    if te is None:
        n_test = min(R.TEST_CAP, round(0.2 * n_all))
        te, pos = tr.select(range(n_test)), n_test
    else:
        te = te.select(range(min(R.TEST_CAP, len(te))))
    if va is None:
        n_val = min(R.VAL_CAP, round(0.1 * (n_all - pos)))
        va, pos = tr.select(range(pos, pos + n_val)), pos + n_val
    else:
        va = va.select(range(min(R.VAL_CAP, len(va))))
    n_train_avail = n_all - pos
    tr = tr.select(range(pos, pos + min(R.TRAIN_CAP, n_train_avail)))
    (tt, ty), (vt, vy), (et, ey) = _rows(tr, spec), _rows(va, spec), _rows(te, spec)
    classes = sorted(set(ty), key=lambda v: (str(type(v)), v))
    remap = {c: i for i, c in enumerate(classes)}

    def keep(texts, ys):
        idx = [i for i, y in enumerate(ys) if y in remap]
        return [[col[i] for i in idx] for col in texts], [remap[ys[i]] for i in idx]

    vt, vy = keep(vt, vy)
    et, ey = keep(et, ey)
    ty = [remap[y] for y in ty]
    maj = 100 * Counter(ey).most_common(1)[0][1] / max(len(ey), 1) if ey else float("nan")
    why = []
    if n_train_avail < R.MIN_TRAIN:
        why.append(f"train rows {n_train_avail} < {R.MIN_TRAIN}")
    if len(ey) < R.MIN_TEST:
        why.append(f"test rows {len(ey)} < {R.MIN_TEST}")
    if not R.MIN_CLASSES <= len(classes) <= R.MAX_CLASSES:
        why.append(f"classes {len(classes)} outside [{R.MIN_CLASSES}, {R.MAX_CLASSES}]")
    out = {"name": name, "texts": {"train": tt, "val": vt, "test": et}, "labels": {"train": ty, "val": vy, "test": ey},
           "info": {"hf_id": spec["id"], "cfg": spec["cfg"], "n_train": len(ty), "n_train_available": n_train_avail,
                    "n_val": len(vy), "n_test": len(ey), "n_classes": len(classes), "test_majority_pct": round(maj, 1),
                    "eligible": not why, "dropped_because": why}}
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump(out, open(path, "wb"))
    return out


def main():
    names = list(R.DATASETS)
    if len(sys.argv) > 1:
        names = [n for n in names if n in sys.argv[1:]]
    manifest = json.load(open("c1_manifest.json")) if os.path.exists("c1_manifest.json") and len(sys.argv) > 1 else {}
    for n in names:
        try:
            info = prepare(n)["info"]
        except Exception as e:
            info = {"hf_id": R.DATASETS[n]["id"], "cfg": R.DATASETS[n]["cfg"], "eligible": False,
                    "dropped_because": [f"load error: {type(e).__name__}: {str(e)[:120]}"]}
        manifest[n] = info
        flag = "ELIGIBLE" if info["eligible"] else "DROPPED"
        print(f"{flag:8s} {n:16s} " + (f"train {info.get('n_train_available', '?'):>7} (use {info.get('n_train', '?'):>5}) val {info.get('n_val', '?'):>5} "
              f"test {info.get('n_test', '?'):>5} classes {info.get('n_classes', '?'):>3} majority {info.get('test_majority_pct', '?')}%  "
              + "; ".join(info["dropped_because"])), flush=True)
        json.dump(manifest, open("c1_manifest.json", "w"), indent=1)
    ok = [n for n, i in manifest.items() if i["eligible"]]
    print(f"\n{len(ok)} eligible datasets of {len(manifest)}")


if __name__ == "__main__":
    main()
