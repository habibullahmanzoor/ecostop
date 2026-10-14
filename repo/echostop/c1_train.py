"""C1 trainer: NLP fine-tuning jobs for the confirmatory EcoStop-SR study (see c1_registry.py).
Same CSV schema, EnergyMeter and audit convention (validation-selected checkpoint, test score logged on improvement)
as the corpus and B1. Strictly one run at a time (concurrent runs would contaminate NVML energy). Resumable: a run
counts as done only if its CSV has the full number of epochs. Dataset-major order, so each dataset's 6 runs
(2 arms x 3 seeds) complete together.

usage: python -m echostop.c1_train [--only name ...] [--seeds 0 1 2] [--arms A B] [--out logs/c1/nlp]
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pynvml
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)

from echostop import c1_data, c1_registry as R
from echostop.b1_train import done, run_loop
from echostop.train import set_seed
from echostop.train_nlp import evaluate_nlp


def gpu_state():
    """Utilisation / memory / power of the GPU just BEFORE a run: records whether anything else was using it."""
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    util, power = [], []
    for _ in range(5):
        util.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
        power.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
        time.sleep(0.2)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h).used / 2 ** 20
    try:
        nproc = len(pynvml.nvmlDeviceGetComputeRunningProcesses(h))
    except pynvml.NVMLError:
        nproc = -1
    return {"pre_util_pct_mean": float(np.mean(util)), "pre_power_w_mean": float(np.mean(power)),
            "pre_mem_used_mib": float(mem), "pre_compute_procs": nproc}


def tokenised(data, arm):
    tok = AutoTokenizer.from_pretrained(R.ARMS[arm]["model"])
    out = {}
    for split in ("train", "val", "test"):
        cols = data["texts"][split]
        enc = tok(*cols, truncation=True, max_length=R.MAX_LEN)
        d = {k: enc[k] for k in enc}
        d["labels"] = data["labels"][split]
        ds = Dataset.from_dict(d)
        ds.set_format(type="torch")
        out[split] = ds
    return tok, out


def train_one(name, arm, seed, data, tok, tds, out_dir):
    rc = R.ARMS[arm]
    run_id = f"{name}_{arm}_seed{seed}"
    if done(out_dir / f"{run_id}.csv", rc["epochs"]):
        return
    set_seed(seed)
    dev = torch.device("cuda")
    col = DataCollatorWithPadding(tok)
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(tds["train"], batch_size=rc["bs"], shuffle=True, collate_fn=col, generator=g)
    va = DataLoader(tds["val"], batch_size=64, shuffle=False, collate_fn=col)
    te = DataLoader(tds["test"], batch_size=64, shuffle=False, collate_fn=col)
    nc = data["info"]["n_classes"]
    model = AutoModelForSequenceClassification.from_pretrained(rc["model"], num_labels=nc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=rc["lr"], weight_decay=rc["wd"])
    total = len(tr) * rc["epochs"]
    sch = get_linear_schedule_with_warmup(opt, int(rc["warm"] * total), total)
    waited = 0.0
    pre = gpu_state()
    while pre["pre_util_pct_mean"] > 10.0 and waited < 3600:          # something else is using the GPU: wait, do not measure
        time.sleep(30)
        waited += 30
        pre = gpu_state()
    pre["waited_for_idle_gpu_s"] = waited

    def train_epoch(ep):
        model.train()
        tot, n = 0.0, 0
        for batch in tr:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), rc["clip"])
            opt.step(); sch.step()
            tot += loss.item(); n += 1
        return tot / max(n, 1)

    meta = {"dataset": name, "arm": arm, "seed": seed, **rc, **{k: v for k, v in data["info"].items() if k != "dropped_because"},
            **pre, "torch": torch.__version__}
    run_loop(run_id, out_dir, rc["epochs"], train_epoch, lambda: evaluate_nlp(model, va, dev),
             lambda: evaluate_nlp(model, te, dev), meta)
    del model, opt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int, default=R.SEEDS)
    ap.add_argument("--arms", nargs="*", default=list(R.ARMS))
    ap.add_argument("--out", default="logs/c1/nlp")
    a = ap.parse_args()
    out_dir = Path(a.out)
    manifest = json.load(open("c1_manifest.json"))
    names = [n for n in R.DATASETS if manifest.get(n, {}).get("eligible")]
    if a.only:
        names = [n for n in names if n in a.only]
    t0 = time.time()
    for n in names:
        data = c1_data.prepare(n)
        for arm in a.arms:
            try:
                tok, tds = tokenised(data, arm)
            except Exception as e:
                print(f"FAILED tokenise {n}/{arm}: {type(e).__name__}: {e}", flush=True)
                continue
            for seed in a.seeds:
                try:
                    train_one(n, arm, seed, data, tok, tds, out_dir)
                except Exception as e:
                    print(f"FAILED {n}/{arm}/seed{seed}: {type(e).__name__}: {e}", flush=True)
                    torch.cuda.empty_cache()
        print(f"-- {n} done, elapsed {time.time() - t0:.0f}s", flush=True)
    print("C1 training pass complete", flush=True)


if __name__ == "__main__":
    main()
