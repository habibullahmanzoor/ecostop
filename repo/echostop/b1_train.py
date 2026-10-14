"""B1 fresh-test trainer. Same CSV schema / EnergyMeter / audit convention as the corpus
(val-selected checkpoint, test score logged on improvement), but new architectures,
optimisers, schedules, datasets and NLP model (see b1_registry.py).

Runs strictly one job at a time: concurrent jobs would contaminate NVML energy.
Resumable: a run counts as done only if its CSV has the full number of epochs.

usage: python -m echostop.b1_train [tabular|nlp|vision|all]
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")  # installed TensorFlow conflicts with protobuf; unused here
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

from echostop import b1_registry as R
from echostop.energy import EnergyMeter
from echostop.train import evaluate, set_seed

OUT = Path("logs/b1")
FIELDS = ["epoch", "train_loss", "val_acc", "is_new_best", "test_acc_at_checkpoint", "energy_joules",
          "duration_seconds", "mean_power_watts", "cumulative_energy_joules", "cumulative_duration_seconds"]


def done(path: Path, epochs: int) -> bool:
    if not path.exists():
        return False
    with open(path, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1 >= epochs


def run_loop(run_id, out_dir, epochs, train_epoch, eval_val, eval_test, meta, sample_interval=0.02, min_epoch_s=0.0):
    """Generic epoch loop; train_epoch() -> mean loss; eval_* -> accuracy (%)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{run_id}.csv"
    meter = EnergyMeter(device_index=0, sample_interval_s=sample_interval)
    meter.start()
    best, cum_e, cum_d = -1.0, 0.0, 0.0
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for ep in range(epochs):
            meter.begin_epoch(ep)
            t0 = time.monotonic()
            loss = train_epoch(ep)
            if min_epoch_s and time.monotonic() - t0 < min_epoch_s:
                time.sleep(min_epoch_s - (time.monotonic() - t0))
            val = eval_val()
            rec = meter.end_epoch()
            new_best = val > best
            test = ""
            if new_best:
                best = val
                test = eval_test()
            cum_e += rec.energy_joules
            cum_d += rec.duration_seconds
            w.writerow({"epoch": ep, "train_loss": loss, "val_acc": val, "is_new_best": int(new_best),
                        "test_acc_at_checkpoint": test, "energy_joules": rec.energy_joules,
                        "duration_seconds": rec.duration_seconds, "mean_power_watts": rec.mean_power_watts,
                        "cumulative_energy_joules": cum_e, "cumulative_duration_seconds": cum_d})
            f.flush()
            print(f"[{run_id}] ep {ep:3d} val {val:6.2f} E {rec.energy_joules:8.1f}J t {rec.duration_seconds:6.1f}s", flush=True)
    meter.stop()
    json.dump({**meta, "gpu": meter.device_name}, open(out_dir / f"{run_id}.meta.json", "w"), indent=2)


# ------------------------------------------------------------------ vision
class ViTTiny(nn.Module):
    def __init__(self, in_ch, nc, img, dim=192, depth=6, heads=3):
        super().__init__()
        patch = 4 if img <= 36 else 8
        self.embed = nn.Conv2d(in_ch, dim, patch, patch)
        n = (img // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, nc)

    def forward(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        return self.head(self.norm(self.enc(x))[:, 0])


def vgg_small(in_ch, nc):
    def blk(i, o):
        return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
    return nn.Sequential(*blk(in_ch, 64), *blk(64, 64), nn.MaxPool2d(2), *blk(64, 128), *blk(128, 128), nn.MaxPool2d(2),
                         *blk(128, 256), *blk(256, 256), nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.3), nn.Linear(256, nc))


def build_vision(arch, in_ch, nc, img):
    from echostop.models import build_model
    if arch == "mobilenetv3":
        m = torchvision.models.mobilenet_v3_small(weights=None, num_classes=nc)
        m.features[0][0] = nn.Conv2d(in_ch, 16, 3, stride=1, padding=1, bias=False)
        return m
    if arch == "vit_tiny":
        return ViTTiny(in_ch, nc, img)
    if arch == "resnet18_step":
        return build_model("resnet18", in_ch, nc)
    if arch == "vgg_small":
        return vgg_small(in_ch, nc)
    raise ValueError(arch)


def train_vision(arch, dataset, seed):
    from echostop.data import get_dataset
    rc = R.VISION_RECIPES[arch]
    run_id = f"{dataset}_{arch}_seed{seed}"
    if done(OUT / "vision" / f"{run_id}.csv", rc["epochs"]):
        return
    set_seed(seed)
    dev = torch.device("cuda")
    b = get_dataset(dataset, seed=seed, batch_size=rc["bs"])
    img = next(iter(b.val_loader))[0].shape[-1]
    model = build_vision(arch, b.in_channels, b.num_classes, img).to(dev)
    if rc["opt"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=rc["lr"], weight_decay=rc["wd"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr=rc["lr"], momentum=0.9, weight_decay=rc["wd"], nesterov=True)
    E, spe = rc["epochs"], len(b.train_loader)
    per_step = rc["sched"] == "onecycle"
    if rc["sched"] == "cosine_warmup":
        wu = rc["warmup"]
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / wu if e < wu else
                                                0.5 * (1 + math.cos(math.pi * (e - wu) / max(1, E - wu))))
    elif rc["sched"] == "step":
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.5 * E), int(0.75 * E)], 0.1)
    else:
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, rc["lr"], epochs=E, steps_per_epoch=spe)
    crit = nn.CrossEntropyLoss()
    clip = rc["opt"] == "adamw"

    def train_epoch(ep):
        model.train()
        tot, n = 0.0, 0
        for x, y in b.train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if per_step:
                sch.step()
            tot += loss.item(); n += 1
        if not per_step:
            sch.step()
        return tot / max(n, 1)

    run_loop(run_id, OUT / "vision", E, train_epoch, lambda: evaluate(model, b.val_loader, dev),
             lambda: evaluate(model, b.test_loader, dev), {"dataset": dataset, "arch": arch, "seed": seed, **rc})


# ------------------------------------------------------------------ tabular
def train_tabular(name, seed):
    from echostop.models import TabularMLP
    from echostop.openml_data import get_openml_dataset
    rc = R.TABULAR_RECIPE
    run_id = f"tabular_{name}_seed{seed}"
    if done(OUT / "tabular" / f"{run_id}.csv", rc["epochs"]):
        return
    set_seed(seed)
    dev = torch.device("cuda")
    b = get_openml_dataset(name, seed=seed, batch_size=rc["bs"])
    model = TabularMLP(b.in_features, b.num_classes, hidden=rc["hidden"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=rc["lr"], weight_decay=rc["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, rc["epochs"])
    crit = nn.CrossEntropyLoss()

    def train_epoch(ep):
        model.train()
        tot, n = 0.0, 0
        for x, y in b.train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item(); n += 1
        sch.step()
        return tot / max(n, 1)

    run_loop(run_id, OUT / "tabular", rc["epochs"], train_epoch, lambda: evaluate(model, b.val_loader, dev),
             lambda: evaluate(model, b.test_loader, dev), {"dataset": name, "seed": seed, **rc},
             sample_interval=0.002, min_epoch_s=0.025)  # same short-epoch padding as the corpus tabular runs


# ------------------------------------------------------------------ NLP
def nlp_bundle(name, seed, bs):
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorWithPadding
    path, cfg, tcol, t2col, lcol, val_split, test_split, max_train = R.NLP_JOBS[name]
    raw = load_dataset(path, cfg) if cfg else load_dataset(path)
    train = raw["train"].shuffle(seed=seed)
    test = raw[test_split]
    if val_split is None:
        n_val = max(1, int(0.1 * len(train)))
        val, train = train.select(range(n_val)), train.select(range(n_val, len(train)))
    else:
        val = raw[val_split]
        if val_split == test_split:      # no separate test: carve it from train-side-held-out validation
            idx = np.random.RandomState(seed).permutation(len(val))
            h = len(val) // 2
            val, test = val.select(idx[:h]), val.select(idx[h:])
    if max_train and len(train) > max_train:
        train = train.select(range(max_train))
    tok = AutoTokenizer.from_pretrained(R.NLP_MODEL)
    labels = sorted(set(train[lcol]))
    remap = {l: i for i, l in enumerate(labels)}

    def prep(ds):
        def f(batch):
            enc = tok(batch[tcol], batch[t2col], truncation=True, max_length=128) if t2col else \
                tok(batch[tcol], truncation=True, max_length=128)
            enc["labels"] = [remap.get(int(l), 0) for l in batch[lcol]]
            return enc
        ds = ds.map(f, batched=True, remove_columns=ds.column_names)
        ds.set_format(type="torch")
        return ds

    col = DataCollatorWithPadding(tok)
    g = torch.Generator().manual_seed(seed)
    mk = lambda ds, sh, bsz: DataLoader(prep(ds), batch_size=bsz, shuffle=sh, collate_fn=col, generator=g if sh else None)
    return mk(train, True, bs), mk(val, False, 64), mk(test, False, 64), len(labels)


def train_nlp(name, seed):
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from echostop.train_nlp import evaluate_nlp
    rc = R.NLP_RECIPE
    run_id = f"{name}_bertsmall_seed{seed}"
    if done(OUT / "nlp" / f"{run_id}.csv", rc["epochs"]):
        return
    set_seed(seed)
    dev = torch.device("cuda")
    tr, va, te, nc = nlp_bundle(name, seed, rc["bs"])
    model = AutoModelForSequenceClassification.from_pretrained(R.NLP_MODEL, num_labels=nc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=rc["lr"], weight_decay=0.01)
    total = len(tr) * rc["epochs"]
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    def train_epoch(ep):
        model.train()
        tot, n = 0.0, 0
        for batch in tr:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            opt.step(); sch.step()
            tot += loss.item(); n += 1
        return tot / max(n, 1)

    run_loop(run_id, OUT / "nlp", rc["epochs"], train_epoch, lambda: evaluate_nlp(model, va, dev),
             lambda: evaluate_nlp(model, te, dev), {"dataset": name, "model": R.NLP_MODEL, "seed": seed, **rc})


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("tabular", "all"):
        for seed in R.SEEDS["tabular"]:
            for name in R.TABULAR_JOBS:
                try:
                    train_tabular(name, seed)
                except Exception as e:
                    print(f"FAILED tabular {name}: {type(e).__name__}: {e}", flush=True)
    if which in ("nlp", "all"):
        for seed in R.SEEDS["nlp"]:
            for name in R.NLP_JOBS:
                try:
                    train_nlp(name, seed)
                except Exception as e:
                    print(f"FAILED nlp {name}: {type(e).__name__}: {e}", flush=True)
                print(f"-- elapsed {time.time() - t0:.0f}s", flush=True)
    if which in ("vision", "all"):
        for seed in R.SEEDS["vision"]:
            for arch, ds in R.VISION_JOBS:
                try:
                    train_vision(arch, ds, seed)
                except Exception as e:
                    print(f"FAILED vision {arch}/{ds}: {type(e).__name__}: {e}", flush=True)
                print(f"-- elapsed {time.time() - t0:.0f}s", flush=True)
    print("B1 training pass complete", flush=True)


if __name__ == "__main__":
    main()
