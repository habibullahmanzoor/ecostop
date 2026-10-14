"""SLM S0 pilot trainer (see SLM_PLAN.md). LoRA fine-tuning of a small causal LM on classification tasks, scored by the
log-likelihood of a label letter after the prompt (no free generation).

Logged every 1/8 epoch (so 1/4, 1/2 and 1-epoch granularities can be obtained by sub-sampling in the analysis):
  train loss, validation accuracy and loss, test accuracy (AUDIT ONLY: never used by a stopping rule), and NVML energy
  and duration of three separate windows: training chunk, validation evaluation, test evaluation.
Deployed energy of a stop at an evaluation point = training energy so far + validation-evaluation energy of the evaluations
the practitioner would have run; test-evaluation energy is excluded (it exists only to audit accuracy loss).

usage: python -m echostop.slm_train [--tasks a b] [--seeds 0 1] [--model qwen05] [--out logs/slm/s0] [--smoke]
Resumable: a run counts as done only if its CSV has every evaluation row and its meta.json exists.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import csv
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pynvml
import torch
import torch.nn.functional as F

from echostop.energy import EnergyMeter
from echostop.train import set_seed

MODELS = {"qwen05": "Qwen/Qwen2.5-0.5B-Instruct"}
# pilot tasks (reuse the C1 dataset splits; these datasets are EXCLUDED from the S1/S2 pools)
PILOT_TASKS = ["okite_news", "scitail", "paws", "cf_airline", "medical_abs", "amazon_stars"]
RECIPE = dict(epochs=3, bs=16, lr=2e-4, wd=0.0, warm=0.05, lora_r=16, lora_alpha=32, lora_dropout=0.05,
              max_chars=1100, max_tokens=320, train_cap=3840, val_cap=500, test_cap=1000, evals_per_epoch=8)
LETTERS = "ABCDEFGHIJ"
FIELDS = ["step", "epoch", "train_loss", "val_acc", "val_loss", "test_acc", "e_train", "d_train", "e_val", "d_val", "e_test", "d_test"]
IDLE_POWER_W = 25.0


def keep_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    except Exception:
        pass


def gpu_state():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    u, p = [], []
    for _ in range(5):
        u.append(pynvml.nvmlDeviceGetUtilizationRates(h).gpu); p.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0); time.sleep(0.2)
    return {"pre_util_raw_pct_mean": float(np.mean(u)), "pre_power_w_mean": float(np.mean(p)),
            "pre_mem_used_mib": pynvml.nvmlDeviceGetMemoryInfo(h).used / 2 ** 20}


def wait_for_idle():
    waited, s = 0.0, gpu_state()
    while s["pre_util_raw_pct_mean"] > 10.0 and s["pre_power_w_mean"] >= IDLE_POWER_W and waited < 3600:
        time.sleep(30); waited += 30; s = gpu_state()
    s["waited_for_idle_gpu_s"] = waited
    return s


def load_task(name, rc):
    d = pickle.load(open(f"cache/c1_data/{name}.pkl", "rb"))
    caps = dict(train=rc["train_cap"], val=rc["val_cap"], test=rc["test_cap"])
    out = {}
    for sp in ("train", "val", "test"):
        cols, y = d["texts"][sp], d["labels"][sp]
        n = min(caps[sp], len(y))
        out[sp] = ([[c[i] for c in cols] for i in range(n)], [int(v) for v in y[:n]])
    return out, int(d["info"]["n_classes"])


def prompt(cols, k, max_chars):
    names = ["Text"] if len(cols) == 1 else [f"Text {i + 1}" for i in range(len(cols))]
    body = "\n".join(f"{n}: {str(c)[:max_chars]}" for n, c in zip(names, cols))
    return f"{body}\nLabel ({'/'.join(LETTERS[:k])}):"


def encode(tok, rows, k, rc):
    enc = tok([prompt(r, k, rc["max_chars"]) for r in rows], truncation=True, max_length=rc["max_tokens"])
    return enc["input_ids"]


def collate(ids, pad, dev):
    L = max(len(x) for x in ids)
    inp = torch.full((len(ids), L), pad, dtype=torch.long)
    att = torch.zeros((len(ids), L), dtype=torch.long)
    for i, x in enumerate(ids):
        inp[i, L - len(x):] = torch.tensor(x); att[i, L - len(x):] = 1
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    return inp.to(dev), att.to(dev), pos.to(dev)


def last_logits(model, inp, att, pos):
    return model(input_ids=inp, attention_mask=att, position_ids=pos, logits_to_keep=1).logits[:, -1, :].float()


@torch.no_grad()
def evaluate(model, ids, y, label_ids, k, pad, dev, order, bs=64):
    model.eval()
    correct, loss = 0, 0.0
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        inp, att, pos = collate([ids[i] for i in idx], pad, dev)
        lg = last_logits(model, inp, att, pos)
        yy = torch.tensor([y[i] for i in idx], device=dev)
        tgt = label_ids[yy]
        loss += F.cross_entropy(lg, tgt, reduction="sum").item()
        correct += (lg[:, label_ids[:k]].argmax(-1) == yy).sum().item()
    return 100.0 * correct / len(order), loss / len(order)


def done(path, n_rows):
    return path.exists() and sum(1 for _ in open(path)) - 1 >= n_rows


def train_one(task, seed, mkey, data, k, tok, out_dir, rc, smoke=False):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    run_id = f"{task}_{mkey}_seed{seed}"
    ids = {sp: encode(tok, [r for r in data[sp][0]], k, rc) for sp in data}
    ys = {sp: data[sp][1] for sp in data}
    n = len(ys["train"])
    spe = math.ceil(n / rc["bs"])
    total = spe * rc["epochs"]
    n_pts = rc["epochs"] * rc["evals_per_epoch"]
    eval_steps = sorted({min(total, max(1, round(j * spe / rc["evals_per_epoch"]))) for j in range(1, n_pts + 1)})
    csv_path = out_dir / f"{run_id}.csv"
    if done(csv_path, len(eval_steps)) and (out_dir / f"{run_id}.meta.json").exists():
        return
    pre = wait_for_idle()
    set_seed(seed)
    dev = torch.device("cuda")
    base = AutoModelForCausalLM.from_pretrained(MODELS[mkey], torch_dtype=torch.bfloat16).to(dev)
    cfg = LoraConfig(r=rc["lora_r"], lora_alpha=rc["lora_alpha"], lora_dropout=rc["lora_dropout"], task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rc["lr"], weight_decay=rc["wd"])
    warm = max(1, int(rc["warm"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    pad = tok.pad_token_id
    label_ids = torch.tensor([tok.encode(" " + LETTERS[i], add_special_tokens=False)[0] for i in range(k)], device=dev)
    order_val = np.argsort([len(x) for x in ids["val"]]); order_test = np.argsort([len(x) for x in ids["test"]])
    g = torch.Generator().manual_seed(seed)
    meter = EnergyMeter(device_index=0, sample_interval_s=0.02)
    meter.start()
    torch.cuda.reset_peak_memory_stats()
    out_dir.mkdir(parents=True, exist_ok=True)
    step, chunk_loss, chunk_n, w = 0, 0.0, 0, 0
    es = set(eval_steps)
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader()
        meter.begin_epoch(w); w += 1
        for ep in range(rc["epochs"]):
            perm = torch.randperm(n, generator=g).tolist()
            for s in range(0, n, rc["bs"]):
                model.train()
                idx = perm[s:s + rc["bs"]]
                inp, att, pos = collate([ids["train"][i] for i in idx], pad, dev)
                lg = last_logits(model, inp, att, pos)
                tgt = label_ids[torch.tensor([ys["train"][i] for i in idx], device=dev)]
                loss = F.cross_entropy(lg, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step()
                step += 1; chunk_loss += loss.item(); chunk_n += 1
                if step in es:
                    torch.cuda.synchronize()
                    rt = meter.end_epoch()
                    meter.begin_epoch(w); w += 1
                    va, vl = evaluate(model, ids["val"], ys["val"], label_ids, k, pad, dev, order_val)
                    torch.cuda.synchronize(); rv = meter.end_epoch()
                    meter.begin_epoch(w); w += 1
                    ta, _ = evaluate(model, ids["test"], ys["test"], label_ids, k, pad, dev, order_test)
                    torch.cuda.synchronize(); re = meter.end_epoch()
                    wr.writerow({"step": step, "epoch": round(step / spe, 4), "train_loss": chunk_loss / max(chunk_n, 1),
                                 "val_acc": va, "val_loss": vl, "test_acc": ta,
                                 "e_train": rt.energy_joules, "d_train": rt.duration_seconds,
                                 "e_val": rv.energy_joules, "d_val": rv.duration_seconds,
                                 "e_test": re.energy_joules, "d_test": re.duration_seconds})
                    f.flush()
                    print(f"[{run_id}] step {step:4d}/{total} ep {step / spe:5.2f} val {va:5.1f} test {ta:5.1f} "
                          f"Etr {rt.energy_joules:6.0f}J Eval {rv.energy_joules:5.0f}J", flush=True)
                    chunk_loss, chunk_n = 0.0, 0
                    meter.begin_epoch(w); w += 1
    meter.stop()
    json.dump({"task": task, "model": MODELS[mkey], "seed": seed, "n_classes": k, "n_train": n, "n_val": len(ys["val"]),
               "n_test": len(ys["test"]), "steps": total, "steps_per_epoch": spe, "eval_steps": eval_steps, **rc, **pre,
               "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30, "torch": torch.__version__,
               "gpu": meter.device_name}, open(out_dir / f"{run_id}.meta.json", "w"), indent=1)
    del model, base, opt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=PILOT_TASKS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--model", default="qwen05")
    ap.add_argument("--out", default="logs/slm/s0")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    keep_awake()
    print("keep-awake request set", flush=True)
    from transformers import AutoTokenizer
    rc = dict(RECIPE)
    if a.smoke:
        rc.update(epochs=1, train_cap=160, val_cap=64, test_cap=64, evals_per_epoch=2)
    tok = AutoTokenizer.from_pretrained(MODELS[a.model], truncation_side="left")
    out_dir = Path(a.out)
    t0 = time.time()
    for task in a.tasks:
        data, k = load_task(task, rc)
        for seed in a.seeds:
            try:
                train_one(task, seed, a.model, data, k, tok, out_dir, rc, a.smoke)
            except Exception as e:
                print(f"FAILED {task}/seed{seed}: {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache()
        print(f"-- {task} done, elapsed {time.time() - t0:.0f}s", flush=True)
    print("SLM pass complete", flush=True)


if __name__ == "__main__":
    main()
