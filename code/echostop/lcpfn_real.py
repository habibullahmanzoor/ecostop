"""Real pretrained LC-PFN (Adriaensen et al. 2023) as a stopping predictor.

Score s_t = P(Q_T > B_t + eps | prefix) under the PFN's predictive distribution at
the final epoch T (the PFN paper's own stopping quantity). Calibrated with the same
fixed-sequence HB wrapper and scored out-of-fold, in two forms:
  LC-PFN (original rule)  : score only, k=1, no flat-gate
  LC-PFN + EcoStop gates  : score plus flat-gate and k=2 consecutive
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import tierA
from echostop.ablation import sim
import functools
from lcpfn import LCPFN

# Official LC-PFN checkpoints are full pickled models (torch>=2.6 defaults to weights_only=True).
# Trusted source: weights downloaded by the lcpfn package itself. Scoped to this script only.
torch.load = functools.partial(torch.load, weights_only=False)

def _patch_layers(model):
    # lcpfn's custom encoder layer predates torch>=2.x, whose TransformerEncoder passes is_causal=...
    for m in model.modules():
        if isinstance(m, torch.nn.GELU) and not hasattr(m, 'approximate'):
            m.approximate = 'none'
        cls = type(m)
        if cls.__name__ == "TransformerEncoderLayer" and not getattr(cls, "_ecostop_patched", False):
            orig = cls.forward
            cls.forward = lambda self, *a, _o=orig, is_causal=None, **k: _o(self, *a, **k)
            cls._ecostop_patched = True


EPS = 0.5
PFN_GRID = [0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CACHE = "cache/lcpfn_traces.pkl"


def pfn_trace(model, df: pd.DataFrame, dev):
    T = len(df)
    n_min = max(2, round(0.125 * T))
    q = df["val_acc"].values.astype(np.float32) / 100.0
    b = df["B_t"].values
    x_all = torch.arange(1, T + 1, dtype=torch.float32, device=dev)
    out = []
    for t in range(T):
        flat = 0
        for s in range(t, 0, -1):
            if b[s] - b[s - 1] > 0.10:
                break
            flat += 1
        if t < n_min or t + 1 < 4:
            out.append({"t": t, "s_t": None, "epochs_flat": None}); continue
        xt = x_all[: t + 1]
        yt = torch.tensor(q[: t + 1], device=dev)
        with torch.no_grad():
            logits = model(x_train=xt, y_train=yt, x_test=x_all[T - 1: T])
            p = model.model.criterion.pi(logits, torch.tensor((b[t] + EPS) / 100.0, device=dev))
        out.append({"t": t, "s_t": float(p.reshape(-1)[0]), "epochs_flat": flat})
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    jobs, _, dom = tierA.load()
    names = list(jobs)
    dom_arr = np.array([dom[k] for k in names])
    if os.path.exists(CACHE):
        traces = pickle.load(open(CACHE, "rb"))
    else:
        model = LCPFN().to(dev)
        _patch_layers(model)
        model.eval()
        traces = {}
        for i, k in enumerate(names):
            traces[k] = [pfn_trace(model, df, dev) for df in jobs[k]]
            if i % 10 == 0:
                print(f"{i}/{len(names)} {k}", flush=True)
        pickle.dump(traces, open(CACHE, "wb"))

    variants = {
        "LC-PFN (original rule: score only, k=1)": lambda g: (lambda df, tr: sim(tr, g, len(df), 0, 1)),
        "LC-PFN + EcoStop gates (flat-gate, k=2)": lambda g: (lambda df, tr: sim(
            tr, g, len(df), max(2, round(0.125 * len(df))), 2)),
    }
    rows = []
    for vname, mk in variants.items():
        plist = [(f"g={g}", mk(g)) for g in PFN_GRID]
        mats = tierA.evaluate_family(jobs, traces, plist)
        # in the PFN rule a HIGHER gamma is more aggressive, same ordering as EcoStop
        for rho in (0.10, 0.15, 0.20):
            for pdom in (False, True):
                r = tierA.crossfit(*mats, dom_arr, rho, 0.05, per_domain=pdom)
                rows.append({"method": vname, "per_domain": pdom, "rho": rho, "oof_risk": r["risk"],
                             "risk_vision": r["risk_by_dom"]["vision"], "mean_saved": r["saved"],
                             "energy_weighted_saved": r["ew"], "uncertified_folds": r["n_uncert"]})
        print(vname, "risk curve:", dict(zip(PFN_GRID, mats[0].mean(1).round(3))), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("figures/tierA/lcpfn_real.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
