"""Freeze the method for the B1 fresh test. Calibrates every policy on the existing 142-job
corpus ONLY, writes b1_protocol.json (thresholds + code/registry hashes + timestamp).
Run once, BEFORE any B1 job is trained; b1_eval.py refuses to run if hashes differ."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import tierA
from echostop.ablation import sim
from echostop.lcpfn_real import PFN_GRID

# Only the method definition and the job list are hash-locked; analysis code may still evolve.
FILES = ["echostop/b1_registry.py", "echostop/methods.py", "echostop/calibration.py"]
TARGETS = [(0.15, 0.05), (0.20, 0.05)]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    if os.path.exists("b1_protocol.json"):
        raise SystemExit("b1_protocol.json already exists: protocol is frozen. Delete only to re-freeze BEFORE any B1 training.")
    jobs, traces, dom = tierA.load()
    names = list(jobs)
    dom_arr = np.array([dom[k] for k in names])
    fam = tierA.build_policies(None)
    mats = {f: tierA.evaluate_family(jobs, traces, pl) for f, pl in fam.items()}

    pfn_tr = pickle.load(open("cache/lcpfn_traces.pkl", "rb"))
    pfn_pl = [(f"g={g}", (lambda df, tr, g=g: sim(tr, g, len(df), max(2, round(0.125 * len(df))), 2))) for g in PFN_GRID]
    mats["LC-PFN + EcoStop gates"] = tierA.evaluate_family(jobs, pfn_tr, pfn_pl)
    fam["LC-PFN + EcoStop gates"] = pfn_pl

    out = {"frozen_utc": datetime.datetime.utcnow().isoformat() + "Z", "eps": tierA.EPS,
           "calibration_jobs": len(names), "targets": {}, "hashes": {f: sha(f) for f in FILES}}
    for rho, beta in TARGETS:
        key = f"rho={rho},beta={beta}"
        out["targets"][key] = {}
        for f, plist in fam.items():
            loss = mats[f][0]
            entry = {"pooled": None, "per_domain": {}}
            i = tierA.select(loss, rho, beta)
            entry["pooled"] = None if i is None else plist[i][0]
            for d in ("vision", "tabular", "nlp"):
                idx = np.where(dom_arr == d)[0]
                j = tierA.select(loss[:, idx], rho, beta)
                entry["per_domain"][d] = None if j is None else plist[j][0]
            out["targets"][key][f] = entry
    json.dump(out, open("b1_protocol.json", "w"), indent=2)
    print(json.dumps(out["targets"], indent=1))


if __name__ == "__main__":
    main()
