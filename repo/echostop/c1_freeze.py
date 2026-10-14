"""Freeze C1: calibrate every rule on the EXISTING jobs only, hash the code/registry/plan, write c1_protocol.json.
Run once, BEFORE any C1 training run exists. Refuses to overwrite a protocol and refuses if C1 logs exist.
--dry: development jobs only, writes cache/c1_protocol_dryrun.json (no hashes) for testing c1_eval.py on mock data."""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import pickle
import platform
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E
from echostop import c1_rules as CR
from echostop import tierA
from echostop.calibration import precompute_score_trace
from echostop.framework_v2 import select as bet_select, wsr_pvalue

FILES = ["echostop/c1_registry.py", "echostop/c1_data.py", "echostop/c1_train.py", "echostop/c1_rules.py", "echostop/c1_eval.py",
         "echostop/c1_freeze.py", "c1_manifest.json", "c1_analysis_plan.md",
         "echostop/methods.py", "echostop/calibration.py", "echostop/ablation.py", "echostop/tierA.py", "echostop/fv2_srgates.py",
         "echostop/lcpfn_real.py", "echostop/framework_v2.py", "echostop/b1_eval.py", "echostop/b1_train.py",
         "echostop/train_nlp.py", "echostop/train.py", "echostop/energy.py"]
RHOS = (0.15, 0.20)
EMP = (0.10, 0.15)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if not a.dry:
        if os.path.exists("c1_protocol.json"):
            raise SystemExit("c1_protocol.json exists: protocol is frozen.")
        if glob.glob("logs/c1/nlp/*.csv"):
            raise SystemExit("C1 training logs exist: refusing to freeze after training has started.")
    dev, eco_d, dom_d = tierA.load()
    pf_d = pickle.load(open("cache/lcpfn_traces.pkl", "rb"))
    jobs = {f"dev:{k}": v for k, v in dev.items()}
    tr = {f"dev:{k}": [{"eco": eco_d[k][i], "pfn": pf_d[k][i]} for i in range(len(v))] for k, v in dev.items()}
    dom = {f"dev:{k}": dom_d[k] for k in dev}
    if not a.dry:
        fr, fdom = E.load_b1()
        pf_f = pickle.load(open("cache/b1_lcpfn_traces.pkl", "rb"))
        for k, v in fr.items():
            e = [precompute_score_trace(df, 0.5) for df in v]
            jobs[f"b1:{k}"] = v
            tr[f"b1:{k}"] = [{"eco": e[i], "pfn": pf_f[k][i]} for i in range(len(v))]
            dom[f"b1:{k}"] = fdom[k]
    names = list(jobs)
    d = np.array([dom[k] for k in names])
    nlp = d == "nlp"
    fams = CR.families(True)
    frozen = {}
    for f, pl in fams.items():
        L, S, ES, et = tierA.evaluate_family(jobs, tr, pl)
        labs = [lab for lab, _ in pl]
        for rho in RHOS:
            i = tierA.select(L, rho, 0.05)
            frozen.setdefault(f"HB pooled rho={rho}", {})[f] = None if i is None else labs[i]
            j = bet_select(L, rho, wsr_pvalue)
            frozen.setdefault(f"betting pooled rho={rho}", {})[f] = None if j is None else labs[j]
        for r in EMP:
            k = CR.fixed_sequence_empirical(L[:, nlp], r)
            frozen.setdefault(f"empirical NLP risk<={r}", {})[f] = None if k < 0 else labs[k]
    out = {"frozen_utc": datetime.datetime.utcnow().isoformat() + "Z", "calibration_jobs": len(names), "nlp_calibration_jobs": int(nlp.sum()),
           "frozen": frozen, "python": platform.python_version()}
    try:
        import lcpfn, torch, transformers
        out["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
    except Exception:
        pass
    if a.dry:
        json.dump(out, open("cache/c1_protocol_dryrun.json", "w"), indent=1)
        print("dry-run protocol written; calibration jobs", len(names), "nlp", int(nlp.sum()))
    else:
        out["hashes"] = {f: sha(f) for f in FILES}
        json.dump(out, open("c1_protocol.json", "w"), indent=1)
        print("FROZEN. calibration jobs", len(names), "nlp", int(nlp.sum()))
    for tag, t in frozen.items():
        print(tag, {k: v for k, v in t.items()})


if __name__ == "__main__":
    main()
