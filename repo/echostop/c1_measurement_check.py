"""C1 measurement-validity check (outcome-blind; written 2026-10-13 BEFORE the evaluation is run).

Rule: a run is measurement-invalid if its longest epoch lasted > 5x the run's median epoch AND > 60 s. Such an epoch spans a
machine suspend (Windows Modern Standby), and the NVML energy integral across the suspend is an artefact (hours of
elapsed time at the last power reading). The rule looks only at epoch durations - never at accuracy, stopping epochs or
savings. Invalid runs are MOVED (not deleted) to logs/c1/quarantine/ and re-run once with the same seed, as the analysis
plan already provides for runs that fail for technical reasons.
usage: python -m echostop.c1_measurement_check [--quarantine]"""
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

DIR, QDIR = "logs/c1/nlp", "logs/c1/quarantine"


def flagged():
    out = []
    for p in sorted(glob.glob(f"{DIR}/*.csv")):
        if not os.path.exists(p.replace(".csv", ".meta.json")):
            continue
        d = pd.read_csv(p)["duration_seconds"].values
        if d.max() > 5 * np.median(d) and d.max() > 60:
            out.append((os.path.basename(p)[:-4], float(np.median(d)), float(d.max())))
    return out


if __name__ == "__main__":
    f = flagged()
    print(f"{len(f)} measurement-invalid runs")
    for n, med, mx in f:
        print(f"  {n}: median epoch {med:.1f} s, longest {mx:.0f} s")
    if "--quarantine" in sys.argv and f:
        os.makedirs(QDIR, exist_ok=True)
        for n, _, _ in f:
            for ext in (".csv", ".meta.json"):
                shutil.move(f"{DIR}/{n}{ext}", f"{QDIR}/{n}{ext}")
        print(f"moved to {QDIR}")
