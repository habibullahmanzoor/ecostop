"""Freeze S1: certify every method on the CALIBRATION jobs only, write s1_protocol.json (thresholds + hashes),
BEFORE any test job is trained. Refuses to overwrite an existing protocol and refuses to freeze with < 14 jobs
(below that the Hoeffding-Bentkus test cannot certify rho <= 0.25 even with zero violations)."""
import datetime
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import s1_analysis as A

if os.path.exists("s1_protocol.json"):
    raise SystemExit("s1_protocol.json exists: frozen already")
cal = A.collect("cal")
n = len(cal)
print(f"calibration jobs: {n}")
if n < 14:
    raise SystemExit(f"only {n} calibration jobs (<14): cannot certify; not freezing")
names, tab = A.method_tables(cal)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
out = {"frozen_utc": datetime.datetime.utcnow().isoformat() + "Z", "n_cal_jobs": n, "cal_jobs": names,
       "registry_sha256": sha("echostop/s1_registry.py"), "analysis_sha256": sha("echostop/s1_analysis.py"),
       "thresholds": {}}
for rho in (0.20, 0.25):
    out["thresholds"][f"rho={rho},beta=0.05"] = A.certify(tab, rho)
json.dump(out, open("s1_protocol.json", "w"), indent=2)
print(json.dumps(out["thresholds"], indent=1))
