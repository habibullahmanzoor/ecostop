"""Run the pilot gate on all 12 job-seeds against the recipe-matched control and write cache/pilot_verdict.json."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop.pilot_cool_analysis import analyse

v = analyse({0, 1})
os.makedirs("cache", exist_ok=True)
json.dump({"flags": [bool(x) for x in v], "continue": bool(all(v))}, open("cache/pilot_verdict.json", "w"))
print("VERDICT:", "CONTINUE to S1" if all(v) else "STOP")
