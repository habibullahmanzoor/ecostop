"""Real pretrained LC-PFN traces for the B1 fresh runs (GPU; run only when no energy-metered job is running)."""
import os, pickle, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from echostop import b1_eval as E
from echostop.lcpfn_real import LCPFN, _patch_layers, pfn_trace

if __name__ == "__main__":
    dev = "cuda"
    jobs, dom = E.load_b1()
    model = LCPFN().to(dev); _patch_layers(model); model.eval()
    out = {}
    for i, (k, dfs) in enumerate(jobs.items()):
        out[k] = [pfn_trace(model, df, dev) for df in dfs]
        print(f"{i+1}/{len(jobs)} {k}", flush=True)
    pickle.dump(out, open("cache/b1_lcpfn_traces.pkl", "wb"))
    print("done")
