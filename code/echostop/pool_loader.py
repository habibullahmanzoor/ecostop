"""Shared loader for the pooled 259-job analyses: returns rule matrices, domains, dataset groups and total energies."""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E, c1_eval as CE, tierA
from echostop.c1_framework import RULES, mats
from echostop.calibration import precompute_score_trace


def load_pool():
    cache = "cache/pool_mats.pkl"
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    dev, eco_d, dom_d = tierA.load(); b1, dom_b = E.load_b1()
    eco_b = {k: [precompute_score_trace(df, 0.5) for df in v] for k, v in b1.items()}
    pfn_d = pickle.load(open("cache/lcpfn_traces.pkl", "rb")); pfn_b = pickle.load(open("cache/b1_lcpfn_traces.pkl", "rb"))
    c1, _, _ = CE.load_logs("logs/c1/nlp"); pfn_c = pickle.load(open("cache/c1_lcpfn_traces.pkl", "rb"))
    eco_c = {k: [precompute_score_trace(df, 0.5) for df in v] for k, v in c1.items()}
    jobs, eco, pfn, dom, grp, tag = {}, {}, {}, [], [], []
    for k, v in dev.items():
        n = f"dev:{k}"; jobs[n], eco[n], pfn[n] = v, eco_d[k], pfn_d[k]; dom.append(dom_d[k]); grp.append(n); tag.append("dev")
    for k, v in b1.items():
        n = f"b1:{k}"; jobs[n], eco[n], pfn[n] = v, eco_b[k], pfn_b[k]; dom.append(dom_b[k]); grp.append(n); tag.append("b1")
    for k in sorted(c1):
        n = f"c1:{k}"; jobs[n], eco[n], pfn[n] = c1[k], eco_c[k], pfn_c[k]; dom.append("nlp"); grp.append("c1:" + k.split("|")[0]); tag.append("c1")
    M = mats(jobs, eco, pfn)
    out = (M, np.array(dom), np.array(grp), np.array(tag), M[RULES[0]]["et"])
    pickle.dump(out, open(cache, "wb"))
    return out
