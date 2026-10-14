"""Where does the framework win? Pooled out-of-fold (same protocol as c1_framework_oof.py, 259 jobs, dataset-grouped folds) broken
down by domain / architecture / schedule / task type. EXPLORATORY: subgroups defined after the fact; bootstrap over dataset groups."""
import itertools, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from echostop import b1_eval as E, c1_eval as CE, tierA
from echostop.c1_framework import DOMS, NK, RULES, mats
from echostop.calibration import precompute_score_trace
from echostop.framework_v2 import BETA, hb_p, select, wsr_pvalue
RHOS, R, K = (0.15, 0.20), 20, 5
dev, eco_d, dom_d = tierA.load(); b1, dom_b = E.load_b1()
eco_b = {k: [precompute_score_trace(df, 0.5) for df in v] for k, v in b1.items()}
pfn_d = pickle.load(open("cache/lcpfn_traces.pkl", "rb")); pfn_b = pickle.load(open("cache/b1_lcpfn_traces.pkl", "rb"))
c1, _, _ = CE.load_logs("logs/c1/nlp"); pfn_c = pickle.load(open("cache/c1_lcpfn_traces.pkl", "rb"))
eco_c = {k: [precompute_score_trace(df, 0.5) for df in v] for k, v in c1.items()}
jobs, eco, pfn, dom, grp, meta = {}, {}, {}, [], [], []
def vis_arch(n): 
    for a in ("mobilenetv3", "vit_tiny", "resnet18_step", "vgg_small", "resnet18", "resnet9"):
        if n.endswith(a): return a
    return "other"
SCHED = {"mobilenetv3": "cosine+warmup", "vit_tiny": "cosine+warmup", "resnet18_step": "step decay", "vgg_small": "one-cycle"}
for k, v in dev.items():
    n = f"dev:{k}"; jobs[n], eco[n], pfn[n] = v, eco_d[k], pfn_d[k]; dom.append(dom_d[k]); grp.append(n)
    meta.append(dict(set="dev", arch=(vis_arch(k) if dom_d[k] == "vision" else ("DistilBERT" if dom_d[k] == "nlp" else "MLP")), sched="cosine" if dom_d[k] != "nlp" else "linear decay", task=""))
for k, v in b1.items():
    n = f"b1:{k}"; jobs[n], eco[n], pfn[n] = v, eco_b[k], pfn_b[k]; dom.append(dom_b[k]); grp.append(n)
    a = vis_arch(k) if dom_b[k] == "vision" else ("BERT-small" if dom_b[k] == "nlp" else "MLP (wide)")
    meta.append(dict(set="fresh", arch=a, sched=SCHED.get(a, "linear decay" if dom_b[k] == "nlp" else "cosine"), task=""))
PAIR = {"mnli", "snli", "qqp", "paws", "scitail", "vitaminc"}
for k in sorted(c1):
    n = f"c1:{k}"; ds, arm = k.split("|"); jobs[n], eco[n], pfn[n] = c1[k], eco_c[k], pfn_c[k]; dom.append("nlp"); grp.append("c1:" + ds)
    meta.append(dict(set="C1", arch="ELECTRA-small" if arm == "A" else "BERT-L6", sched="linear decay", task="text-pair" if ds in PAIR else "single text"))
dom, grp = np.array(dom), np.array(grp); meta = pd.DataFrame(meta); J = len(dom)
M = mats(jobs, eco, pfn); et = M[RULES[0]]["et"]
idxmap = {r: np.round(np.linspace(0, M[r]["P"] - 1, NK)).astype(int) for r in RULES}
assigns = [dict(zip(DOMS, c)) for c in itertools.product(RULES, repeat=3)]
comp = []
for a in assigns:
    L, ES = np.zeros((NK, J)), np.zeros((NK, J))
    for d in DOMS:
        m = dom == d; r = a[d]; L[:, m], ES[:, m] = M[r]["L"][idxmap[r]][:, m], M[r]["ES"][idxmap[r]][:, m]
    comp.append((L, ES))
def folds(rng):
    f = np.zeros(J, int)
    for d in DOMS:
        gs = np.unique(grp[dom == d]); rng.shuffle(gs); gf = {g: i % K for i, g in enumerate(gs)}
        for j in np.where(dom == d)[0]: f[j] = gf[grp[j]]
    return f
rng = np.random.default_rng(0); res = {}
for rho in RHOS:
    acc = {m: np.zeros((R, J)) for m in ("framework", "HB patience", "HB EcoStop", "framework risk", "HB patience risk")}
    for rep in range(R):
        fold = folds(rng)
        for f in range(K):
            te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
            best, bew = None, -1.0
            for i, (L, ES) in enumerate(comp):
                k = select(L[:, tr], rho, wsr_pvalue, beta=BETA / len(assigns))
                if k is not None and ES[k, tr].sum() / et[tr].sum() > bew: best, bew = (i, k), ES[k, tr].sum() / et[tr].sum()
            if best: acc["framework"][rep, te] = comp[best[0]][1][best[1], te]; acc["framework risk"][rep, te] = comp[best[0]][0][best[1], te]
            for name, r in (("HB patience", RULES[1]), ("HB EcoStop", RULES[0])):
                i = select(M[r]["L"][:, tr], rho, hb_p)
                if i is not None:
                    acc[name][rep, te] = M[r]["ES"][i, te]
                    if name == "HB patience": acc["HB patience risk"][rep, te] = M[r]["L"][i, te]
    res[rho] = {m: a.mean(0) for m, a in acc.items()}
uniq = np.unique(grp); gi = np.array([np.searchsorted(uniq, g) for g in grp])
W = np.random.default_rng(1).multinomial(len(uniq), np.ones(len(uniq)) / len(uniq), size=3000)[:, gi].astype(float)
def row(mask, label, rho):
    r = res[rho]; e = et[mask].sum()
    def ci(a, b):
        d = (a - b)[mask]; w = W[:, mask]; bs = (w @ d) / (w @ et[mask]) * 100
        return 100 * d.sum() / e, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    f, p, c = r["framework"][mask].sum() / e * 100, r["HB patience"][mask].sum() / e * 100, r["HB EcoStop"][mask].sum() / e * 100
    d1 = ci(r["framework"], r["HB patience"]); d2 = ci(r["HB EcoStop"], r["HB patience"])
    verdict = lambda t: "WIN" if t[1] > 0 else ("LOSS" if t[2] < 0 else "tie")
    return dict(rho=rho, subgroup=label, jobs=int(mask.sum()), energy_share=100 * e / et.sum(), framework_EW=f, patienceHB_EW=p,
                fw_minus_pat=d1[0], lo=d1[1], hi=d1[2], fw=verdict(d1), rule_minus_pat=d2[0], rlo=d2[1], rhi=d2[2], rule=verdict(d2),
                fw_risk=r["framework risk"][mask].mean(), pat_risk=r["HB patience risk"][mask].mean())
rows = []
for rho in RHOS:
    rows.append(row(np.ones(J, bool), "ALL", rho))
    for d in DOMS: rows.append(row(dom == d, f"domain: {d}", rho))
    for lab, col in (("arch", "arch"), ("schedule", "sched"), ("set", "set"), ("task", "task")):
        for v in meta[col].unique():
            if v == "": continue
            m = (meta[col] == v).values
            if m.sum() >= 6: rows.append(row(m, f"{lab}: {v}", rho))
df = pd.DataFrame(rows); df.to_csv("figures/tierA/subgroup_wins.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 30)
for rho in RHOS:
    print(f"\n=== rho = {rho}: framework and EcoStop rule vs calibrated patience (both under Hoeffding-Bentkus), EW pp [95% CI] ===")
    x = df[df.rho == rho]
    print(x.assign(fw=x.apply(lambda r: f"{r.fw_minus_pat:+5.1f} [{r.lo:+5.1f},{r.hi:+5.1f}] {r.fw}", axis=1),
                   rule=x.apply(lambda r: f"{r.rule_minus_pat:+5.1f} [{r.rlo:+5.1f},{r.rhi:+5.1f}] {r.rule}", axis=1))
          [["subgroup", "jobs", "energy_share", "framework_EW", "patienceHB_EW", "fw", "rule", "fw_risk"]].round(1).to_string(index=False))
