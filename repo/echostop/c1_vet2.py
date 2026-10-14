"""C1 vetting: for every candidate (hub id, config) stream the first rows of every split and report columns, label
type, label values seen, split names. No model is trained or evaluated here. Output: cache/c1_vet2.json"""
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
from datasets import get_dataset_split_names, load_dataset

CANDS = [
    ("community-datasets/financial_phrasebank", "sentences_allagree"), ("community-datasets/liar", None),
    ("community-datasets/health_fact", None), ("community-datasets/climate_fever", None), ("community-datasets/sick", None),
    ("community-datasets/emo", None), ("community-datasets/tweets_hate_speech_detection", None),
    ("community-datasets/scicite", None), ("community-datasets/medical_questions_pairs", None),
    ("community-datasets/hatexplain", None), ("community-datasets/ethos", "binary"), ("community-datasets/dyk", None),
    ("community-datasets/sentiment140", None), ("community-datasets/amazon_polarity", None),
    ("community-datasets/health_fact", None), ("community-datasets/hyperpartisan_news_detection", "byarticle"),
    ("rexarski/eli5_category", None), ("launch/open_question_type", None), ("tasksource/crowdflower", "economic-news"),
    ("tasksource/crowdflower", "political-media-bias"), ("tasksource/crowdflower", "sentiment-self-driving-cars"),
    ("tasksource/crowdflower", "gender-classifier"), ("tasksource/crowdflower", "us-economic-newspaper"),
    ("aps/super_glue", "multirc"), ("mteb/arxiv-clustering-s2s", None), ("mteb/reddit-clustering", None),
    ("mteb/stackexchange-clustering", None), ("mteb/twentynewsgroups-clustering", None),
    ("Setfit/sst2" if False else "SetFit/rte", None),
]
if len(sys.argv) > 1:
    CANDS = [c for c in CANDS if any(a in c[0] for a in sys.argv[1:])]


def vet(ident, cfg):
    t0 = time.time()
    try:
        splits = get_dataset_split_names(ident, cfg) if cfg else get_dataset_split_names(ident)
    except Exception as e:
        return {"id": ident, "cfg": cfg, "ok": False, "why": f"{type(e).__name__}: {str(e)[:110]}"}
    out = {"id": ident, "cfg": cfg, "ok": True, "splits": {}}
    for sp in splits:
        try:
            ds = load_dataset(ident, cfg, split=sp, streaming=True) if cfg else load_dataset(ident, split=sp, streaming=True)
            rows = []
            for i, r in enumerate(ds):
                rows.append(r)
                if i >= 199:
                    break
            feats = {k: str(v)[:60] for k, v in (ds.features or {}).items()}
            cols = list(rows[0].keys()) if rows else []
            ex = {k: (str(rows[0][k])[:70]) for k in cols} if rows else {}
            labcol = [k for k in cols if k in ("label", "labels", "category", "topic", "class", "gold", "type", "intent", "sentiment")]
            lv = sorted({str(r[labcol[0]]) for r in rows})[:12] if labcol else []
            out["splits"][sp] = {"cols": cols, "ex": ex, "label_col": labcol, "label_vals": lv, "feat": feats}
        except Exception as e:
            out["splits"][sp] = {"err": f"{type(e).__name__}: {str(e)[:110]}"}
    out["sec"] = round(time.time() - t0, 1)
    return out


def main():
    res = []
    for ident, cfg in CANDS:
        r = vet(ident, cfg)
        res.append(r)
        if not r["ok"]:
            print(f"X {ident} [{cfg}] {r['why']}", flush=True)
            continue
        sp = r["splits"]
        first = next((v for v in sp.values() if "cols" in v), None)
        if first is None:
            print(f"X {ident} [{cfg}] no readable split: {list(sp.values())[0]}", flush=True)
            continue
        print(f"OK {ident} [{cfg}] splits={list(sp)} cols={first['cols']} label={first['label_col']} vals={first['label_vals']}", flush=True)
        time.sleep(0.3)
    json.dump(res, open("cache/c1_vet2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
