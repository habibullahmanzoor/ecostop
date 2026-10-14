"""C1 vetting: for every candidate (hub id, config) stream the first rows of every split and report columns, label
type, label values seen, split names. No model is trained or evaluated here. Output: cache/c1_vet.json"""
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
from datasets import get_dataset_split_names, load_dataset

CANDS = [
    ("SetFit/20_newsgroups", None), ("SetFit/amazon_counterfactual_en", None), ("SetFit/toxic_conversations", None),
    ("SetFit/ade_corpus_v2_classification", None), ("SetFit/onestop_english", None),
    ("SetFit/amazon_massive_intent_en-US", None), ("SetFit/amazon_massive_scenario_en-US", None),
    ("SetFit/amazon_reviews_multi_en", None), ("SetFit/sentiment140", None), ("SetFit/insincere-questions", None),
    ("SetFit/mnli", None), ("SetFit/qqp", None), ("SetFit/ethos", None), ("SetFit/hate_speech18", None),
    ("SetFit/yahoo_answers_topics", None), ("SetFit/dbpedia_14", None), ("SetFit/tweets_hate_speech_detection", None),
    ("fancyzhx/dbpedia_14", None), ("community-datasets/yahoo_answers_topics", None),
    ("zeroshot/twitter-financial-news-sentiment", None), ("zeroshot/twitter-financial-news-topic", None),
    ("cardiffnlp/tweet_topic_single", None), ("mteb/tweet_sentiment_extraction", None),
    ("mteb/mtop_domain", "en"), ("mteb/mtop_intent", "en"), ("mteb/toxic_conversations_50k", None),
    ("Sp1786/multiclass-sentiment-analysis-dataset", None), ("heegyu/news-category-dataset", None),
    ("okite97/news-data", None), ("google-research-datasets/paws", "labeled_final"),
    ("nyu-mll/glue", "mnli"), ("nyu-mll/glue", "qqp"), ("stanfordnlp/snli", None), ("allenai/scitail", "snli_format"),
    ("takala/financial_phrasebank", "sentences_allagree"), ("nickmuchi/financial-classification", None),
    ("FinanceInc/auditor_sentiment", None), ("TimSchopf/medical_abstracts", None),
    ("armanc/pubmed-rct20k", None), ("knowledgator/events_classification_biotech", None),
    ("OxAISH-AL-LLM/wiki_toxic", None), ("badmatr11x/hate-offensive-speech", None),
    ("thefrankhsu/hate_speech_twitter", None), ("bdotloh/empathetic-dialogues-contexts", None),
    ("sealuzh/app_reviews", None), ("juliensimon/amazon-shoe-reviews", None),
    ("jonaskoenig/Questions-vs-Statements-Classification", None), ("aps/super_glue", "wic"),
    ("lmsys/toxic-chat", "toxicchat0124"), ("skg/toxigen-data", "annotated"), ("tals/vitaminc", None),
    ("fhamborg/news_sentiment_newsmtsc", "mt"), ("Hello-SimpleAI/HC3", "all"), ("ma2za/many_emotions", "raw"),
    ("tyqiangz/multilingual-sentiments", "english"), ("cardiffnlp/tweet_sentiment_multilingual", "english"),
    ("pietrolesci/pubmed-200k-rct", None), ("ucirvine/reuters21578", None), ("rungalileo/20_Newsgroups_Fixed", None),
    ("mrm8488/ImageNet1K-val", None), ("dmitva/human_ai_generated_text", None), ("hendrycks/ethics", "commonsense"),
    ("mteb/amazon_reviews_multi", "en"), ("kuroneko5943/amz20", None), ("Yelp/yelp_review_full", None),
    ("ccdv/patent-classification", None), ("coastalcph/lex_glue", "ledgar"), ("mteb/medrxiv-clustering-p2p", None),
    ("ag2428/..." if False else "tasksource/crowdflower", "tweet_global_warming"),
    ("tasksource/crowdflower", "airline-sentiment"), ("tasksource/crowdflower", "political-media-message"),
    ("tasksource/crowdflower", "emotion"), ("tasksource/crowdflower", "political-media-audience"),
    ("tasksource/crowdflower", "economy-relevance"), ("tasksource/crowdflower", "corporate-messaging"),
    ("tasksource/crowdflower", "text_emotion"), ("tasksource/crowdflower", "brands-and-product-emotions"),
    ("tasksource/crowdflower", "disasters"),
    ("SetFit/sst2_few", None) if False else ("SetFit/emotion", None),
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
    json.dump(res, open("cache/c1_vet.json", "w"), indent=1)


if __name__ == "__main__":
    main()
