"""C1: confirmatory study of EcoStop-SR on short-schedule NLP fine-tuning. Job registry, fixed BEFORE any C1 job is
trained and hash-locked by c1_freeze.py.

Why this study exists. EcoStop-SR (schedule-relative gates, echostop/fv2_srgates.py) was found AFTER seeing the 142
development jobs and the 35 B1 jobs; its advantage over calibrated patience is significant on development jobs and not
significant on the 8 B1 NLP jobs. C1 is a new, larger test on short fine-tuning schedules, with models, schedule lengths
and datasets absent from everything the rule was designed on.

Datasets. 43 candidate English text-classification datasets on the Hugging Face Hub, curated from public Hub listings and
vetted ONLY by loading them (columns, splits, label type); no model has been trained or evaluated on any of them.
Datasets already in the 142-job corpus or the B1 set are excluded. Each candidate then passes fixed size rules
(c1_data.py): >= 2,500 training rows, >= 800 test rows, 2..100 classes. Candidates that fail a rule are dropped
automatically and the drop is recorded in c1_manifest.json; no other dataset is added or removed.

Arms. Two recipe arms per dataset (a job = dataset x arm), 3 seeds each:
  A  ELECTRA-small discriminator, 8 epochs, lr 3e-4 (the value recommended for the small model in the ELECTRA paper;
     a smoke test with lr 1e-4 on 20 Newsgroups, seed 0, was still climbing at epoch 8, so 1e-4 was replaced BEFORE
     the freeze and before any stopping rule was evaluated on any C1 data)
  B  BERT uncased L-6 H-512 (Google miniature), 12 epochs, lr 5e-5
Both: batch 32, AdamW (wd 0.01), linear warm-up over 10% of steps then linear decay to 0, gradient clipping 1.0, max
length 128, fp32. Neither model nor either length (8, 12) occurs in the calibration pool (DistilBERT 10 epochs; BERT-small
L4 15 epochs).
"""

SPLIT_SEED = 0                      # fixed data split per dataset (the run seed only changes init and batch order)
SEEDS = [0, 1, 2]
MAX_LEN = 128
TRAIN_CAP, VAL_CAP, TEST_CAP = 8000, 2000, 4000
MIN_TRAIN, MIN_TEST = 2500, 800
MIN_CLASSES, MAX_CLASSES = 2, 100

ARMS = {
    "A": dict(model="google/electra-small-discriminator", epochs=8, lr=3e-4, bs=32, wd=0.01, warm=0.1, clip=1.0),
    "B": dict(model="google/bert_uncased_L-6_H-512_A-8", epochs=12, lr=5e-5, bs=32, wd=0.01, warm=0.1, clip=1.0),
}

# name: dict(id, cfg, text=[1 or 2 string columns], label, train, val (None -> carve 10% of train), test (None -> carve 20%))
D = dict
DATASETS = {
    "newsgroups20":   D(id="SetFit/20_newsgroups", cfg=None, text=["text"], label="label", train="train", val=None, test="test"),
    "amazon_cf":      D(id="SetFit/amazon_counterfactual_en", cfg=None, text=["text"], label="label", train="train", val=None, test="test"),
    "toxic_conv":     D(id="SetFit/toxic_conversations", cfg=None, text=["text"], label="label", train="train", val=None, test="test"),
    "ade_corpus":     D(id="SetFit/ade_corpus_v2_classification", cfg=None, text=["text"], label="label", train="train", val=None, test="test"),
    "massive_intent": D(id="SetFit/amazon_massive_intent_en-US", cfg=None, text=["text"], label="label", train="train", val="validation", test="test"),
    "massive_scen":   D(id="SetFit/amazon_massive_scenario_en-US", cfg=None, text=["text"], label="label", train="train", val="validation", test="test"),
    "amazon_stars":   D(id="SetFit/amazon_reviews_multi_en", cfg=None, text=["text"], label="label", train="train", val="validation", test="test"),
    "insincere_q":    D(id="SetFit/insincere-questions", cfg=None, text=["text"], label="label", train="train", val=None, test="test"),
    "mnli":           D(id="SetFit/mnli", cfg=None, text=["text1", "text2"], label="label", train="train", val=None, test="validation"),
    "qqp":            D(id="SetFit/qqp", cfg=None, text=["text1", "text2"], label="label", train="train", val=None, test="validation"),
    "hate18":         D(id="SetFit/hate_speech18", cfg=None, text=["text"], label="label", train="train", val=None, test=None),
    "dbpedia14":      D(id="fancyzhx/dbpedia_14", cfg=None, text=["title", "content"], label="label", train="train", val=None, test="test"),
    "yahoo_topics":   D(id="community-datasets/yahoo_answers_topics", cfg=None, text=["question_title", "question_content"], label="topic", train="train", val=None, test="test"),
    "fin_tw_sent":    D(id="zeroshot/twitter-financial-news-sentiment", cfg=None, text=["text"], label="label", train="train", val=None, test="validation"),
    "fin_tw_topic":   D(id="zeroshot/twitter-financial-news-topic", cfg=None, text=["text"], label="label", train="train", val=None, test="validation"),
    "tweet_sent_ext": D(id="mteb/tweet_sentiment_extraction", cfg=None, text=["text"], label="label", train="train", val=None, test="test"),
    "multiclass_sent": D(id="Sp1786/multiclass-sentiment-analysis-dataset", cfg=None, text=["text"], label="label", train="train", val="validation", test="test"),
    "news_category":  D(id="heegyu/news-category-dataset", cfg=None, text=["headline", "short_description"], label="category", train="train", val=None, test=None),
    "okite_news":     D(id="okite97/news-data", cfg=None, text=["Title", "Excerpt"], label="Category", train="train", val=None, test="test"),
    "paws":           D(id="google-research-datasets/paws", cfg="labeled_final", text=["sentence1", "sentence2"], label="label", train="train", val="validation", test="test"),
    "snli":           D(id="stanfordnlp/snli", cfg=None, text=["premise", "hypothesis"], label="label", train="train", val="validation", test="test"),
    "scitail":        D(id="allenai/scitail", cfg="snli_format", text=["sentence1", "sentence2"], label="gold_label", train="train", val="validation", test="test"),
    "fin_class":      D(id="nickmuchi/financial-classification", cfg=None, text=["text"], label="labels", train="train", val=None, test="test"),
    "auditor_sent":   D(id="FinanceInc/auditor_sentiment", cfg=None, text=["sentence"], label="label", train="train", val=None, test="test"),
    "medical_abs":    D(id="TimSchopf/medical_abstracts", cfg=None, text=["medical_abstract"], label="condition_label", train="train", val=None, test="test"),
    "pubmed_rct":     D(id="armanc/pubmed-rct20k", cfg=None, text=["text"], label="label", train="train", val="validation", test="test"),
    "hate_twitter":   D(id="thefrankhsu/hate_speech_twitter", cfg=None, text=["tweet"], label="label", train="train", val=None, test="test"),
    "empathetic":     D(id="bdotloh/empathetic-dialogues-contexts", cfg=None, text=["situation"], label="emotion", train="train", val="validation", test="test"),
    "app_reviews":    D(id="sealuzh/app_reviews", cfg=None, text=["review"], label="star", train="train", val=None, test=None),
    "shoe_reviews":   D(id="juliensimon/amazon-shoe-reviews", cfg=None, text=["text"], label="labels", train="train", val=None, test="test"),
    "questions_stmt": D(id="jonaskoenig/Questions-vs-Statements-Classification", cfg=None, text=["text"], label="labels", train="train", val="validation", test="test"),
    "toxic_chat":     D(id="lmsys/toxic-chat", cfg="toxicchat0124", text=["user_input"], label="toxicity", train="train", val=None, test="test"),
    "vitaminc":       D(id="tals/vitaminc", cfg=None, text=["claim", "evidence"], label="label", train="train", val="validation", test="test"),
    "patent":         D(id="ccdv/patent-classification", cfg=None, text=["text"], label="label", train="train", val="validation", test="test"),
    "ledgar":         D(id="coastalcph/lex_glue", cfg="ledgar", text=["text"], label="label", train="train", val="validation", test="test"),
    "cf_airline":     D(id="tasksource/crowdflower", cfg="airline-sentiment", text=["text"], label="label", train="train", val=None, test=None),
    "cf_pol_message": D(id="tasksource/crowdflower", cfg="political-media-message", text=["text"], label="label", train="train", val=None, test=None),
    "cf_pol_audience": D(id="tasksource/crowdflower", cfg="political-media-audience", text=["text"], label="label", train="train", val=None, test=None),
    "cf_pol_bias":    D(id="tasksource/crowdflower", cfg="political-media-bias", text=["text"], label="label", train="train", val=None, test=None),
    "cf_corporate":   D(id="tasksource/crowdflower", cfg="corporate-messaging", text=["text"], label="label", train="train", val=None, test=None),
    "cf_emotion":     D(id="tasksource/crowdflower", cfg="text_emotion", text=["text"], label="label", train="train", val=None, test=None),
    "cf_warming":     D(id="tasksource/crowdflower", cfg="tweet_global_warming", text=["text"], label="label", train="train", val=None, test=None),
    "cf_economic":    D(id="tasksource/crowdflower", cfg="economic-news", text=["text"], label="label", train="train", val=None, test=None),
}
