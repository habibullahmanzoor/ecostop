"""Real-data NLP leg (plan section 16, "NLP task" + calibration-corpus pool):
SST-2, rotten_tomatoes, TREC via Hugging Face `datasets`, DistilBERT
tokenizer. Three real text-classification datasets, deliberately different
in size and task shape (binary sentiment on movie snippets/full sentences,
6-way coarse question classification) to test whether EcoStop's NLP-leg
result generalizes beyond a single dataset -- session finding: the original
SST-2-only leg was n=1 for the whole modality, too thin to claim
generalization from.

TEST must be untouched by both training and checkpoint selection everywhere
in this project (data.py/tabular_data.py convention); see each loader for
how that's satisfied given what splits each dataset actually ships.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


@dataclass
class NLPBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int


def _make_loaders(train_ds, val_ds, test_ds, tokenizer, text_col, label_col,
                   remove_cols, seed, batch_size, text_col2: str | None = None):
    def _tok(batch):
        if text_col2 is not None:
            return tokenizer(batch[text_col], batch[text_col2], truncation=True, max_length=MAX_LENGTH)
        return tokenizer(batch[text_col], truncation=True, max_length=MAX_LENGTH)

    def _prep(ds):
        ds = ds.map(_tok, batched=True, remove_columns=remove_cols)
        if label_col != "labels":
            ds = ds.rename_column(label_col, "labels")
        ds.set_format(type="torch")
        return ds

    train_ds, val_ds, test_ds = _prep(train_ds), _prep(val_ds), _prep(test_ds)
    collator = DataCollatorWithPadding(tokenizer)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collator, generator=g)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collator)
    return train_loader, val_loader, test_loader


def get_sst2(seed: int, batch_size: int = 32) -> NLPBundle:
    """GLUE's official test split has hidden labels (-1, leaderboard-only) --
    unusable for our audited test-set metric. Use the labeled validation
    split (872 examples) as TEST, and carve our own validation split out of
    the labeled train set (67349 examples) for checkpoint selection."""
    raw = load_dataset("nyu-mll/glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    val_size = 4000

    train_full = raw["train"].shuffle(seed=seed)
    val_ds = train_full.select(range(val_size))
    train_ds = train_full.select(range(val_size, len(train_full)))
    test_ds = raw["validation"]

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "sentence", "label",
        ["sentence", "idx"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=2)


def get_rotten_tomatoes(seed: int, batch_size: int = 32) -> NLPBundle:
    """Ships real, fully-labeled train/validation/test splits already --
    no carving needed, unlike SST-2."""
    raw = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = raw["train"].shuffle(seed=seed)
    val_ds = raw["validation"]
    test_ds = raw["test"]

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "text", "label",
        ["text"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=2)


def get_trec(seed: int, batch_size: int = 32) -> NLPBundle:
    """6-way coarse question classification (DESC/ENTY/HUM/LOC/NUM/ABBR) --
    a different task shape from binary sentiment, deliberately. Ships only
    train (5452) + test (500); carve a validation split out of train."""
    raw = load_dataset("SetFit/TREC-QC")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    val_size = 500

    train_full = raw["train"].shuffle(seed=seed)
    val_ds = train_full.select(range(val_size))
    train_ds = train_full.select(range(val_size, len(train_full)))
    test_ds = raw["test"]

    remove_cols = [c for c in train_full.column_names if c != "label_coarse"]
    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "text", "label_coarse",
        remove_cols, seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=6)


def get_emotion(seed: int, batch_size: int = 32) -> NLPBundle:
    """6-class emotion classification (sadness/joy/love/anger/fear/surprise)
    -- ships real, fully-labeled train/validation/test already."""
    raw = load_dataset("dair-ai/emotion")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = raw["train"].shuffle(seed=seed)
    val_ds = raw["validation"]
    test_ds = raw["test"]

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "text", "label",
        ["text"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=6)


def get_tweet_eval_sentiment(seed: int, batch_size: int = 32) -> NLPBundle:
    """3-class tweet sentiment (negative/neutral/positive) -- a different
    text register (short, informal, real tweets) from the other legs."""
    raw = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = raw["train"].shuffle(seed=seed)
    val_ds = raw["validation"]
    test_ds = raw["test"]

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "text", "label",
        ["text"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=3)


def get_poem_sentiment(seed: int, batch_size: int = 32) -> NLPBundle:
    """4-class sentiment on poem verses (negative/positive/no_impact/mixed)
    -- the smallest dataset in the NLP leg (892 train), a different genre
    and much shorter training set than everything else here."""
    raw = load_dataset("google-research-datasets/poem_sentiment")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = raw["train"].shuffle(seed=seed)
    val_ds = raw["validation"]
    test_ds = raw["test"]

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "verse_text", "label",
        ["verse_text", "id"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=4)


def get_banking77(seed: int, batch_size: int = 32) -> NLPBundle:
    """77-class intent classification on banking customer queries --
    deliberately the highest-class-count job in the whole corpus, real
    fine-grained intent data. Ships only train/test; carve a validation
    split out of train."""
    raw = load_dataset("legacy-datasets/banking77")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    val_size = 1000

    train_full = raw["train"].shuffle(seed=seed)
    val_ds = train_full.select(range(val_size))
    train_ds = train_full.select(range(val_size, len(train_full)))
    test_ds = raw["test"]

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "text", "label",
        ["text"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=77)


def _get_glue_task(config: str, text_col: str, label_col: str, num_classes: int,
                    seed: int, batch_size: int, text_col2: str | None = None) -> NLPBundle:
    """Shared loader for GLUE tasks (session second-round addition, prompted
    by the paper's own §6.3.1/§8 domain-imbalance flag): GLUE's official
    test split has hidden labels (-1, leaderboard-only) for every task, not
    just SST-2 -- same fix as get_sst2: use the labeled validation split as
    TEST, carve our own validation split out of the labeled train set."""
    raw = load_dataset("nyu-mll/glue", config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    val_size = min(1000, len(raw["train"]) // 10)

    train_full = raw["train"].shuffle(seed=seed)
    val_ds = train_full.select(range(val_size))
    train_ds = train_full.select(range(val_size, len(train_full)))
    test_ds = raw["validation"]

    cols = [text_col] + ([text_col2] if text_col2 else []) + ["idx"]
    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, text_col, label_col,
        cols, seed, batch_size, text_col2=text_col2,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=num_classes)


def get_cola(seed: int, batch_size: int = 32) -> NLPBundle:
    """Binary grammatical-acceptability judgment, 8.5k train -- a very
    different task shape from sentiment/topic classification."""
    return _get_glue_task("cola", "sentence", "label", 2, seed, batch_size)


def get_mrpc(seed: int, batch_size: int = 32) -> NLPBundle:
    """Binary paraphrase detection on sentence pairs, 3.7k train."""
    return _get_glue_task("mrpc", "sentence1", "label", 2, seed, batch_size, text_col2="sentence2")


def get_rte(seed: int, batch_size: int = 32) -> NLPBundle:
    """Binary textual entailment on sentence pairs, 2.5k train -- the
    smallest GLUE task added here."""
    return _get_glue_task("rte", "sentence1", "label", 2, seed, batch_size, text_col2="sentence2")


def get_qnli(seed: int, batch_size: int = 32) -> NLPBundle:
    """Binary question/sentence entailment, 105k train -- the largest job
    added in this batch, deliberately kept for size diversity."""
    return _get_glue_task("qnli", "question", "label", 2, seed, batch_size, text_col2="sentence")


def get_wnli(seed: int, batch_size: int = 32) -> NLPBundle:
    """Binary coreference/entailment, 635 train -- the smallest job in the
    whole corpus, a genuine stress test of the schedule-relative
    safeguard (§4.2b) at an extreme."""
    return _get_glue_task("wnli", "sentence1", "label", 2, seed, batch_size, text_col2="sentence2")


def _get_tweet_eval_task(config: str, num_classes: int, seed: int, batch_size: int) -> NLPBundle:
    """Shared loader for tweet_eval sub-tasks beyond sentiment (session
    second-round addition): ships real train/validation/test splits, same
    convention as get_tweet_eval_sentiment."""
    raw = load_dataset("cardiffnlp/tweet_eval", config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = raw["train"].shuffle(seed=seed)
    val_ds = raw["validation"]
    test_ds = raw["test"]

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "text", "label",
        ["text"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=num_classes)


def get_tweet_eval_irony(seed: int, batch_size: int = 32) -> NLPBundle:
    return _get_tweet_eval_task("irony", 2, seed, batch_size)


def get_tweet_eval_hate(seed: int, batch_size: int = 32) -> NLPBundle:
    return _get_tweet_eval_task("hate", 2, seed, batch_size)


def get_tweet_eval_offensive(seed: int, batch_size: int = 32) -> NLPBundle:
    return _get_tweet_eval_task("offensive", 2, seed, batch_size)


def get_tweet_eval_emoji(seed: int, batch_size: int = 32) -> NLPBundle:
    """20-way emoji prediction, 45k train -- the highest-class-count job in
    the whole tweet_eval family besides banking77."""
    return _get_tweet_eval_task("emoji", 20, seed, batch_size)


def get_tweet_eval_stance_abortion(seed: int, batch_size: int = 32) -> NLPBundle:
    return _get_tweet_eval_task("stance_abortion", 3, seed, batch_size)


def get_tweet_eval_stance_atheism(seed: int, batch_size: int = 32) -> NLPBundle:
    return _get_tweet_eval_task("stance_atheism", 3, seed, batch_size)


def get_tweet_eval_stance_climate(seed: int, batch_size: int = 32) -> NLPBundle:
    return _get_tweet_eval_task("stance_climate", 3, seed, batch_size)


def get_tweet_eval_stance_feminist(seed: int, batch_size: int = 32) -> NLPBundle:
    return _get_tweet_eval_task("stance_feminist", 3, seed, batch_size)


def get_tweet_eval_stance_hillary(seed: int, batch_size: int = 32) -> NLPBundle:
    return _get_tweet_eval_task("stance_hillary", 3, seed, batch_size)


def get_imdb(seed: int, batch_size: int = 32) -> NLPBundle:
    """Binary movie-review sentiment, 25k train -- much longer documents
    than anything else in the NLP leg (full reviews, not snippets),
    truncated to MAX_LENGTH like every other job here."""
    raw = load_dataset("stanfordnlp/imdb")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    val_size = 2500

    train_full = raw["train"].shuffle(seed=seed)
    val_ds = train_full.select(range(val_size))
    train_ds = train_full.select(range(val_size, len(train_full)))
    test_ds = raw["test"]

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "text", "label",
        ["text"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=2)


def get_ag_news(seed: int, batch_size: int = 32) -> NLPBundle:
    """4-way news-topic classification, 120k train -- the largest single
    job in the whole corpus, deliberately kept for size diversity (§8)."""
    raw = load_dataset("fancyzhx/ag_news")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    val_size = 5000

    train_full = raw["train"].shuffle(seed=seed)
    val_ds = train_full.select(range(val_size))
    train_ds = train_full.select(range(val_size, len(train_full)))
    test_ds = raw["test"]

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "text", "label",
        ["text"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=4)


def get_yelp_polarity(seed: int, batch_size: int = 32) -> NLPBundle:
    """Binary Yelp review sentiment. The official corpus is enormous
    (560k train / 38k test) -- capped to keep training time comparable to
    the corpus's other large jobs (AG News, QNLI), added specifically to
    help close the NLP domain-alone certification gap (§6.3.1)."""
    raw = load_dataset("fancyzhx/yelp_polarity")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_size, val_size, test_size = 60_000, 3_000, 5_000

    train_full = raw["train"].shuffle(seed=seed)
    val_ds = train_full.select(range(val_size))
    train_ds = train_full.select(range(val_size, val_size + train_size))
    test_ds = raw["test"].shuffle(seed=0).select(range(test_size))

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "text", "label",
        ["text"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=2)


def get_amazon_polarity(seed: int, batch_size: int = 32) -> NLPBundle:
    """Binary Amazon review sentiment. The official corpus is enormous
    (3.6M train / 400k test) -- capped the same way as Yelp Polarity, for
    the same reason. Uses the review body (`content`), not the title."""
    raw = load_dataset("fancyzhx/amazon_polarity")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_size, val_size, test_size = 60_000, 3_000, 5_000

    train_full = raw["train"].shuffle(seed=seed)
    val_ds = train_full.select(range(val_size))
    train_ds = train_full.select(range(val_size, val_size + train_size))
    test_ds = raw["test"].shuffle(seed=0).select(range(test_size))

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "content", "label",
        ["title", "content"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=2)


def get_clinc_oos(seed: int, batch_size: int = 32) -> NLPBundle:
    """151-way intent classification (150 in-scope intents + out-of-scope),
    the "small" official config (50 train examples/intent) -- ships real
    train/validation/test splits already. The highest-class-count job in
    the NLP leg after banking77, added to help close the domain-alone
    certification gap (§6.3.1)."""
    raw = load_dataset("DeepPavlov/clinc_oos", "small")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = raw["train"].shuffle(seed=seed)
    val_ds = raw["validation"]
    test_ds = raw["test"]

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "text", "label",
        ["text", "label_text"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=151)


def get_sms_spam(seed: int, batch_size: int = 32) -> NLPBundle:
    """Binary SMS spam/ham classification, 5574 total -- ships no official
    split at all (single 'train' split), so we carve train/val/test
    ourselves, same spirit as every other carved split in this project.
    Heavily class-imbalanced (~87% ham), a real-world property left as-is
    rather than rebalanced."""
    raw = load_dataset("ucirvine/sms_spam")["train"].shuffle(seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    n = len(raw)
    n_test = int(n * 0.2)
    n_val = int(n * 0.1)
    test_ds = raw.select(range(n_test))
    val_ds = raw.select(range(n_test, n_test + n_val))
    train_ds = raw.select(range(n_test + n_val, n))

    train_loader, val_loader, test_loader = _make_loaders(
        train_ds, val_ds, test_ds, tokenizer, "sms", "label",
        ["sms"], seed, batch_size,
    )
    return NLPBundle(train_loader, val_loader, test_loader, num_classes=2)


NLP_DATASETS = {
    "sst2": get_sst2,
    "rotten_tomatoes": get_rotten_tomatoes,
    "trec": get_trec,
    "emotion": get_emotion,
    "tweet_eval_sentiment": get_tweet_eval_sentiment,
    "poem_sentiment": get_poem_sentiment,
    "banking77": get_banking77,
    "cola": get_cola,
    "mrpc": get_mrpc,
    "rte": get_rte,
    "qnli": get_qnli,
    "wnli": get_wnli,
    "tweet_eval_irony": get_tweet_eval_irony,
    "tweet_eval_hate": get_tweet_eval_hate,
    "tweet_eval_offensive": get_tweet_eval_offensive,
    "tweet_eval_emoji": get_tweet_eval_emoji,
    "tweet_eval_stance_abortion": get_tweet_eval_stance_abortion,
    "tweet_eval_stance_atheism": get_tweet_eval_stance_atheism,
    "tweet_eval_stance_climate": get_tweet_eval_stance_climate,
    "tweet_eval_stance_feminist": get_tweet_eval_stance_feminist,
    "tweet_eval_stance_hillary": get_tweet_eval_stance_hillary,
    "imdb": get_imdb,
    "ag_news": get_ag_news,
    "yelp_polarity": get_yelp_polarity,
    "amazon_polarity": get_amazon_polarity,
    "clinc_oos": get_clinc_oos,
    "sms_spam": get_sms_spam,
}


def get_nlp_dataset(name: str, seed: int, batch_size: int = 32) -> NLPBundle:
    return NLP_DATASETS[name](seed=seed, batch_size=batch_size)
