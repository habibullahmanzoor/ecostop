"""Corpus composition chart -- superseded by Table 1 in the paper's Sec. 5
(a bar chart of names+counts didn't fit any page-space budget legibly;
a table represents the same data more compactly). Kept only for reference;
not wired into any numbered figure slot. Every count here is read directly
from the discovered run-sets / dataset lists, not estimated.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOMAIN_COLOR = {"vision": "#0072B2", "nlp": "#D55E00", "tabular": "#009E73"}

# (label, count, domain) -- grouped by dataset family/content, not by when each was added
ROWS = [
    ("Tabular: scikit-learn benchmarks\n(Digits, Breast Cancer, Wine, Iris)", 4, "tabular"),
    ("Tabular: OpenML datasets\n(cleaned, real, no synthetic)", 89, "tabular"),
    ("NLP: sentiment, topic & intent tasks\n(SST-2, Rotten Tomatoes, TREC, Emotion,\nTweetEval-Sentiment, Poem Sentiment, Banking77)", 7, "nlp"),
    ("NLP: grammaticality, entailment &\nstance tasks (GLUE benchmark, TweetEval\nstance/irony/hate/offensive/emoji, IMDB, AG News)", 16, "nlp"),
    ("Vision: headline pilot\n(CIFAR-10, Fashion-MNIST, ResNet-9/18)", 4, "vision"),
    ("Vision: digit & character datasets\n(MNIST, KMNIST, EMNIST-Letters)", 3, "vision"),
    ("Vision: house-number, traffic-sign &\nfine-grained image datasets (QMNIST, EMNIST,\nCIFAR-100, SVHN, GTSRB, Flowers-102, DTD,\nFGVC-Aircraft, RenderedSST2, Oxford-IIIT Pet, SEMEION)", 12, "vision"),
    ("Vision: satellite, natural-image &\ncharacter-set datasets\n(EuroSAT, Imagenette, EMNIST-ByClass)", 3, "vision"),
    ("Vision: step-decay schedule test\n(CIFAR-10, step-decay learning rate)", 1, "vision"),
    ("NLP: review sentiment & intent tasks\n(Yelp Polarity, Amazon Polarity,\nCLINC-OOS, SMS Spam)", 4, "nlp"),
]

DOMAIN_TOTAL = {"tabular": 93, "nlp": 27, "vision": 23}
DOMAIN_LABEL = {"tabular": "Tabular", "nlp": "NLP (fine-tuning)", "vision": "Vision (from scratch)"}


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 12.5))
    y = np.arange(len(ROWS))[::-1]

    for yi, (label, count, domain) in zip(y, ROWS):
        ax.barh(yi, count, color=DOMAIN_COLOR[domain], zorder=3, height=0.6)
        ax.text(count + 1.2, yi, str(count), va="center", fontsize=13, fontweight="bold",
                color=DOMAIN_COLOR[domain])

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=12.5)
    ax.set_xlabel("Number of distinct datasets", fontsize=14)
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8, zorder=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=DOMAIN_COLOR[d],
                              label=f"{DOMAIN_LABEL[d]} — {DOMAIN_TOTAL[d]} total")
               for d in ["tabular", "nlp", "vision"]]
    ax.legend(handles=handles, frameon=False, fontsize=13.5, loc="lower right")

    ax.set_title("Composition of the 143-dataset evaluation corpus,\nby domain and dataset family",
                  fontsize=17)
    fig.tight_layout()
    out = "figures/retired_corpus_composition_chart.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
