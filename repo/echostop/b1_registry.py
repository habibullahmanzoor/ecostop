"""B1 fresh test set: job registry. Fixed BEFORE any B1 job is trained (see b1_freeze.py,
which hashes this file together with the frozen method constants).

Fresh = not used to design or calibrate the method: new architectures, new schedules and
lengths, new optimisers, and datasets/tasks absent from the 142-job corpus.
"""

# --- vision: new architectures x datasets (3 seeds each) ---------------------------------
VISION_RECIPES = {
    # arch: optimiser, lr, weight decay, schedule, epochs, batch size
    "mobilenetv3": dict(opt="adamw", lr=2e-3, wd=0.05, sched="cosine_warmup", warmup=3, epochs=50, bs=256),
    "vit_tiny":    dict(opt="adamw", lr=1e-3, wd=0.05, sched="cosine_warmup", warmup=5, epochs=50, bs=256),
    "resnet18_step": dict(opt="sgd", lr=0.1, wd=5e-4, sched="step", epochs=40, bs=256),
    "vgg_small":   dict(opt="sgd", lr=0.05, wd=5e-4, sched="onecycle", epochs=30, bs=256),
}
VISION_JOBS = (
    [("mobilenetv3", d) for d in ("cifar10", "fashion_mnist", "gtsrb", "dtd", "flowers102")]
    + [("vit_tiny", d) for d in ("cifar10", "fashion_mnist", "kmnist", "oxford_pet")]
    + [("resnet18_step", d) for d in ("fashion_mnist", "dtd", "oxford_pet", "semeion")]
    + [("vgg_small", d) for d in ("stl10", "rendered_sst2", "mnist")]
)

# --- NLP: a different model (small BERT), new tasks, 15 epochs (3 seeds each) -------------
NLP_MODEL = "google/bert_uncased_L-4_H-512_A-8"
NLP_RECIPE = dict(epochs=15, bs=32, lr=5e-5)
# name: (hf path, config, text col, text2 col, label col, official val split or None, test split, max train rows)
NLP_JOBS = {
    "subj":          ("SetFit/subj", None, "text", None, "label", None, "test", None),
    "cr":            ("SetFit/CR", None, "text", None, "label", None, "test", None),
    "bbc_news":      ("SetFit/bbc-news", None, "text", None, "label", None, "test", None),
    "hate_offensive": ("SetFit/hate_speech_offensive", None, "text", None, "label", None, "test", None),
    "student_questions": ("SetFit/student-question-categories", None, "text", None, "label", None, "test", 30000),
    "boolq":         ("super_glue", "boolq", "question", "passage", "label", "validation", "validation", None),
    "sst5":          ("SetFit/sst5", None, "text", None, "label", "validation", "test", None),
    "enron_spam":    ("SetFit/enron_spam", None, "text", None, "label", None, "test", None),
}

# --- tabular: 14 unused OpenML datasets, wider MLP, AdamW, 100 epochs (4 seeds each) -----
TABULAR_RECIPE = dict(hidden=256, opt="adamw", lr=2e-3, wd=1e-2, epochs=100, bs=64)
TABULAR_JOBS = ["nursery", "pc4", "pendigits", "churn", "MagicTelescope", "ecoli", "hill-valley",
                "gas-drift", "nomao", "madelon", "dna", "artificial-characters", "collins", "autos"]

SEEDS = {"vision": [0, 1, 2], "nlp": [0, 1, 2], "tabular": [0, 1, 2, 3]}
