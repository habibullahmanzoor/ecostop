"""S1 (certified early cooldown): job registry. Hash-locked by s1_freeze.py BEFORE any test job is trained.

Recipe (as the pilot): SGD momentum 0.9 nesterov, wd 5e-4, batch 256, linear warm-up over the first WARM epochs,
stable LR for T_STABLE epochs with checkpoints, cooldown branches of C epochs from each checkpoint
(linear decay), reference = branch from the last checkpoint (stable 32 + cooldown 8 = 40 epochs).
Every job-seed also gets a recipe-matched 40-epoch cosine run (the standard full run; stop-only baseline and
common reference).

Calibration set  = the 6 pilot datasets (already trained: logs/pilot_cool, logs/pilot_cosine; ResNet-9, lr 0.05)
                   + 5 new ResNet-9 datasets (lr 0.05) + the 6 pilot datasets at peak lr 0.1 (recipe variant).
Test set (frozen) = ResNet-18 (a new architecture), lr 0.05, fresh seeds.
Datasets ResNet-9 cannot learn in 40 epochs (DTD, Flowers-102, Aircraft, Oxford-Pet, RenderedSST2: <=52%) are excluded.
"""
WARM = 2
T_STABLE = 32
CKPTS = [8, 12, 16, 20, 24, 32]
C = 8
T_COSINE = 40

PILOT = ["fashion_mnist", "kmnist", "mnist", "cifar10", "gtsrb", "imagenette"]
CAL_NEW = [(d, "resnet9", 0.05) for d in ["emnist_letters", "qmnist", "cifar100", "svhn", "emnist_balanced"]]
CAL_VARIANT = [(d, "resnet9", 0.1) for d in PILOT]
CAL_SEEDS = [0, 1]

TEST = [(d, "resnet18", 0.05) for d in
        ["fashion_mnist", "kmnist", "mnist", "cifar10", "gtsrb", "imagenette", "emnist_letters", "qmnist"]]
TEST_SEEDS = [10, 11]
