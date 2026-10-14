"""Dataset loaders for the pilot (plan §40): Fashion-MNIST and CIFAR-10.

Produces train/validation/test splits per run (plan §11.0): validation is
used for all stopping/checkpoint decisions, test is held out and touched
only to compute the audited outcome `A_t` (plan §11.0.1). The official
torchvision test split is reserved as `test`; `validation` is carved out of
the official training split so `test` is never touched during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

DATA_ROOT = "./data"


@dataclass
class DatasetBundle:
    name: str
    in_channels: int
    num_classes: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def _split_train_val(dataset, val_fraction: float, seed: int):
    n = len(dataset)
    n_val = int(n * val_fraction)
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )
    return train_subset, val_subset


def get_fashion_mnist(
    seed: int = 0,
    val_fraction: float = 0.1,
    batch_size: int = 256,
    num_workers: int = 2,
) -> DatasetBundle:
    mean, std = (0.2860,), (0.3530,)
    train_tf = T.Compose(
        [
            T.RandomCrop(28, padding=2),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_full = torchvision.datasets.FashionMNIST(
        DATA_ROOT, train=True, download=True, transform=train_tf
    )
    # separate instance with eval transform for the validation subset,
    # sharing indices with train_full via the same seeded split
    train_full_eval = torchvision.datasets.FashionMNIST(
        DATA_ROOT, train=True, download=True, transform=eval_tf
    )
    train_subset, val_indices_holder = _split_train_val(train_full, val_fraction, seed)
    _, val_subset = _split_train_val(train_full_eval, val_fraction, seed)

    test_ds = torchvision.datasets.FashionMNIST(
        DATA_ROOT, train=False, download=True, transform=eval_tf
    )

    return DatasetBundle(
        name="fashion_mnist",
        in_channels=1,
        num_classes=10,
        train_loader=DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        val_loader=DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        test_loader=DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    )


def _get_mnist_family(
    torchvision_cls, name: str, num_classes: int, mean: tuple, std: tuple,
    image_size: int = 28, in_channels: int = 1, seed: int = 0, val_fraction: float = 0.1,
    batch_size: int = 256, num_workers: int = 2, **dataset_kwargs,
) -> DatasetBundle:
    """Shared loader for datasets with a boolean `train=True/False` API
    (the original MNIST family: MNIST, KMNIST, EMNIST -- session addition to
    grow the vision domain beyond its original 2-dataset x 2-architecture
    pilot -- the paper's own §6.3.1 flags this as the corpus's biggest
    remaining imbalance). Generalized here (in_channels, image_size) to also
    cover the second-round batch: USPS, QMNIST (still grayscale, `train=`
    API) and CIFAR-100 (RGB, `train=` API, same shape as CIFAR-10)."""
    train_tf = T.Compose([
        T.RandomCrop(image_size, padding=2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_full = torchvision_cls(DATA_ROOT, train=True, download=True, transform=train_tf, **dataset_kwargs)
    train_full_eval = torchvision_cls(DATA_ROOT, train=True, download=True, transform=eval_tf, **dataset_kwargs)
    train_subset, _ = _split_train_val(train_full, val_fraction, seed)
    _, val_subset = _split_train_val(train_full_eval, val_fraction, seed)

    test_ds = torchvision_cls(DATA_ROOT, train=False, download=True, transform=eval_tf, **dataset_kwargs)

    return DatasetBundle(
        name=name, in_channels=in_channels, num_classes=num_classes,
        train_loader=DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True),
        val_loader=DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True),
    )


def _get_split_str_family(
    torchvision_cls, name: str, num_classes: int, mean: tuple, std: tuple,
    image_size: int, in_channels: int = 3, train_split: str = "train", test_split: str = "test",
    resize_native: bool = False, seed: int = 0, val_fraction: float = 0.1,
    batch_size: int = 256, num_workers: int = 2, **dataset_kwargs,
) -> DatasetBundle:
    """Shared loader for datasets using a string `split=` API instead of a
    boolean `train=` one, with only two official splits (no built-in
    validation) -- SVHN, GTSRB, Food101, STL10. Our own validation split is
    carved from the official train split, same convention as the `train=`
    family and as the NLP loaders that ship no validation split of their
    own. `resize_native=True` resizes variable-native-size images down to
    `image_size` before the crop augmentation (GTSRB/Food101 ship images at
    many different native resolutions); fixed-size datasets (SVHN, STL10)
    skip that step since it would be a no-op."""
    resize = [T.Resize((image_size, image_size), antialias=True)] if resize_native else []
    train_tf = T.Compose([
        *resize,
        T.RandomCrop(image_size, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([*resize, T.ToTensor(), T.Normalize(mean, std)])

    train_full = torchvision_cls(DATA_ROOT, split=train_split, download=True, transform=train_tf, **dataset_kwargs)
    train_full_eval = torchvision_cls(DATA_ROOT, split=train_split, download=True, transform=eval_tf, **dataset_kwargs)
    train_subset, _ = _split_train_val(train_full, val_fraction, seed)
    _, val_subset = _split_train_val(train_full_eval, val_fraction, seed)

    test_ds = torchvision_cls(DATA_ROOT, split=test_split, download=True, transform=eval_tf, **dataset_kwargs)

    return DatasetBundle(
        name=name, in_channels=in_channels, num_classes=num_classes,
        train_loader=DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True),
        val_loader=DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True),
    )


def _get_three_way_split_family(
    torchvision_cls, name: str, num_classes: int, mean: tuple, std: tuple,
    image_size: int, in_channels: int = 3, train_split: str = "train",
    val_split: str = "val", test_split: str = "test",
    batch_size: int = 256, num_workers: int = 2, **dataset_kwargs,
) -> DatasetBundle:
    """Shared loader for datasets that ship real train/val/test splits of
    their own (Flowers102, DTD, FGVCAircraft, RenderedSST2) -- no carving
    needed, same convention as rotten_tomatoes/emotion/tweet_eval in the NLP
    leg. `seed` is intentionally not a parameter: these splits are fixed by
    the dataset, not re-drawn per seed."""
    train_tf = T.Compose([
        T.Resize((image_size, image_size), antialias=True),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([T.Resize((image_size, image_size), antialias=True), T.ToTensor(), T.Normalize(mean, std)])

    train_ds = torchvision_cls(DATA_ROOT, split=train_split, download=True, transform=train_tf, **dataset_kwargs)
    val_ds = torchvision_cls(DATA_ROOT, split=val_split, download=True, transform=eval_tf, **dataset_kwargs)
    test_ds = torchvision_cls(DATA_ROOT, split=test_split, download=True, transform=eval_tf, **dataset_kwargs)

    return DatasetBundle(
        name=name, in_channels=in_channels, num_classes=num_classes,
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True),
    )


def _get_no_split_family(
    torchvision_cls, name: str, num_classes: int, mean: tuple, std: tuple,
    image_size: int, in_channels: int = 3, val_fraction: float = 0.1,
    test_fraction: float = 0.2, seed: int = 0, batch_size: int = 256,
    num_workers: int = 2, **dataset_kwargs,
) -> DatasetBundle:
    """Shared loader for datasets that ship no official split at all
    (EuroSAT, alongside the earlier custom SEMEION loader) -- carve
    train/val/test ourselves, seeded, same spirit as every other carved
    split in this project (never touch the test portion for anything but
    the final audited outcome)."""
    tf = T.Compose([T.Resize((image_size, image_size), antialias=True), T.ToTensor(), T.Normalize(mean, std)])
    full = torchvision_cls(DATA_ROOT, download=True, transform=tf, **dataset_kwargs)

    n = len(full)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        full, [n_train, n_val, n_test], generator=generator
    )

    return DatasetBundle(
        name=name, in_channels=in_channels, num_classes=num_classes,
        train_loader=DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True),
        val_loader=DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True),
        test_loader=DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True),
    )


def get_eurosat(seed: int = 0, **kwargs) -> DatasetBundle:
    """Sentinel-2 satellite land-use classification, 10 classes, ships no
    official split -- added to help close the vision domain-alone
    certification gap (§6.3.1)."""
    return _get_no_split_family(torchvision.datasets.EuroSAT, "eurosat", 10,
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), image_size=64,
                                 seed=seed, **kwargs)


def get_mnist(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_mnist_family(torchvision.datasets.MNIST, "mnist", 10,
                              (0.1307,), (0.3081,), seed=seed, **kwargs)


def get_kmnist(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_mnist_family(torchvision.datasets.KMNIST, "kmnist", 10,
                              (0.1918,), (0.3483,), seed=seed, **kwargs)


def get_emnist_letters(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_mnist_family(torchvision.datasets.EMNIST, "emnist_letters", 27,
                              (0.1722,), (0.3309,), seed=seed, split="letters", **kwargs)


def get_qmnist(seed: int = 0, **kwargs) -> DatasetBundle:
    """QMNIST reconstructs MNIST with a much larger labeled test set;
    `train=True/False` still selects the standard 60k-train split we use
    here, same normalization as MNIST since it's the same source images."""
    return _get_mnist_family(torchvision.datasets.QMNIST, "qmnist", 10,
                              (0.1307,), (0.3081,), seed=seed, what=None, compat=True, **kwargs)


def get_emnist_balanced(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_mnist_family(torchvision.datasets.EMNIST, "emnist_balanced", 47,
                              (0.1751,), (0.3332,), seed=seed, split="balanced", **kwargs)


def get_emnist_digits(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_mnist_family(torchvision.datasets.EMNIST, "emnist_digits", 10,
                              (0.1733,), (0.3317,), seed=seed, split="digits", **kwargs)


def get_emnist_byclass(seed: int = 0, **kwargs) -> DatasetBundle:
    """EMNIST's largest split: 62 classes (digits + upper + lower case,
    unmerged, unlike 'balanced'), 814k characters total -- added specifically
    to help close the vision domain-alone certification gap (§6.3.1)."""
    return _get_mnist_family(torchvision.datasets.EMNIST, "emnist_byclass", 62,
                              (0.1736,), (0.3317,), seed=seed, split="byclass", **kwargs)


def get_cifar100(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_mnist_family(torchvision.datasets.CIFAR100, "cifar100", 100,
                              (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762),
                              image_size=32, in_channels=3, seed=seed, **kwargs)


def get_svhn(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_split_str_family(torchvision.datasets.SVHN, "svhn", 10,
                                  (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970),
                                  image_size=32, seed=seed, **kwargs)


def get_gtsrb(seed: int = 0, **kwargs) -> DatasetBundle:
    """German Traffic Sign Recognition Benchmark, 43 classes, images at many
    different native resolutions -- resized to 32x32 like the other
    CIFAR-scale jobs."""
    return _get_split_str_family(torchvision.datasets.GTSRB, "gtsrb", 43,
                                  (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                  image_size=32, resize_native=True, seed=seed, **kwargs)


def get_food101(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_split_str_family(torchvision.datasets.Food101, "food101", 101,
                                  (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                  image_size=64, resize_native=True, seed=seed, **kwargs)


def get_stl10(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_split_str_family(torchvision.datasets.STL10, "stl10", 10,
                                  (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713),
                                  image_size=64, resize_native=True, seed=seed, **kwargs)


def get_flowers102(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_three_way_split_family(torchvision.datasets.Flowers102, "flowers102", 102,
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), image_size=64, **kwargs)


def get_dtd(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_three_way_split_family(torchvision.datasets.DTD, "dtd", 47,
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), image_size=64,
                                        partition=1, **kwargs)


def get_fgvc_aircraft(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_three_way_split_family(torchvision.datasets.FGVCAircraft, "fgvc_aircraft", 100,
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), image_size=64, **kwargs)


def get_rendered_sst2(seed: int = 0, **kwargs) -> DatasetBundle:
    return _get_three_way_split_family(torchvision.datasets.RenderedSST2, "rendered_sst2", 2,
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), image_size=64, **kwargs)


def get_country211(seed: int = 0, **kwargs) -> DatasetBundle:
    """OpenAI's geolocation benchmark, 211 countries, real train/valid/test
    splits (small: 150/50/100 images per class) -- added to help close the
    vision domain-alone certification gap (§6.3.1)."""
    return _get_three_way_split_family(torchvision.datasets.Country211, "country211", 211,
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), image_size=64,
                                        val_split="valid", **kwargs)


def get_imagenette(seed: int = 0, **kwargs) -> DatasetBundle:
    """10-class subset of ImageNet (tench, English springer, cassette player,
    chainsaw, church, French horn, garbage truck, gas pump, golf ball,
    parachute); the 160px release keeps the download small. Ships only
    train/val -- carve our own validation split out of train, same
    convention as SVHN/GTSRB."""
    return _get_split_str_family(torchvision.datasets.Imagenette, "imagenette", 10,
                                  (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), image_size=64,
                                  resize_native=True, seed=seed, test_split="val", size="160px", **kwargs)


def get_oxford_pet(seed: int = 0, val_fraction: float = 0.1, batch_size: int = 256,
                    num_workers: int = 2) -> DatasetBundle:
    """OxfordIIITPet ships only trainval/test (no separate val) -- carve our
    own the same way the `train=`-API family does."""
    image_size = 64
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    train_tf = T.Compose([T.Resize((image_size, image_size), antialias=True), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize(mean, std)])
    eval_tf = T.Compose([T.Resize((image_size, image_size), antialias=True), T.ToTensor(), T.Normalize(mean, std)])

    train_full = torchvision.datasets.OxfordIIITPet(DATA_ROOT, split="trainval", download=True, transform=train_tf)
    train_full_eval = torchvision.datasets.OxfordIIITPet(DATA_ROOT, split="trainval", download=True, transform=eval_tf)
    train_subset, _ = _split_train_val(train_full, val_fraction, seed)
    _, val_subset = _split_train_val(train_full_eval, val_fraction, seed)
    test_ds = torchvision.datasets.OxfordIIITPet(DATA_ROOT, split="test", download=True, transform=eval_tf)

    return DatasetBundle(
        name="oxford_pet", in_channels=3, num_classes=37,
        train_loader=DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True),
        val_loader=DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True),
    )


def get_semeion(seed: int = 0, val_fraction: float = 0.1, test_fraction: float = 0.2,
                 batch_size: int = 256, num_workers: int = 2) -> DatasetBundle:
    """SEMEION handwritten digits (16x16 grayscale, 1593 total) ships no
    official split at all -- carve train/val/test ourselves, seeded, same
    spirit as every other carved split in this project (never touch the
    test portion for anything but the final audited outcome)."""
    mean, std = (0.2415,), (0.3186,)
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    full = torchvision.datasets.SEMEION(DATA_ROOT, download=True, transform=tf)

    n = len(full)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        full, [n_train, n_val, n_test], generator=generator
    )

    return DatasetBundle(
        name="semeion", in_channels=1, num_classes=10,
        train_loader=DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True),
        val_loader=DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True),
        test_loader=DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True),
    )


def get_cifar10(
    seed: int = 0,
    val_fraction: float = 0.1,
    batch_size: int = 256,
    num_workers: int = 2,
) -> DatasetBundle:
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_full = torchvision.datasets.CIFAR10(
        DATA_ROOT, train=True, download=True, transform=train_tf
    )
    train_full_eval = torchvision.datasets.CIFAR10(
        DATA_ROOT, train=True, download=True, transform=eval_tf
    )
    train_subset, _ = _split_train_val(train_full, val_fraction, seed)
    _, val_subset = _split_train_val(train_full_eval, val_fraction, seed)

    test_ds = torchvision.datasets.CIFAR10(
        DATA_ROOT, train=False, download=True, transform=eval_tf
    )

    return DatasetBundle(
        name="cifar10",
        in_channels=3,
        num_classes=10,
        train_loader=DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        val_loader=DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
        test_loader=DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    )


def get_cifar10_train_loader_at_size(
    image_size: int,
    seed: int = 0,
    val_fraction: float = 0.1,
    batch_size: int = 256,
    num_workers: int = 2,
) -> DataLoader:
    """Train-split loader resized to `image_size` (plan §16 REQUIRED --
    progressive resizing, the cheapest genuine variable-per-epoch-energy
    setting). Crop/flip augmentation happens at native 32x32, then the
    result is resized down -- keeps augmentation geometry consistent across
    phases. Validation/test stay at native resolution (see train_progressive.py).
    """
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.Resize(image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    train_full = torchvision.datasets.CIFAR10(
        DATA_ROOT, train=True, download=True, transform=train_tf
    )
    train_subset, _ = _split_train_val(train_full, val_fraction, seed)
    return DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


LOADERS = {
    "fashion_mnist": get_fashion_mnist,
    "cifar10": get_cifar10,
    "mnist": get_mnist,
    "kmnist": get_kmnist,
    "emnist_letters": get_emnist_letters,
    "qmnist": get_qmnist,
    "emnist_balanced": get_emnist_balanced,
    "emnist_digits": get_emnist_digits,
    "emnist_byclass": get_emnist_byclass,
    "cifar100": get_cifar100,
    "svhn": get_svhn,
    "gtsrb": get_gtsrb,
    "food101": get_food101,
    "stl10": get_stl10,
    "flowers102": get_flowers102,
    "dtd": get_dtd,
    "fgvc_aircraft": get_fgvc_aircraft,
    "rendered_sst2": get_rendered_sst2,
    "oxford_pet": get_oxford_pet,
    "semeion": get_semeion,
    "eurosat": get_eurosat,
    "country211": get_country211,
    "imagenette": get_imagenette,
}


def get_dataset(name: str, **kwargs) -> DatasetBundle:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset: {name!r}, have {list(LOADERS)}")
    return LOADERS[name](**kwargs)
