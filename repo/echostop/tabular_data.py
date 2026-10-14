"""Real tabular datasets for the calibration-corpus job pool (plan §16).
All genuine measured data (handwritten digits, diagnostic measurements,
chemistry assays, flower measurements) -- no synthetic generation, per
explicit instruction. train/val/test split per plan §11.0 (decisions use
validation; test is held out for the audited outcome A_t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TabularBundle:
    name: str
    in_features: int
    num_classes: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


LOADERS = {
    "digits": load_digits,
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "iris": load_iris,
}


def get_tabular_dataset(name: str, seed: int = 0, batch_size: int = 32) -> TabularBundle:
    if name not in LOADERS:
        raise ValueError(f"unknown tabular dataset: {name!r}, have {list(LOADERS)}")
    data = LOADERS[name]()
    x, y = data.data.astype(np.float32), data.target.astype(np.int64)

    # 60/20/20 train/val/test, seeded
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.4, random_state=seed, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    scaler = StandardScaler().fit(x_train)
    x_train, x_val, x_test = (scaler.transform(a) for a in (x_train, x_val, x_test))

    def to_loader(xa, ya, shuffle):
        ds = TensorDataset(torch.from_numpy(xa).float(), torch.from_numpy(ya).long())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return TabularBundle(
        name=name,
        in_features=x.shape[1],
        num_classes=len(np.unique(y)),
        train_loader=to_loader(x_train, y_train, True),
        val_loader=to_loader(x_val, y_val, False),
        test_loader=to_loader(x_test, y_test, False),
    )
