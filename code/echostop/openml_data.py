"""Real tabular datasets from OpenML (plan §16) -- extends the sklearn
built-in four with more distinct real jobs, cheaply. Handles the usual
real-world messiness (categorical columns, missing values) that OpenML
datasets have and sklearn's bundled toy datasets don't.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# (name, version) -- version="active" lets sklearn resolve the canonical one
CANDIDATE_DATASETS = [
    ("credit-g", "active"),
    ("vehicle", "active"),
    ("segment", "active"),
    ("sonar", "active"),
    ("ionosphere", "active"),
    ("spambase", "active"),
    ("banknote-authentication", "active"),
    ("blood-transfusion-service-center", "active"),
    ("balance-scale", "active"),
    ("car", "active"),
    ("tic-tac-toe", "active"),
    ("monks-problems-1", "active"),
    ("monks-problems-2", "active"),
    ("vowel", "active"),
    ("glass", "active"),
    ("ecoli", "active"),
    ("climate-model-simulation-crashes", "active"),
    ("steel-plates-fault", "active"),
    ("wdbc", "active"),
    ("diabetes", "active"),
]

CANDIDATE_DATASETS_BATCH2 = [
    ("australian", "active"),
    ("heart-statlog", "active"),
    ("heart-c", "active"),
    ("hepatitis", "active"),
    ("liver-disorders", "active"),
    ("haberman", "active"),
    ("vertebra-column", "active"),
    ("page-blocks", "active"),
    ("waveform-5000", "active"),
    ("phoneme", "active"),
    ("kc1", "active"),
    ("kc2", "active"),
    ("pc1", "active"),
    ("pc3", "active"),
    ("analcatdata_authorship", "active"),
    ("mfeat-fourier", "active"),
    ("mfeat-karhunen", "active"),
    ("optdigits", "active"),
    ("satimage", "active"),
    ("texture", "active"),
]

# session addition: pushing the calibration-corpus job pool toward the
# "low hundreds" the plan's own sample-size table says a tighter rho
# (0.1-0.15) needs -- the independence unit is the JOB (plan §6.6.2), so
# this only helps by adding distinct datasets, not more seeds of existing
# ones.
CANDIDATE_DATASETS_BATCH3 = [
    ("breast-w", "active"),
    ("credit-approval", "active"),
    ("colic", "active"),
    ("sick", "active"),
    ("hypothyroid", "active"),
    ("labor", "active"),
    ("lymph", "active"),
    ("zoo", "active"),
    ("autos", "active"),
    ("anneal", "active"),
    ("dermatology", "active"),
    ("heart-h", "active"),
    ("kr-vs-kp", "active"),
    ("mfeat-zernike", "active"),
    ("mfeat-morphological", "active"),
    ("mfeat-factors", "active"),
    ("soybean", "active"),
    ("tae", "active"),
    ("vote", "active"),
    ("yeast", "active"),
    ("cmc", "active"),
    ("eucalyptus", "active"),
    ("cylinder-bands", "active"),
    ("postoperative-patient-data", "active"),
]

# session addition #2: batch 3 got the calibration pool to n=54 and
# tightened the certified target from rho=0.3 to rho=0.25 (with a sensible,
# non-degenerate gamma_hat, not one pinned to the grid ceiling). Pushing
# toward n~100-150 is what the plan's own sample-size table says a genuinely
# useful rho=0.15-0.2 needs -- this is the cheapest, most mechanical lever
# available (each job trains in seconds to low minutes) and directly
# strengthens the paper's most distinctive, most reviewer-scrutinized claim
# (the finite-sample risk guarantee), more than adding a new domain would.
CANDIDATE_DATASETS_BATCH4 = [
    ("letter", "active"),
    ("nursery", "active"),
    ("mushroom", "active"),
    ("splice", "active"),
    ("wilt", "active"),
    ("monks-problems-3", "active"),
    ("prnn_crabs", "active"),
    ("biomed", "active"),
    ("thoracic-surgery", "active"),
    ("planning-relax", "active"),
    ("parkinsons", "active"),
    ("ilpd", "active"),
    ("SPECTF", "active"),
    ("breast-cancer", "active"),
    ("seismic-bumps", "active"),
    ("qsar-biodeg", "active"),
    ("ozone-level-8hr", "active"),
    ("wall-robot-navigation", "active"),
    ("cardiotocography", "active"),
    ("mfeat-pixel", "active"),
    ("solar-flare", "active"),
    ("audiology", "active"),
    ("primary-tumor", "active"),
    ("arrhythmia", "active"),
    ("electricity", "active"),
]

# session addition #3 -- deliberately excludes any dataset known to be
# procedurally-generated for benchmarking (e.g. "banana", "ringnorm",
# "twonorm", "corral") to keep the real-data-only constraint intact even
# though those are common, well-behaved OpenML entries.
CANDIDATE_DATASETS_BATCH5 = [
    ("jm1", "active"),
    ("kc3", "active"),
    ("mc1", "active"),
    ("mc2", "active"),
    ("mw1", "active"),
    ("cm1", "active"),
    ("musk", "active"),
    ("eeg-eye-state", "active"),
    ("hayes-roth", "active"),
    ("backache", "active"),
    ("bank-marketing", "active"),
    ("connect-4", "active"),
    ("adult", "active"),
    ("wine-quality-red", "active"),
    ("wine-quality-white", "active"),
    ("abalone", "active"),
    ("seeds", "active"),
    ("spectrometer", "active"),
]


@dataclass
class TabularBundle:
    name: str
    in_features: int
    num_classes: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def fetch_and_clean(name: str, version: str = "active") -> tuple[np.ndarray, np.ndarray]:
    """Fetch one OpenML dataset and return clean (X, y) arrays. Raises on
    failure -- caller is expected to catch and skip."""
    bunch = fetch_openml(name, version=version, as_frame=True, parser="auto")
    x_df = bunch.data.copy()
    y_raw = bunch.target

    # encode categoricals (object/category dtype columns)
    cat_cols = x_df.select_dtypes(include=["object", "category"]).columns
    num_cols = x_df.columns.difference(cat_cols)

    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        x_df[num_cols] = num_imputer.fit_transform(x_df[num_cols])
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        x_df[cat_cols] = cat_imputer.fit_transform(x_df[cat_cols])
        x_df = pd.get_dummies(x_df, columns=list(cat_cols))

    x = x_df.to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(np.int64)

    if x.shape[0] < 60 or x.shape[1] == 0:
        raise ValueError(f"{name}: too small or no usable features after cleaning")
    n_classes = len(np.unique(y))
    if n_classes < 2:
        raise ValueError(f"{name}: fewer than 2 classes after cleaning")
    # cap absurdly wide one-hot expansions (e.g. high-cardinality categoricals)
    if x.shape[1] > 500:
        raise ValueError(f"{name}: too many features after encoding ({x.shape[1]})")
    return x, y


def get_openml_dataset(name: str, version: str = "active", seed: int = 0,
                        batch_size: int = 32) -> TabularBundle:
    x, y = fetch_and_clean(name, version)

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
