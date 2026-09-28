"""Loading and feature extraction for the three open IoT data sets used by Montori et al.

The data sets are the authors' own public release (https://github.com/stradivarius/
TSopendatastreams). Each `.tkse` file is a headerless CSV whose first nine columns are
metadata and whose remaining columns are the sensor readings:

    0 id | 1 name | 2 channel id | 3 channel name | 4 description
    5 timestamp | 6 latitude | 7 longitude | 8 class label | 9.. readings

Swissex and UrbObs carry no metadata (columns 1-7 are empty), which matches the paper:
only ThingSpeak provides multiple information domains.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N_META = 9
LABEL_COL = 8
NAME_COL = 1
CHANNEL_COL = 3
DESCRIPTION_COL = 4

DATASETS = {
    "TS": ("dataset_ThingspeakEU", 96, 21, True),
    "Swissex": ("dataset_Swissex", 445, 11, False),
    "UrbObs": ("dataset_UrbanObservatory", 864, 16, False),
}


@dataclass
class Split:
    """One train or test partition of a data set."""

    series: np.ndarray          # (n, m) raw, non-normalised sensor readings
    labels: np.ndarray          # (n,) integer class labels
    name: np.ndarray            # (n,) datastream name, '' where absent
    channel: np.ndarray         # (n,) channel name, '' where absent
    description: np.ndarray     # (n,) free-text description, '' where absent

    def __len__(self) -> int:
        return len(self.labels)


@dataclass
class Dataset:
    key: str
    train: Split
    test: Split
    n_classes: int
    has_metadata: bool

    @property
    def n_points(self) -> int:
        return self.train.series.shape[1]

    def summary(self) -> dict:
        return {"dataset": self.key,
                "train": len(self.train),
                "test": len(self.test),
                "total": len(self.train) + len(self.test),
                "points": self.n_points,
                "classes": self.n_classes,
                "metadata": self.has_metadata}


def _read(path: Path, n_points: int) -> Split:
    raw = pd.read_csv(path, header=None, low_memory=False)
    expected = N_META + n_points
    if raw.shape[1] != expected:
        raise ValueError(f"{path.name}: expected {expected} columns, found {raw.shape[1]}")

    series = raw.iloc[:, N_META:].to_numpy(dtype=np.float64)
    labels = raw.iloc[:, LABEL_COL].to_numpy(dtype=int)

    def text(col: int) -> np.ndarray:
        return raw.iloc[:, col].fillna("").astype(str).str.strip().to_numpy()

    return Split(series=series, labels=labels, name=text(NAME_COL),
                 channel=text(CHANNEL_COL), description=text(DESCRIPTION_COL))


def load(key: str, root: str | Path = "data") -> Dataset:
    """Load one data set using the authors' own stratified 70/30 train/test split."""
    if key not in DATASETS:
        raise KeyError(f"unknown data set {key!r}; choose from {sorted(DATASETS)}")
    folder, n_points, n_classes, has_meta = DATASETS[key]
    base = Path(root) / folder
    ds = Dataset(key=key,
                 train=_read(base / "TRAIN.tkse", n_points),
                 test=_read(base / "TEST.tkse", n_points),
                 n_classes=n_classes, has_metadata=has_meta)

    seen = set(ds.train.labels) | set(ds.test.labels)
    if len(seen) != n_classes:
        raise ValueError(f"{key}: expected {n_classes} classes, found {len(seen)}")
    return ds


def load_all(root: str | Path = "data") -> dict[str, Dataset]:
    return {k: load(k, root) for k in DATASETS}


# --------------------------------------------------------------------- features
BOS_NAMES = ["mean", "median", "max", "min", "std", "rms",
             "q25", "q75", "iqr", "kurtosis", "range"]


def bag_of_summaries(series: np.ndarray) -> np.ndarray:
    """The paper's bag-of-summaries features, computed on non-normalised data.

    Section IV-B2 states eleven features and names ten of them: mean, median, maximum,
    minimum, standard deviation, RMS, quantile, IQR, kurtosis and range. "Quantile" is
    read here as the lower and upper quartiles, which makes the count eleven and makes
    the IQR internally consistent. This assumption is documented in the report.
    """
    x = np.asarray(series, dtype=np.float64)
    q25 = np.percentile(x, 25, axis=1)
    q75 = np.percentile(x, 75, axis=1)
    feats = np.column_stack([
        x.mean(axis=1),
        np.median(x, axis=1),
        x.max(axis=1),
        x.min(axis=1),
        x.std(axis=1),
        np.sqrt((x ** 2).mean(axis=1)),
        q25,
        q75,
        q75 - q25,
        stats.kurtosis(x, axis=1, bias=False),
        x.max(axis=1) - x.min(axis=1),
    ])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def z_normalise(series: np.ndarray) -> np.ndarray:
    """Per-series z-normalisation, the canonical preprocessing for TSC algorithms."""
    x = np.asarray(series, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd
