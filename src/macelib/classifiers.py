"""Standalone classifiers used by the MACE framework.

Each classifier exposes `fit`, `predict_proba` and `predict`, and every `predict_proba`
returns an (n, c) array whose columns follow `self.classes_` in ascending label order.
MACE needs a full ranking of classes per example, not just a top-1 decision, so the
nearest-neighbour and NLP classifiers convert distances into probabilities exactly as
the authors do rather than returning a degenerate 0/1 distribution.

Two defects in the authors' reference implementation are deliberately not reproduced:
`dictionary = {}` is declared at class scope there, so it is shared mutable state across
instances, and class order is taken from `set(y_train)` iteration rather than being
sorted. Both are noted in the report; neither changes the published numbers when only a
single instance is fitted at a time and labels are small integers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rapidfuzz import process as rf_process
from rapidfuzz.distance import DamerauLevenshtein
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin, clone

EPS_TS = 1e-3      # the authors add 0.001 before inverting time-series distances
EPS_NLP = 1e-6     # and 0.000001 before inverting edit distances


@dataclass
class Views:
    """The three information domains a datastream can be seen through."""

    series: np.ndarray          # raw readings, (n, m)
    bos: np.ndarray             # bag-of-summaries features, (n, 11)
    text: np.ndarray            # metadata text, (n,)

    def __len__(self) -> int:
        return len(self.series)

    def subset(self, idx) -> "Views":
        return Views(series=self.series[idx], bos=self.bos[idx], text=self.text[idx])


def _inverse_distance_proba(dist_to_class: np.ndarray, eps: float, power: int) -> np.ndarray:
    """Turn per-class distances into probabilities, as the authors do."""
    d = (dist_to_class + eps) ** power
    inv = 1.0 / d
    return inv / inv.sum(axis=1, keepdims=True)


def _min_per_class(dist: np.ndarray, train_labels: np.ndarray,
                   classes: np.ndarray) -> np.ndarray:
    """Collapse an (n_test, n_train) distance matrix to (n_test, n_classes) minima."""
    out = np.empty((dist.shape[0], len(classes)), dtype=np.float64)
    for j, c in enumerate(classes):
        cols = np.flatnonzero(train_labels == c)
        out[:, j] = dist[:, cols].min(axis=1) if cols.size else np.inf
    return out


class NearestNeighbourED(BaseEstimator, ClassifierMixin):
    """1NN-ED: minimum Euclidean distance to each class, inverted into probabilities.

    `normalise` applies per-series z-normalisation, the canonical TSC preprocessing.
    The paper reports that whole-series methods do better on non-normalised open IoT
    data because magnitude itself discriminates sensor types, so the pool uses the
    non-normalised variant.
    """

    def __init__(self, normalise: bool = False):
        self.normalise = normalise

    def _prep(self, X: Views) -> np.ndarray:
        s = np.asarray(X.series, dtype=np.float64)
        if not self.normalise:
            return s
        mu = s.mean(axis=1, keepdims=True)
        sd = s.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        return (s - mu) / sd

    def fit(self, X: Views, y):
        self.classes_ = np.unique(y)
        self._train = self._prep(X)
        self._y = np.asarray(y)
        return self

    def predict_proba(self, X: Views) -> np.ndarray:
        d = cdist(self._prep(X), self._train, metric="euclidean")
        return _inverse_distance_proba(_min_per_class(d, self._y, self.classes_),
                                       EPS_TS, power=1)

    def predict(self, X: Views) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class DDLNLPClassifier(BaseEstimator, ClassifierMixin):
    """The paper's dictionary Damerau-Levenshtein NLP classifier (Algorithm 2).

    Training builds a bag of datastream names per class. At test time the minimum
    length-normalised Damerau-Levenshtein distance to each class's bag is computed, and
    the distances are squared, inverted and normalised into class probabilities.
    """

    def fit(self, X: Views, y):
        self.classes_ = np.unique(y)
        self._y = np.asarray(y)
        self._names = [str(t) for t in X.text]
        return self

    def predict_proba(self, X: Views) -> np.ndarray:
        queries = [str(t) for t in X.text]
        dist = rf_process.cdist(queries, self._names,
                                scorer=DamerauLevenshtein.normalized_distance,
                                workers=-1, dtype=np.float32).astype(np.float64)
        return _inverse_distance_proba(_min_per_class(dist, self._y, self.classes_),
                                       EPS_NLP, power=2)

    def predict(self, X: Views) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class BOSClassifier(BaseEstimator, ClassifierMixin):
    """Any scikit-learn estimator applied to the eleven bag-of-summaries features."""

    def __init__(self, estimator, scale: bool = False):
        self.estimator = estimator
        self.scale = scale

    def fit(self, X: Views, y):
        from sklearn.preprocessing import StandardScaler

        self.classes_ = np.unique(y)
        self._est = clone(self.estimator)
        feats = np.asarray(X.bos, dtype=np.float64)
        if self.scale:
            self._scaler = StandardScaler().fit(feats)
            feats = self._scaler.transform(feats)
        self._est.fit(feats, y)
        return self

    def _feats(self, X: Views) -> np.ndarray:
        feats = np.asarray(X.bos, dtype=np.float64)
        return self._scaler.transform(feats) if self.scale else feats

    def predict_proba(self, X: Views) -> np.ndarray:
        feats = self._feats(X)
        if hasattr(self._est, "predict_proba"):
            proba = self._est.predict_proba(feats)
        else:
            # RidgeClassifier exposes only a decision function; a softmax over it gives
            # the ranking MACE needs. The paper keeps Ridge out of the ensemble pool
            # anyway, so this only affects the standalone comparison and its AUC.
            scores = np.atleast_2d(self._est.decision_function(feats))
            if scores.shape[1] == 1:                       # binary edge case
                scores = np.hstack([-scores, scores])
            scores = scores - scores.max(axis=1, keepdims=True)
            exp = np.exp(scores)
            proba = exp / exp.sum(axis=1, keepdims=True)
        # align to self.classes_ in case a CV fold is missing a rare class
        if len(self._est.classes_) == len(self.classes_) and \
                np.array_equal(self._est.classes_, self.classes_):
            return proba
        full = np.zeros((proba.shape[0], len(self.classes_)))
        for j, c in enumerate(self._est.classes_):
            full[:, np.searchsorted(self.classes_, c)] = proba[:, j]
        rows = full.sum(axis=1, keepdims=True)
        rows[rows == 0] = 1.0
        return full / rows

    def predict(self, X: Views) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
