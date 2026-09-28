"""Classifier pools, with the grid search the paper specifies for the BOS algorithms.

Section IV-B2 states the bag-of-summaries algorithms "have been optimized via a grid
search on their parameters" without listing the grids, so conventional grids are used
here and documented in the report. The search is run once per data set on the training
split with stratified 5-fold CV on accuracy, and the winning parameters are then held
fixed, which keeps the later cross-validation steps of MACE comparable across
classifiers.

Not reproduced: BOPF, SDE and LTS. The paper runs BOPF from the authors' own C++ code
and reports that all three shapelet- and dictionary-based methods "fail across IoT data
sets"; none of them appears in the MACE pool. Their omission therefore does not affect
Tables I and II, and is justified in the report.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .classifiers import BOSClassifier, DDLNLPClassifier, NearestNeighbourED

RS = 42

# name -> (estimator, grid, needs feature scaling)
BOS_SPECS = {
    "DecisionTree": (DecisionTreeClassifier(random_state=RS),
                     {"max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 5]}, False),
    "SVM": (SVC(probability=True, random_state=RS),
            {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1]}, True),
    "kNN": (KNeighborsClassifier(),
            {"n_neighbors": [1, 3, 5, 7, 9], "weights": ["uniform", "distance"]}, True),
    "LogReg": (LogisticRegression(max_iter=2000, random_state=RS),
               {"C": [0.1, 1, 10]}, True),
    "Ridge": (RidgeClassifier(random_state=RS), {"alpha": [0.1, 1, 10]}, True),
    "GaussianNB": (GaussianNB(), {"var_smoothing": [1e-9, 1e-7, 1e-5]}, True),
    "RF": (RandomForestClassifier(random_state=RS, n_jobs=-1),
           {"n_estimators": [100, 300], "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2]}, False),
    "GradBoost": (GradientBoostingClassifier(random_state=RS),
                  {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                   "max_depth": [3, 5]}, False),
}


def tune_bos(bos_features: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """Grid-search every BOS algorithm once on the training split."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RS)
    best = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (est, grid, scale) in BOS_SPECS.items():
            steps = ([("scale", StandardScaler())] if scale else []) + [("clf", est)]
            pipe = Pipeline(steps)
            search = GridSearchCV(pipe, {f"clf__{k}": v for k, v in grid.items()},
                                  cv=cv, scoring="accuracy", n_jobs=-1)
            search.fit(bos_features, y)
            best[name] = {k.replace("clf__", ""): v
                          for k, v in search.best_params_.items()}
            best[name]["_cv_accuracy"] = float(search.best_score_)
    return best


def _make_bos(name: str, params: dict):
    est, _, scale = BOS_SPECS[name]
    clean = {k: v for k, v in params.items() if not k.startswith("_")}
    return lambda: BOSClassifier(est.__class__(**{**est.get_params(), **clean}), scale=scale)


def standalone_pool(best: dict, has_metadata: bool) -> dict:
    """Every standalone classifier reproduced, for the Section V-A comparison."""
    pool = {
        "1NN-ED": lambda: NearestNeighbourED(normalise=False),
        "1NN-ED (z-norm)": lambda: NearestNeighbourED(normalise=True),
    }
    for name in BOS_SPECS:
        pool[name] = _make_bos(name, best.get(name, {}))
    if has_metadata:
        pool["DDL-NLP"] = lambda: DDLNLPClassifier()
    return pool


def mace_pool(best: dict, has_metadata: bool) -> dict:
    """The five-classifier pool the paper uses for Tables I and II.

    Gamma = {1NN-ED, GradBoost, RF, kNN, SVM} for UrbObs and Swissex, with DDL-NLP
    replacing SVM for ThingSpeak so that the ensemble spans two information domains.
    """
    pool = {
        "1NN-ED": lambda: NearestNeighbourED(normalise=False),
        "GradBoost": _make_bos("GradBoost", best.get("GradBoost", {})),
        "RF": _make_bos("RF", best.get("RF", {})),
        "kNN": _make_bos("kNN", best.get("kNN", {})),
    }
    pool["DDL-NLP" if has_metadata else "SVM"] = (
        (lambda: DDLNLPClassifier()) if has_metadata
        else _make_bos("SVM", best.get("SVM", {})))
    return pool
