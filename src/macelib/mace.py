"""The MACE framework of Montori et al. (IEEE IoT Journal, 2023).

MACE is a cascade. Each stage produces a ranking of classes; a filtering function keeps
the most probable of them, and the next stage may only choose from that survivor set,
intersecting its own ranking with the filter in an order-preserving way. Training is two
stratified 5-fold cross-validation passes: the first fixes which classifiers are used
and in what order, the second tunes the single filtering parameter z.

Implemented here:
  * top-accuracy curves A_i(k)                     (Section III-C)
  * the dominating heuristic for selection          (Equation 4)
  * the backward search heuristic for ordering      (Equations 5-6)
  * the Top-k, PF and SoF filtering strategies      (Section III-D)
  * soft-voting and brute-force baselines           (Section V-B)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold

STRATEGIES = ("Top-k", "PF", "SoF")
PF_GRID = tuple(np.round(np.arange(0.10, 1.00, 0.05), 2))
SOF_GRID = tuple(np.round(np.arange(-1.50, 2.01, 0.10), 2))


# ------------------------------------------------------------------ top accuracy
def top_accuracy_curve(proba: np.ndarray, y_true: np.ndarray,
                       classes: np.ndarray) -> np.ndarray:
    """A_i(k) for k = 1..c: how often the true class is inside the top-k ranking."""
    order = np.argsort(-proba, axis=1, kind="stable")
    truth = np.searchsorted(classes, y_true)
    hit_rank = (order == truth[:, None]).argmax(axis=1)
    c = proba.shape[1]
    return np.array([(hit_rank < k).mean() for k in range(1, c + 1)])


def out_of_fold_proba(make_clf, X, y, folds) -> np.ndarray:
    """Stratified K-fold out-of-fold probabilities, aligned to sorted class order."""
    classes = np.unique(y)
    oof = np.zeros((len(y), len(classes)))
    for tr, va in folds:
        clf = make_clf().fit(X.subset(tr), y[tr])
        p = clf.predict_proba(X.subset(va))
        if p.shape[1] != len(classes):          # a fold can miss a very rare class
            full = np.zeros((p.shape[0], len(classes)))
            for j, cl in enumerate(clf.classes_):
                full[:, np.searchsorted(classes, cl)] = p[:, j]
            p = full
        oof[va] = p
    return oof


# ------------------------------------------------------------- selection/ordering
def dominating_selection(curves: dict[str, np.ndarray]) -> list[str]:
    """Equation 4: drop classifier i when some j is at least as good at every k."""
    names = list(curves)
    keep = []
    for i in names:
        others = [curves[j] for j in names if j != i]
        if not others:
            keep.append(i)
            continue
        best_other = np.max(np.vstack(others), axis=0)
        if np.any(curves[i] > best_other + 1e-12):   # strictly best for some k
            keep.append(i)
    return keep or names


def backward_order(curves: dict[str, np.ndarray], names: list[str]) -> list[str]:
    """Equations 5-6: walk k from c-1 down to 1, taking each new unique dominator."""
    if len(names) <= 1:
        return list(names)
    c = len(next(iter(curves.values())))
    order: list[str] = []
    for k in range(c - 1, 0, -1):
        vals = np.array([curves[n][k - 1] for n in names])
        best = vals.max()
        winners = [n for n, v in zip(names, vals) if v >= best - 1e-12]
        if len(winners) != 1:                  # ties are not allowed
            continue
        if winners[0] not in order:
            order.append(winners[0])
        if len(order) == len(names):
            break
    for n in names:                            # anything never uniquely dominant
        if n not in order:
            order.append(n)
    return order


def topk_thresholds(curves: dict[str, np.ndarray], order: list[str]) -> list[int]:
    """Largest k at which the next stage dominates the current one.

    The search is confined to the region where filtering can still do work, namely the
    k at which the preceding classifier has not yet reached perfect top-k recall. Once
    A_a(k) = 1 the filter contains every class the next stage could need and discards
    nothing, so the domination test is trivially satisfied from there to k = c and would
    always return c, i.e. no filtering at all.

    The paper does not state this restriction, but its own worked example requires it:
    for UrbObs it reports k = 7, which is precisely the largest k below the point where
    1NN-ED saturates. Taking the unrestricted maximum would give 15 or 16 instead.
    """
    c = len(next(iter(curves.values())))
    ks = []
    for a, b in zip(order, order[1:]):
        informative = [k for k in range(1, c + 1) if curves[a][k - 1] < 1.0 - 1e-12]
        limit = max(informative) if informative else 1
        ok = [k for k in range(1, limit + 1)
              if curves[b][k - 1] >= curves[a][k - 1] - 1e-12]
        ks.append(max(ok) if ok else 1)
    return ks


# ------------------------------------------------------------------- the cascade
def _filter(rank: np.ndarray, proba_row: np.ndarray, strategy: str, z: float,
            k_fixed: int | None) -> np.ndarray:
    """Return the survivor classes from the current ranking."""
    if strategy == "Top-k":
        keep = max(1, min(int(k_fixed), len(rank)))
        return rank[:keep]
    if strategy == "PF":
        keep = max(1, int(np.ceil(z * len(rank))))
        return rank[:keep]
    if strategy == "SoF":
        p = proba_row[rank]
        s = p.sum()
        p = p / s if s > 0 else np.full_like(p, 1.0 / len(p))
        thr = p.mean() + z * p.std()
        survivors = rank[p >= thr]
        return survivors if survivors.size else rank[:1]
    raise ValueError(f"unknown strategy {strategy!r}")


def cascade_predict(probas: list[np.ndarray], strategy: str, z: float,
                    k_fixed: list[int] | None = None,
                    return_scores: bool = False):
    """Run the cascade over a list of (n, c) probability matrices, in stage order.

    Vectorised over examples. The survivor set is carried as a boolean mask; at each
    stage the current classifier ranks only the surviving classes, the filter keeps the
    strongest of them, and the next stage may choose only from those. The final
    prediction is the surviving class the last stage ranks highest, which is exactly the
    top-1 of C_out,p in the paper's notation.

    Returns the index into the class axis of the top-1 prediction per example.
    """
    n, c = probas[0].shape
    rows = np.arange(n)
    allowed = np.ones((n, c), dtype=bool)

    for stage in range(len(probas) - 1):
        p = probas[stage]
        scores = np.where(allowed, p, -np.inf)
        order = np.argsort(-scores, axis=1, kind="stable")
        pos = np.empty((n, c), dtype=int)
        pos[rows[:, None], order] = np.arange(c)[None, :]
        size = allowed.sum(axis=1)

        if strategy == "SoF":
            masked = np.where(allowed, p, 0.0)
            total = masked.sum(axis=1, keepdims=True)
            total[total == 0] = 1.0
            q = masked / total
            mean = q.sum(axis=1) / np.maximum(size, 1)
            var = np.where(allowed, (q - mean[:, None]) ** 2, 0.0).sum(axis=1) \
                / np.maximum(size, 1)
            thr = mean + z * np.sqrt(var)
            keep = allowed & (q >= thr[:, None])
            empty = ~keep.any(axis=1)
            if empty.any():                      # never let a filter empty out
                keep[empty] = allowed[empty] & (pos[empty] == 0)
            allowed = keep
        else:
            if strategy == "Top-k":
                k = np.full(n, max(1, int(k_fixed[stage])))
            elif strategy == "PF":
                k = np.maximum(1, np.ceil(z * size).astype(int))
            else:
                raise ValueError(f"unknown strategy {strategy!r}")
            k = np.minimum(k, np.maximum(size, 1))
            allowed = allowed & (pos < k[:, None])

    final = np.where(allowed, probas[-1], -np.inf)
    pred = np.argmax(final, axis=1)
    if not return_scores:
        return pred
    # The distribution the cascade actually emits: the last stage restricted to the
    # survivors and renormalised. Every filtered-out class carries zero mass, which is
    # the whole point of a hard filter and must be visible to a ranking metric such as
    # AUC. Scoring the last stage's unrestricted output instead would measure that
    # classifier alone and hide the effect of the cascade entirely.
    scores = np.where(allowed, probas[-1], 0.0)
    total = scores.sum(axis=1, keepdims=True)
    # A row can survive filtering yet carry no mass, because tree ensembles emit exact
    # zeros: if every surviving class happens to have probability 0 under the last
    # stage, the masked row is empty. The cascade expresses no preference among those
    # survivors, so the honest score is uniform over them rather than an undefined row.
    empty = (total[:, 0] == 0)
    if empty.any():
        scores[empty] = allowed[empty].astype(float)
        total[empty, 0] = scores[empty].sum(axis=1)
    return pred, scores / total


# ------------------------------------------------------------------------ metrics
def evaluate(y_true, y_pred, proba=None, classes=None) -> dict:
    """Accuracy, precision, recall, F1 (macro and weighted) and macro one-vs-rest AUC.

    The paper reports only accuracy and F1 and does not state its averaging scheme, so
    both macro and weighted are reported here; the task additionally requires precision,
    recall and AUC.
    """
    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    res["auc_macro_ovr"] = np.nan
    if proba is not None and classes is not None:
        try:
            present = np.isin(classes, np.unique(y_true))
            if present.sum() > 1:
                p = proba[:, present]
                p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
                res["auc_macro_ovr"] = roc_auc_score(
                    y_true, p, multi_class="ovr", average="macro",
                    labels=classes[present])
        except ValueError:
            pass
    return res


# --------------------------------------------------------------------------- MACE
@dataclass
class MACEResult:
    strategy: str
    order: list[str]
    z: float
    cv_accuracy: float
    k_fixed: list[int] | None = None
    metrics: dict = field(default_factory=dict)


class MACE:
    """The metadata-assisted cascading ensemble.

    `pool` maps a classifier name to a zero-argument factory returning a fresh estimator.
    """

    def __init__(self, pool: dict, n_splits: int = 5, random_state: int = 42):
        self.pool = pool
        self.n_splits = n_splits
        self.random_state = random_state

    # -- training -------------------------------------------------------------
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.random_state)
        self._folds = list(skf.split(np.zeros(len(y)), y))
        self._y = np.asarray(y)

        self.oof_ = {n: out_of_fold_proba(f, X, self._y, self._folds)
                     for n, f in self.pool.items()}
        self.curves_ = {n: top_accuracy_curve(p, self._y, self.classes_)
                        for n, p in self.oof_.items()}

        self.selected_ = dominating_selection(self.curves_)
        self.order_ = backward_order(self.curves_, self.selected_)
        self.topk_ = topk_thresholds(self.curves_, self.order_)

        self.fitted_ = {n: f().fit(X, self._y) for n, f in self.pool.items()}
        return self

    def _cv_accuracy(self, order, strategy, z, k_fixed):
        """Second CV step, scored on the out-of-fold probabilities already computed."""
        probas = [self.oof_[n] for n in order]
        pred_idx = cascade_predict(probas, strategy, z, k_fixed)
        return accuracy_score(self._y, self.classes_[pred_idx])

    def tune(self, strategy: str, order: list[str] | None = None) -> MACEResult:
        order = order or self.order_
        if strategy == "Top-k":
            k_fixed = topk_thresholds(self.curves_, order)
            acc = self._cv_accuracy(order, strategy, 0.0, k_fixed)
            return MACEResult(strategy, order, float("nan"), acc, k_fixed)
        grid = PF_GRID if strategy == "PF" else SOF_GRID
        best = max(((z, self._cv_accuracy(order, strategy, z, None)) for z in grid),
                   key=lambda t: t[1])
        return MACEResult(strategy, order, float(best[0]), best[1], None)

    # -- prediction -----------------------------------------------------------
    def predict_proba_stages(self, X, order):
        return [self.fitted_[n].predict_proba(X) for n in order]

    def predict(self, X, result: MACEResult) -> np.ndarray:
        probas = self.predict_proba_stages(X, result.order)
        idx = cascade_predict(probas, result.strategy, result.z, result.k_fixed)
        return self.classes_[idx]

    def score(self, X, y, result: MACEResult) -> dict:
        probas = self.predict_proba_stages(X, result.order)
        idx, scores = cascade_predict(probas, result.strategy, result.z,
                                      result.k_fixed, return_scores=True)
        return evaluate(y, self.classes_[idx], scores, self.classes_)

    # -- baselines ------------------------------------------------------------
    def soft_voting(self, X, y, names=None) -> dict:
        names = names or list(self.pool)
        proba = np.mean([self.fitted_[n].predict_proba(X) for n in names], axis=0)
        return evaluate(y, self.classes_[proba.argmax(axis=1)], proba, self.classes_)

    def brute_force(self, strategy: str = "SoF", max_len: int | None = None):
        """Every ordering of every non-empty subset of the pool, scored by CV accuracy."""
        names = list(self.pool)
        max_len = max_len or len(names)
        best = None
        for r in range(1, max_len + 1):
            for order in permutations(names, r):
                order = list(order)
                if r == 1:
                    acc = accuracy_score(
                        self._y, self.classes_[self.oof_[order[0]].argmax(axis=1)])
                    cand = MACEResult(strategy, order, float("nan"), acc, None)
                else:
                    cand = self.tune(strategy, order)
                if best is None or cand.cv_accuracy > best.cv_accuracy:
                    best = cand
        return best
