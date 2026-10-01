"""MACE-SF: a soft-fusion, imbalance-aware cascade (Part 2 of the HD task).

Three limitations of MACE are addressed, each evidenced by the Part 1 reproduction.

1. Hard filtering is irrecoverable. Once a class is removed from C_out it can never be
   predicted, however confident a later stage is. On UrbObs the truth leaves 1NN-ED's
   top-k filter for 1.1% of examples and those are lost outright, which is why the
   cascade there scores below the classifier it ends on.

2. Filtering destroys the probability distribution. The reproduction measures a macro
   one-vs-rest AUC of 0.924 for MACE on ThingSpeak against 0.962 for soft voting: MACE
   makes good top-1 decisions but the scores behind them are no longer usable for
   ranking, thresholding or abstention, which an annotation service needs.

3. The heuristics optimise accuracy only. The authors state F1 is "a side-effect".
   ThingSpeak carries 21 classes in a 108:1 imbalance, and the reproduction shows tree
   classifiers reaching 0.71 accuracy at 0.43 macro F1 - the rare sensor types that most
   need automatic annotation are exactly the ones being missed.

MACE-SF keeps the cascading idea and changes the mechanism:

  * soft (leaky) filtering - a filtered class keeps a small retention weight eps instead
    of being deleted, so the cascade carries a belief rather than a surviving set and an
    early mistake can still be overturned. MACE is the eps -> 0 special case.
  * multiplicative belief fusion with a stage exponent, so every stage contributes
    evidence in proportion to its confidence rather than acting as a gate.
  * logit adjustment by class prior, the standard correction for long-tailed problems
    [Menon et al., ICLR 2021], applied once to the fused belief.
  * macro F1 as the objective for selection, ordering and tuning, replacing accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .mace import evaluate, out_of_fold_proba

# eps = 0 recovers MACE's hard filter; keep = 1.0 removes filtering entirely, so both
# the original behaviour and the no-cascade limit sit inside the search space and the
# tuner is free to reject the proposed mechanism if it does not help.
EPS_GRID = (0.0, 0.02, 0.05, 0.15, 0.35)
GAMMA_GRID = (0.5, 1.0, 1.5)
TAU_GRID = (0.0, 0.25, 0.5, 1.0)
KEEP_GRID = (0.25, 0.4, 0.6, 1.0)
# log-linear pooling multiplies the experts and suits complementary information
# domains; linear pooling averages them and is the more robust choice when the
# experts are correlated, as they are when every classifier reads the same features.
# Leaving the rule to cross-validation lets one method cover both regimes, and makes
# soft voting itself a reachable special case (mode="lin", eps=1, gamma=1).
MODE_GRID = ("log", "lin")


def macro_f1_curve(proba: np.ndarray, y_true: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """A macro-averaged analogue of the paper's top-accuracy curve A_i(k).

    For each k the top-k hit indicator is averaged within each class and then across
    classes, so a rare class counts as much as a common one. This is the quantity the
    selection and ordering heuristics maximise in MACE-SF.
    """
    order = np.argsort(-proba, axis=1, kind="stable")
    truth = np.searchsorted(classes, y_true)
    hit_rank = (order == truth[:, None]).argmax(axis=1)
    c = proba.shape[1]
    out = np.zeros(c)
    for k in range(1, c + 1):
        hit = hit_rank < k
        per_class = [hit[y_true == cl].mean() for cl in classes if (y_true == cl).any()]
        out[k - 1] = float(np.mean(per_class))
    return out


def soft_cascade(probas: list[np.ndarray], eps: float, gamma: float, keep: float,
                 tau: float, priors: np.ndarray, mode: str = "log") -> np.ndarray:
    """Propagate a belief through the stages and return the fused distribution.

    The cascade is genuinely sequential and order dependent, mirroring MACE: the filter
    at each step is applied to the *running belief* - the analogue of the paper's
    C_out,i - and not to the incoming classifier in isolation. Filtering a stage's own
    output instead would make the product commutative and the ensemble would stop being
    a cascade at all.

    A class outside the running belief's top `keep` fraction is damped by `eps` rather
    than deleted, so a later stage can still recover it. eps = 0 recovers hard
    filtering of the same kind MACE applies, though the survivor set is fixed by a
    keep fraction rather than by MACE's Top-k, PF or SoF rules; eps = 1 removes
    filtering altogether and leaves a pure pooling of experts.
    """
    n, c = probas[0].shape
    rows = np.arange(n)
    k = max(1, int(np.ceil(keep * c)))

    belief = np.clip(probas[0], 1e-12, None) ** gamma
    belief = belief / belief.sum(axis=1, keepdims=True)

    for p in probas[1:]:
        order = np.argsort(-belief, axis=1, kind="stable")     # rank the running belief
        pos = np.empty((n, c), dtype=int)
        pos[rows[:, None], order] = np.arange(c)[None, :]
        belief = belief * np.where(pos < k, 1.0, eps)          # soft filter, not a gate
        belief = belief / np.clip(belief.sum(axis=1, keepdims=True), 1e-12, None)
        q = np.clip(p, 1e-12, None) ** gamma
        belief = belief * q if mode == "log" else belief + q   # fuse the next stage
        total = belief.sum(axis=1, keepdims=True)
        total[total == 0] = 1.0
        belief = belief / total

    if tau > 0:                    # logit adjustment for the long tail
        belief = belief / np.clip(priors, 1e-12, None) ** tau
        belief = belief / belief.sum(axis=1, keepdims=True)
    return belief


@dataclass
class SFResult:
    order: list[str]
    eps: float
    gamma: float
    keep: float
    tau: float
    mode: str
    cv_macro_f1: float
    metrics: dict = field(default_factory=dict)

    def describe(self) -> str:
        return (f"order={'>'.join(self.order)} eps={self.eps} gamma={self.gamma} "
                f"keep={self.keep} tau={self.tau} mode={self.mode}")


class MACESoftFusion:
    """The proposed method. Same interface as MACE, different mechanism and objective."""

    def __init__(self, pool: dict, n_splits: int = 5, random_state: int = 42,
                 max_stages: int = 4):
        self.pool = pool
        self.n_splits = n_splits
        self.random_state = random_state
        self.max_stages = max_stages

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._y = y
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.random_state)
        folds = list(skf.split(np.zeros(len(y)), y))

        self.oof_ = {n: out_of_fold_proba(f, X, y, folds) for n, f in self.pool.items()}
        self.curves_ = {n: macro_f1_curve(p, y, self.classes_) for n, p in self.oof_.items()}
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        self.priors_ = counts / counts.sum()

        # selection: keep a classifier that is strictly best at some k on the macro
        # curve, so a specialist on rare classes survives even if its accuracy is poor
        names = list(self.curves_)
        keep = []
        for i in names:
            others = [self.curves_[j] for j in names if j != i]
            if not others or np.any(self.curves_[i] > np.max(np.vstack(others), axis=0) + 1e-12):
                keep.append(i)
        self.selected_ = keep or names

        self.fitted_ = {n: f().fit(X, y) for n, f in self.pool.items()}
        self.best_ = self._search()
        return self

    def _cv_macro_f1(self, order, eps, gamma, keep, tau, mode) -> float:
        belief = soft_cascade([self.oof_[n] for n in order], eps, gamma, keep, tau,
                              self.priors_, mode)
        pred = self.classes_[belief.argmax(axis=1)]
        return f1_score(self._y, pred, average="macro", zero_division=0)

    def _search(self) -> SFResult:
        """Tune the ordering and the four parameters on cross-validated macro F1."""
        best = None
        pool = self.selected_
        orders = [list(o) for r in range(1, min(self.max_stages, len(pool)) + 1)
                  for o in permutations(pool, r)]
        for order in orders:
            for eps in EPS_GRID:
                for gamma in GAMMA_GRID:
                    for keep in KEEP_GRID:
                        for tau in TAU_GRID:
                            for mode in MODE_GRID:
                                s = self._cv_macro_f1(order, eps, gamma, keep, tau, mode)
                                if best is None or s > best.cv_macro_f1:
                                    best = SFResult(order, eps, gamma, keep, tau, mode, s)
        return best

    def predict_proba(self, X, result: SFResult | None = None) -> np.ndarray:
        r = result or self.best_
        probas = [self.fitted_[n].predict_proba(X) for n in r.order]
        return soft_cascade(probas, r.eps, r.gamma, r.keep, r.tau, self.priors_, r.mode)

    def predict(self, X, result: SFResult | None = None) -> np.ndarray:
        return self.classes_[self.predict_proba(X, result).argmax(axis=1)]

    def score(self, X, y, result: SFResult | None = None) -> dict:
        proba = self.predict_proba(X, result)
        return evaluate(y, self.classes_[proba.argmax(axis=1)], proba, self.classes_)

    def ablation(self, X, y) -> dict:
        """Turn each contribution off in turn, to show which part is doing the work."""
        b = self.best_
        variants = {
            "MACE-SF (full)": b,
            "  without soft filtering (eps=0)": SFResult(b.order, 0.0, b.gamma, b.keep, b.tau, b.mode, 0),
            "  without logit adjustment (tau=0)": SFResult(b.order, b.eps, b.gamma, b.keep, 0.0, b.mode, 0),
            "  without filtering at all (keep=1)": SFResult(b.order, b.eps, b.gamma, 1.0, b.tau, b.mode, 0),
        }
        return {name: self.score(X, y, r) for name, r in variants.items()}
