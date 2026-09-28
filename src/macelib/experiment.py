"""End-to-end Part 1 reproduction of Montori et al. across the three IoT data sets."""
from __future__ import annotations

import time
import warnings
from itertools import permutations

import numpy as np
import pandas as pd

from .classifiers import Views
from .data import bag_of_summaries, load
from .mace import MACE, STRATEGIES, cascade_predict, evaluate
from .pools import mace_pool, standalone_pool, tune_bos

# what the paper reports, for side-by-side comparison (accuracy, F1)
PAPER = {
    "UrbObs": {"Best standalone": (0.881, 0.879), "Soft Voting": (0.891, 0.864),
               "MACE Top-k": (0.906, 0.879), "MACE PF": (0.906, 0.879),
               "MACE SoF": (0.906, 0.879), "MACE brute-force (SoF)": (0.922, 0.887)},
    "Swissex": {"Best standalone": (0.788, 0.832), "Soft Voting": (0.837, 0.816),
                "MACE Top-k": (0.846, 0.832), "MACE PF": (0.832, 0.832),
                "MACE SoF": (0.827, 0.800), "MACE brute-force (SoF)": (0.865, 0.840)},
    "TS": {"Best standalone": (0.757, 0.719), "Soft Voting": (0.796, 0.578),
           "MACE Top-k": (0.843, 0.677), "MACE PF": (0.834, 0.637),
           "MACE SoF": (0.882, 0.794), "MACE brute-force (SoF)": (0.885, 0.767)},
}


def make_views(split) -> Views:
    return Views(series=split.series, bos=bag_of_summaries(split.series), text=split.name)


def run_dataset(key: str, root: str = "data", random_state: int = 42,
                verbose: bool = True) -> dict:
    """Reproduce the full pipeline for one data set."""
    t_start = time.perf_counter()
    ds = load(key, root=root)
    Xtr, ytr = make_views(ds.train), ds.train.labels
    Xte, yte = make_views(ds.test), ds.test.labels
    say = print if verbose else (lambda *a, **k: None)
    say(f"=== {key}: {ds.summary()}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        best_params = tune_bos(Xtr.bos, ytr)
        say(f"  grid search {time.perf_counter()-t0:.0f}s")

        # ---- Section V-A: every standalone classifier, on the test split
        standalone = {}
        for name, factory in standalone_pool(best_params, ds.has_metadata).items():
            t0 = time.perf_counter()
            clf = factory().fit(Xtr, ytr)
            p = clf.predict_proba(Xte)
            m = evaluate(yte, clf.classes_[p.argmax(1)], p, clf.classes_)
            m["seconds"] = time.perf_counter() - t0
            standalone[name] = m
        say(f"  {len(standalone)} standalone classifiers evaluated")

        # ---- Sections V-B and V-C: the MACE pool
        mace = MACE(mace_pool(best_params, ds.has_metadata),
                    random_state=random_state).fit(Xtr, ytr)
        say(f"  pool={list(mace.pool)}")
        say(f"  selected={mace.selected_} order={mace.order_} top-k={mace.topk_}")

        pool_test = {}
        for name in mace.pool:
            p = mace.fitted_[name].predict_proba(Xte)
            pool_test[name] = evaluate(yte, mace.classes_[p.argmax(1)], p, mace.classes_)

        results = {}
        bs = max(pool_test.items(), key=lambda kv: kv[1]["accuracy"])
        results["Best standalone"] = dict(bs[1], detail=bs[0])
        results["Soft Voting"] = mace.soft_voting(Xte, yte)

        chosen = {}
        for strat in STRATEGIES:
            res = mace.tune(strat)
            chosen[strat] = res
            results[f"MACE {strat}"] = dict(
                mace.score(Xte, yte, res),
                detail=f"order={'>'.join(res.order)} z={res.z:.2f} k={res.k_fixed}"
                       if res.z == res.z else f"order={'>'.join(res.order)} k={res.k_fixed}")

        # brute force, read two ways (see report): selected by CV, and the oracle best
        bf_cv, bf_oracle = _brute_force(mace, Xte, yte)
        results["MACE brute-force (SoF)"] = bf_cv
        results["MACE brute-force (oracle test)"] = bf_oracle

    say(f"  total {time.perf_counter()-t_start:.0f}s\n")
    return {"dataset": key, "summary": ds.summary(), "best_params": best_params,
            "standalone": standalone, "pool_test": pool_test, "results": results,
            "curves": {k: v.tolist() for k, v in mace.curves_.items()},
            "selected": mace.selected_, "order": mace.order_, "topk": mace.topk_,
            "chosen": {k: vars(v) for k, v in chosen.items()},
            "mace": mace, "views": (Xtr, ytr, Xte, yte), "n_classes": ds.n_classes}


def _brute_force(mace: MACE, Xte, yte):
    """Every ordering of every subset, under SoF.

    The paper's "brute-force optimum" is ambiguous: it may mean the combination chosen
    by cross-validation, or the best accuracy achievable on the test split. Both are
    computed; the report compares them and argues the published figure matches the
    latter.
    """
    names = list(mace.pool)
    test_proba = {n: mace.fitted_[n].predict_proba(Xte) for n in names}
    best_cv, best_oracle = None, None
    for r in range(1, len(names) + 1):
        for order in permutations(names, r):
            order = list(order)
            res = mace.tune("SoF", order) if r > 1 else None
            if r == 1:
                from sklearn.metrics import accuracy_score
                cv = accuracy_score(mace._y,
                                    mace.classes_[mace.oof_[order[0]].argmax(1)])
                z = float("nan")
            else:
                cv, z = res.cv_accuracy, res.z
            probas = [test_proba[n] for n in order]
            if r > 1:
                idx, sc_p = cascade_predict(probas, "SoF", z, return_scores=True)
            else:
                idx, sc_p = probas[0].argmax(axis=1), probas[0]
            sc = evaluate(yte, mace.classes_[idx], sc_p, mace.classes_)
            sc["detail"] = f"order={'>'.join(order)} z={z:.2f}" if z == z else f"order={order[0]}"
            if best_cv is None or cv > best_cv[0]:
                best_cv = (cv, sc)
            if best_oracle is None or sc["accuracy"] > best_oracle["accuracy"]:
                best_oracle = sc
    return best_cv[1], best_oracle


def comparison_table(run: dict) -> pd.DataFrame:
    """Reproduced accuracy and macro F1 beside the values printed in the paper."""
    key, rows = run["dataset"], []
    for name, m in run["results"].items():
        paper = PAPER[key].get(name)
        rows.append({
            "strategy": name,
            "acc (ours)": m["accuracy"],
            "acc (paper)": paper[0] if paper else np.nan,
            "acc diff": m["accuracy"] - paper[0] if paper else np.nan,
            "F1 macro (ours)": m["f1_macro"],
            "F1 (paper)": paper[1] if paper else np.nan,
            "F1 diff": m["f1_macro"] - paper[1] if paper else np.nan,
            "precision": m["precision_macro"],
            "recall": m["recall_macro"],
            "AUC": m["auc_macro_ovr"],
        })
    return pd.DataFrame(rows).set_index("strategy")
