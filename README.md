# SIT720 11.1HD — Reproducing MACE, and a Soft-Fusion Alternative

Jason Hu (219220123)

Reproduction of, and a proposed improvement on:

> F. Montori, K. Liao, M. De Giosa, P. P. Jayaraman, L. Bononi, T. Sellis and D. Georgakopoulos,
> "A Metadata-Assisted Cascading Ensemble Classification Framework for Automatic Annotation of
> Open IoT Data," *IEEE Internet of Things Journal*, vol. 10, no. 15, pp. 13401–13413, Aug. 2023.
> DOI: [10.1109/JIOT.2023.3262365](https://doi.org/10.1109/JIOT.2023.3262365)

**Part 1** reproduces the paper's MACE cascading ensemble on all three of its open IoT data sets.
**Part 2** proposes **MACE-SF**, a soft-fusion imbalance-aware cascade that addresses three
limitations the reproduction exposes.

---

## Quick start

```bash
python -m pip install -r requirements.txt
jupyter notebook 11.1HD.ipynb        # Run All, about 15 minutes
```

Run the notebook **from this folder**, so the relative path `data/` resolves. Everything is
deterministic (`random_state = 42` throughout); a second run reproduces every reported figure to
four decimal places.

## What is here

| Path | Contents |
|---|---|
| `11.1HD.ipynb` | The full study: Parts 1 and 2, all tables and figures |
| `src/macelib/data.py` | Data loading and the eleven bag-of-summaries features |
| `src/macelib/classifiers.py` | 1NN-ED, DDL-NLP and the bag-of-summaries wrapper |
| `src/macelib/pools.py` | Grid searches and the classifier pools |
| `src/macelib/mace.py` | MACE: top-accuracy curves, both heuristics, three filtering strategies |
| `src/macelib/proposed.py` | MACE-SF, the proposed method |
| `src/macelib/experiment.py` | End-to-end runner and the comparison against the published tables |
| `data/` | The authors' three data sets, with their own 70/30 split |

## Data

The data sets are the authors' own public release, used unmodified, including their stratified
70/30 train/test split — so no splitting decision of mine can flatter the comparison.

* Data: <https://github.com/stradivarius/TSopendatastreams>
* Authors' reference code: <https://github.com/matteodeggi/IoT_Classification>
  (consulted for algorithmic detail; not redistributed here, as its licence is unstated)

Each `.tkse` file is a headerless CSV: nine metadata columns (id, name, channel id, channel name,
description, timestamp, latitude, longitude, class) followed by the readings.

| Data set | Streams | Readings | Classes | Metadata | Imbalance |
|---|---|---|---|---|---|
| ThingSpeak (TS) | 2,121 | 96 | 21 | yes | 108:1 |
| Swissex | 346 | 445 | 11 | no | 5.6:1 |
| Urban Observatory | 1,065 | 864 | 16 | no | 16.3:1 |

Only ThingSpeak carries metadata, so it is the only set on which a genuinely multi-domain ensemble
can be demonstrated — which is also where both MACE and MACE-SF show their largest gains.

## Documented assumptions

The paper leaves several details unspecified. Each of these is argued in the report.

1. **"Eleven features" but ten names.** Section IV-B2 lists mean, median, maximum, minimum,
   standard deviation, RMS, quantile, IQR, kurtosis and range. "Quantile" is read as the lower and
   upper quartiles, which makes the count eleven and the IQR internally consistent.
2. **Macro rather than weighted F1.** The paper never says. The reproduction settles it
   empirically: soft voting on UrbObs gives a macro F1 of 0.8635 against the published 0.864,
   while the weighted F1 is 0.8935.
3. **The Top-k threshold excludes saturated k.** Every `A_i(c)` equals 1 by construction, so the
   domination test is trivially true there and would return no filtering at all. Restricting the
   search to where the preceding classifier has not yet reached perfect top-k recall returns
   exactly the **k = 7** the paper reports in its UrbObs worked example.
4. **Grid search spaces.** The paper says the bag-of-summaries algorithms were "optimized via a
   grid search" without listing the grids; conventional ones are used and printed in the notebook.
5. **Brute-force optimum.** Ambiguous in the paper. Both readings are computed — the combination
   selected by cross-validation, and the best achievable on the test split.
6. **BOPF, SDE, LTS and 1NN-DTW are not reproduced.** The paper runs BOPF from the authors' C++
   code, reports that all three "fail across IoT data sets", and excludes every one of them from
   the MACE pool, so they cannot affect Tables I or II.

Two defects in the authors' released code are noted but deliberately not reproduced: `dictionary`
is declared at class scope in both `TSC_1NN` and `NLP_Classifier`, making it shared mutable state
across instances, and class order is taken from `set(y_train)` iteration rather than sorted.

## Reproducing a single result

```python
import sys; sys.path.insert(0, "src")
from macelib.experiment import run_dataset, comparison_table

run = run_dataset("TS", root="data")      # ThingSpeak, about five minutes
print(comparison_table(run).round(4))     # reproduced values beside the published ones
```

```python
from macelib.proposed import MACESoftFusion

Xtr, ytr, Xte, yte = run["views"]
sf = MACESoftFusion(run["mace"].pool, random_state=42).fit(Xtr, ytr)
print(sf.best_.describe())
print(sf.score(Xte, yte))
print(sf.ablation(Xte, yte))
```

## Requirements

Python 3.10 or newer. `rapidfuzz` supplies the Damerau–Levenshtein distance that DDL-NLP needs; the
authors used `pyxdameraulevenshtein`, which no longer builds on current Python.

## Deliverables

* Technical report: `SIT720-11.1HD-Report.pdf`
* Video presentation: link in the report
* This repository: code, data and instructions
