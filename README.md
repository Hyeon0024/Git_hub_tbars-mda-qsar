# QSAR models for TBARS/MDA lipid-peroxidation inhibition

[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-brightgreen.svg)](LICENSE-DATA)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](requirements.txt)

Curated data, the fitted model, the validation code, and every table and
figure reported in the accompanying article. A Gradient Boosting regressor on
217 RDKit 2D descriptors predicts pIC₅₀ for inhibition of lipid peroxidation
measured as TBARS or malondialdehyde (MDA).

The dataset is a retrospective compilation of published IC₅₀ values. **No new
measurements were made for this work.**

> **Before using the predictions, read [Known limits](#known-limits).** The
> endpoint pools assay protocols, and performance on an independent external
> set is far worse than the internal holdout suggests. These numbers are
> reported here rather than buried.

---

## Results at a glance

| Design | n | R² | RMSE | MAE |
|---|---:|---:|---:|---:|
| Training, pooled out-of-fold | 302 | 0.602 | 0.471 | 0.352 |
| Random 80:20 holdout | 76 | **0.740** | 0.442 | 0.313 |
| Scaffold-grouped five-fold | 378 | **0.513** | 0.541 | 0.407 |
| Independent external set | 41 | **−0.385** | — | 0.583 |

Two reference predictors are evaluated under both internal designs. The
training-set mean reaches R² ≈ −0.03 throughout. A 1-NN predictor on
descriptor distance reaches R² = 0.611 on the random holdout but −0.020 under
the scaffold-grouped design — so the model's margin over a nearest-neighbour
lookup is real only once analogues of a scaffold cannot be split across
partitions.

**Use 0.513, not 0.740, as the expected accuracy for an unseen scaffold.**

## Quick start

```bash
git clone https://github.com/[AUTHORS]/tbars-mda-qsar.git
cd tbars-mda-qsar
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Predict with the fitted model:

```python
import joblib, pandas as pd

b = joblib.load("models/gradient_boosting_fitted.joblib")
X = pd.read_csv("data/rdkit_descriptors_378.csv")[b["descriptor_columns"]]
pred = b["pipeline"].predict(X)          # predicted pIC50
```

For new molecules, compute the same 217 descriptors with the same RDKit
version — `descriptor_block()` in
[`src/15_rdkit_descriptor_study.py`](src/15_rdkit_descriptor_study.py) is the
function used here — then subset to `b["descriptor_columns"]` in that order.
Median imputation and zero-variance removal are inside the pipeline, so pass
raw descriptor values and leave missing ones as `NaN`.

Check the applicability domain before trusting a prediction: see
[Applicability domain](#applicability-domain).

## Repository layout

```
├── data/
│   ├── raw_records_4470.csv            retrieved records, before curation
│   ├── curated_structures_378.csv      the modelling set, one row per structure
│   ├── curated_measurements_443.csv    the measurements they were aggregated from
│   ├── rdkit_descriptors_378.csv       217 descriptors, in split_manifest row order
│   ├── split_manifest.csv              train/holdout assignment and scaffold
│   └── ldl_literature_candidates.csv   human-LDL compounds from the literature
├── models/
│   └── gradient_boosting_fitted.joblib the model behind every reported prediction
├── results/
│   ├── figures/        Figures 1-4
│   ├── tables/         Tables 1-4
│   ├── predictions/    every observed-vs-predicted table
│   ├── shap/           SHAP attributions for the holdout
│   └── reports/        machine-readable summaries of every reported number
├── supplementary/      Tables S1-S5, Figures S1-S3
├── src/                the scripts, numbered in execution order
├── run_all.sh          reproduce everything, then compare against results/
├── requirements.txt    pinned versions
├── CITATION.cff
├── LICENSE             MIT, for src/
└── LICENSE-DATA        CC BY 4.0, for data/, results/, supplementary/
```

### Row order is the contract

`curated_structures_378.csv`, `rdkit_descriptors_378.csv` and
`split_manifest.csv` share one row order. Feature order is
`descriptor_columns` inside the joblib bundle. Twelve of the 217 descriptor
columns contain missing values; they are imputed by the column median *inside*
the pipeline, so imputation is fitted on each fold's fitting portion only and
never sees the evaluation rows.

## The model

`models/gradient_boosting_fitted.joblib` is a dict:

| Key | Contents |
|---|---|
| `pipeline` | `Pipeline`: median imputation → zero-variance removal → `GradientBoostingRegressor` |
| `descriptor_columns` | the 217 descriptor names, in the order the pipeline expects |
| `training_struct_keys` | the 302 structures it was fitted on |
| `applicability_cutoff` | 11.6672 |
| `seed` | 42 |

Gradient Boosting was selected by the highest mean five-fold **training**
cross-validation R² among Random Forest, XGBoost, Gradient Boosting and Extra
Trees — the holdout played no part in the choice. Hyperparameters: 800
estimators, learning rate 0.075, maximum depth 3, subsample 0.7, minimum leaf
size 5. All four families, their search spaces and the selected values are in
[`supplementary/TableS1_hyperparameter_search.csv`](supplementary/TableS1_hyperparameter_search.csv);
the three non-selected families' parity plots are Figures S1–S3.

### Applicability domain

The cutoff is the 95th percentile of out-of-fold nearest-neighbour distances
in standardized descriptor space, set on training folds only: **11.6672**. Of
the 76 holdout structures, 73 fall inside (RMSE 0.395) and 3 outside
(RMSE 1.070) — a 2.7-fold difference.
[`results/predictions/applicability_domain_holdout.csv`](results/predictions/applicability_domain_holdout.csv)
carries the per-structure distance, the inside/outside flag and the error.

### What drives the predictions

SHAP attributions are dominated by the VSA descriptor families
(`EState_VSA`, `VSA_EState`, `SMR_VSA`, `PEOE_VSA`, `SlogP_VSA`), which
distribute surface area across electronic and lipophilicity bins. They are not
reducible to a single interpretable mechanism, and we do not claim one.
Per-structure values are in [`results/shap/`](results/shap).

## Known limits

- **The endpoint pools protocols.** Of the 443 measurements, the oxidation
  initiator is unreported in 54.4% and the reaction time in 48.5%; no record
  states how MDA was separated before quantification. Among the 32 structures
  with more than one measurement, 28 pool different conditions, with
  within-structure activity spreads up to 142.9-fold. Tables S2–S4 give the
  full audit.
- **External performance is much worse than internal.** On 41 structures from
  five publications that the dataset's text rules had missed, R² = −0.385 and
  r = 0.16 (Table S5). The predicted spread is 33% of the observed spread;
  across four evaluation panels the bias correlates with
  (observed − training mean) at r = −0.994 with slope −0.998. This is
  **regression to the mean**, not a bias specific to any one assay matrix.
- **The human-LDL comparison is illustrative only.** The three compounds in
  Table 4 were absent from the training data, were not measured here, and were
  selected after their errors were known.
- **Predictions are pIC₅₀ for a mixture of assays**, not for one standardized
  reaction system, and TBARS is an indirect measurement.

## Reproducing the results

```bash
./run_all.sh                 # writes to rerun/, then compares with results/
./run_all.sh --check         # compare an existing rerun/ against results/
AUDIT_NJOBS=8 ./run_all.sh   # more cores for the tree ensembles; same numbers
```

Results go to `rerun/` so the published `results/` tree is never overwritten,
and the script ends by comparing the key metrics. The full run takes roughly
30–60 minutes on one core, dominated by model selection.

| Script | Produces |
|---|---|
| `01_build_dataset.py` | the curated data from the retrieved records |
| `11_protocol_audit.py` | Tables S2–S4 |
| `15_rdkit_descriptor_study.py` | model selection, both designs, the applicability domain, SHAP, `rdkit_study_report.json` |
| `16_rdkit_figures.py` | Figures 1–4 |
| `21_external_missed_set.py` | the external test set and Table S5 |
| `20_ldl_candidates.py` | the human-LDL predictions (Table 4) |
| `26_build_deposit.py` | Table S1, Figures S1–S3, the serialized model |
| `27_assemble_deposit.py` | the archive tree and its checksum manifest |

`09_generalization_audit.py` is imported by `15_` for the scaffold key and the
split helpers; it is not run on its own.

Tables 1–3 are generated from `results/reports/rdkit_study_report.json` by
`27_assemble_deposit.py` rather than written by hand, so they cannot drift
from the reported run.

### Two things that will break reproduction

1. **Package versions.** The tree ensembles and the RDKit descriptor set both
   change between releases. Use `requirements.txt` as pinned.
2. **Split ordering.** The partition is
   `train_test_split(..., test_size=0.2, random_state=42, shuffle=True)`, and
   **both index arrays are sorted afterwards** — the inner cross-validation
   folds depend on that ordering, so an unsorted reimplementation silently
   produces different numbers. `data/split_manifest.csv` records the partition
   directly, which is the reliable way to reuse it.

The seed is 42 throughout.

## Citing

Please cite the article, this repository, and the original measurement
publications — the DOI of every source is in
`data/curated_structures_378.csv` and `data/curated_measurements_443.csv`.
See [`CITATION.cff`](CITATION.cff).

<!-- [AUTHORS: before making this public —
     1. fill in the author list and ORCIDs in CITATION.cff
     2. set the copyright holder in LICENSE and confirm LICENSE-DATA
     3. replace every [AUTHORS] in the URLs above with the GitHub account
     4. add the article DOI here and in CITATION.cff once accepted
     5. add the Zenodo DOI badge after archiving a release ] -->
