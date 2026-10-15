# Temporal classification benchmark

Training: 2020–2021. Model selection: 2022 macro F1. Final refit: 2020–2022. Untouched test: 2023. Partial 2024 is excluded. Records sharing a date stay together.

| Candidate | Validation macro F1 | Accuracy | Log loss |
|---|---:|---:|---:|
| prior_baseline | 0.0123 | 0.2162 | 2.2797 |
| logistic | 0.0201 | 0.2229 | 2.2786 |
| forest_regularized | 0.0209 | 0.2233 | 2.2928 |
| forest_balanced | 0.0038 | 0.0040 | 3.1946 |

Selected **forest_regularized** using validation results only.

| 2023 test | Macro F1 | Accuracy | Log loss |
|---|---:|---:|---:|
| selected_model | 0.0216 | 0.2200 | 2.2546 |
| prior_baseline | 0.0101 | 0.1725 | 2.2542 |

Test macro F1 difference versus the prior baseline: **+0.0115**.

## Interpretation and limits

- This predicts a category conditional on a source record, not whether a crime will occur.
- Scores are uncalibrated; balanced class weights can distort category probabilities.
- Date alone contains limited information. Rare-class performance and the confusion matrix are included in `metrics.json`; accuracy alone is inadequate.
- One validation year and one test year do not establish robustness across future years.
- Source records are not deduplicated unique incidents; reporting and coverage may change.
- Splits use DATEEND, not availability timestamps. This retrospective snapshot cannot establish what was known at each historical training cutoff.
- Dates preserve the source calendar day. Hour, address, person, and location are not used.
- No performance claim is made against the old random-split notebook on different data.
- The saved model retains the evaluated 2020–2022 fit; it is not a current operational model.

See `model-card.json` for versions and the input database SHA-256.
