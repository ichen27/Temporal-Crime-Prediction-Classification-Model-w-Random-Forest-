# Syracuse Temporal Crime Classification

A reproducible experiment asking: **how much does the calendar tell us about the category of a recorded crime?** It uses the historical snapshot from [Syracuse Crime Data Analysis](https://github.com/ichen27/Syracuse-Crime-Data-Analysis), with a shared audited loader and chronological evaluation.

**Result:** the selected regularized random forest achieved **22.00% accuracy** and **0.0216 macro F1** on 2023, versus **17.25% accuracy** and **0.0101 macro F1** for a training-prior baseline. Probability log loss was essentially unchanged (2.2546 versus 2.2542; lower is better). Calendar features alone remain weak, especially for rare categories. These are held-out results, not a comparison against the old random-split implementation.

This model estimates a **category conditional on a source record**, not the probability that a crime will occur. It does not predict individuals or locations, and the output is not an operational public-safety forecast.

Read the [benchmark](../reports/benchmark.md), [full metrics](../reports/metrics.json), [model metadata](../reports/model-card.json), and [artifact index](../ARTIFACTS.md).

## Quickstart

Requires Git and Python 3.12 (recommended). The dependency lock installs the shared analysis package at an immutable Git commit.

```bash
git clone --branch feat/temporal-model-evaluation https://github.com/ichen27/Temporal-Crime-Prediction-Classification-Model-w-Random-Forest-.git
cd Temporal-Crime-Prediction-Classification-Model-w-Random-Forest-
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m syracuse_model.download --output work/crime_data.db
syracuse-model train --database work/crime_data.db --output reports
syracuse-model predict --model reports/model.joblib --date 2023-07-15
```

On Windows activate with `.venv\Scripts\activate`. The downloader retrieves the original repository database from a pinned revision, verifies its SHA-256, and refuses to overwrite different existing content. Training leaves it unchanged. Outputs include `model.joblib`, `model-card.json`, `metrics.json`, and `benchmark.md`. To use an existing compatible snapshot, supply its path directly.

You can skip downloading/training and inspect the included model using the final prediction command. Load joblib files only from trusted sources; they can execute code. Use the locked dependency versions for the provided artifact. A [sample prediction](../reports/example-prediction.json) shows the output without requiring Python.

## Evaluation design

| Stage | Calendar dates | Purpose |
|---|---|---|
| Training | 2020–2021 | Fit each candidate |
| Validation | 2022 | Select by macro F1, with stable candidate-order tie breaking |
| Refit | 2020–2022 | Fit the selected specification and baseline |
| Test | 2023 | Evaluate once after selection |
| Excluded | Partial 2024 | Outside this experiment |

All rows sharing a date stay in the same partition. Candidates are a prior baseline, regularized logistic regression, a depth/leaf-limited random forest, and its class-balanced variant. Features include month, weekday, day of month, and their cyclic encodings. Year is omitted because an absolute year feature does not give a forest a reliable extrapolation rule. The model keeps string labels directly, avoiding separate label-encoder files.

`metrics.json` includes accuracy, macro/weighted F1, log loss, per-class precision/recall/support, confusion matrices, unseen-category counts, data reconciliation, and partition sizes. Evaluation retains unseen test labels as errors instead of silently dropping them. Macro F1 uses the union of training and evaluation categories; label sets can differ between validation and test, so compare candidates within each split.

Selection prioritizes macro F1, not probability calibration. Class balancing did not help on validation. The saved artifact retains the 2020–2022 fit used in the test; it is not retrained on the held-out year.

## Data quality and limitations

The shared loader fixes the 2022 Part 2 column-name mismatch (`CODE DEFINED`) and parses its US-style dates. It audits 55,851 rows from the 2020–2024 source tables and retains 55,779. The 2019 source table is excluded. It preserves the calendar day written in `DATEEND`, rather than shifting mixed midnight conventions across dates. See the companion analysis for provenance details and source terms.

Records are not verified unique incidents. Later source tables can contain older DATEEND values. The city’s 2023 metadata defines `DATEEND` as the date reported; this corrects the earlier incident-end-date assumption. Historical publication/revision timestamps remain unavailable, so this retrospective snapshot cannot establish what was actually available at each historical training cutoff. Reporting practices, coverage, and category definitions may change. A single validation/test pair does not establish future robustness. Scores are uncalibrated, and date-only features provide limited information. Dates outside the fitted interval are explicitly flagged by the prediction command.

The [original scripts](../legacy/) remain available for historical context. Their random split and original preprocessing are outside the maintained workflow. No new live-data acquisition or operational deployment is included.

## Development and contributions

```bash
ruff check src tests
pytest -q
python -m syracuse_model.download
syracuse-model train --database work/crime_data.db --output work/benchmark
```

CI runs lint, unit/integration tests, acquisition verification, a full benchmark, and a prediction smoke test. Tests cover date-group separation, unseen labels, invalid dates, artifact round trips, download integrity, and the independence of model selection from test labels. Submit changes on a feature branch with a PR and evidence. For issues, include reproduction commands and package versions; use synthetic records when sharing examples.

Metric definitions and the prior baseline follow [scikit-learn's evaluation documentation](https://scikit-learn.org/stable/modules/model_evaluation.html) and [dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators).

## Attribution and licensing

Original project: Ivan Chen. Third-party packages retain their licenses. The original repositories specify no code/dataset license; this update does not grant new redistribution rights. Confirm source terms and obtain a license decision before redistribution or a licensed release.
