# Artifact index

Source hash, runtime versions, and fitted dates are recorded in each experiment’s metadata. All linked artifacts are committed; environments, downloaded duplicate inputs, caches, and scratch runs are excluded.

## Count-forecasting experiment

| Artifact | Purpose and use |
|---|---|
| [findings.md](reports/forecasting/findings.md) | Target comparison and honest held-out result. |
| [selection.json](reports/forecasting/selection.json) | 2022-only screening scores and the specification frozen before testing. |
| [metrics.json](reports/forecasting/metrics.json) | Data audit, complete validation/test predictions and aggregate errors. |
| [forecast.png](reports/forecasting/forecast.png) | Finished observed/model/baseline chart; open in any image viewer. |
| [model-card.json](reports/forecasting/model-card.json) | Feature contract, source hash, fitted cutoff, versions and limitations. |
| [forecast.joblib](reports/forecasting/forecast.joblib) | Pre-2023 fitted count regressor. Use the locked environment and `syracuse-forecast predict`; load only trusted artifacts. |
| [example-history.json](reports/forecasting/example-history.json) | Thirteen completed weeks of larceny counts, an editable input fixture. |
| [example-forecast.json](reports/forecasting/example-forecast.json) | Forecast of January 2–8, 2023 from the fixture and saved model. |
| [forecasting.py](src/syracuse_model/forecasting.py) | Editable target aggregation, features, rolling evaluation and inference. |
| [forecast_report.py](src/syracuse_model/forecast_report.py) | Editable findings and figure generator. |

The README experiment command regenerates the model, metrics, findings and chart. The example input is a committed aggregate fixture; run the README prediction command and redirect stdout to regenerate its output. The downloader retrieves the existing companion database at an immutable revision and checks its SHA-256.

## Preserved category-classification experiment

| Artifact | Purpose and use |
|---|---|
| [experiment guide](docs/classification-experiment.md) | Historical task description and reproduction commands. |
| [benchmark.md](reports/benchmark.md) | Original chronological category benchmark. |
| [metrics.json](reports/metrics.json) | Full audit, per-class results and confusion matrices. |
| [model-card.json](reports/model-card.json) | Classifier features, dates, seed, versions and source hash. |
| [model.joblib](reports/model.joblib) | Fitted 2020–2022 classifier; use `syracuse-model predict` with the locked environment. |
| [example-prediction.json](reports/example-prediction.json) | Historical category scores for 2023-07-15. |
| [source](src/syracuse_model/evaluation.py) | Editable original chronological evaluation. |
| [legacy](legacy/) | Original uploaded scripts and README. |
