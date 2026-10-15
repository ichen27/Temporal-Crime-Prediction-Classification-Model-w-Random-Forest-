# Artifact index

Version: model package 0.2.0. Source hash, runtime versions, and fitted dates are in the model card.

| Artifact | Purpose and use |
|---|---|
| [benchmark.md](reports/benchmark.md) | Readable selection and held-out results, with limitations. |
| [metrics.json](reports/metrics.json) | Full machine-readable audit, per-class results and confusion matrices. |
| [model-card.json](reports/model-card.json) | Feature contract, fitted dates, seed, versions, and source SHA-256. |
| [model.joblib](reports/model.joblib) | Fitted 2020–2022 classifier; load using the locked environment and `syracuse-model predict`. Only load trusted joblib files. |
| [example-prediction.json](reports/example-prediction.json) | Sample date-conditioned category scores for 2023-07-15. |
| [src/syracuse_model](src/syracuse_model/) | Editable acquisition, evaluation, and prediction source. |
| [legacy](legacy/) | Original scripts and README, retained for historical context. |

The small fitted model is committed so users can inspect it without retraining. Training reproduces the reports/model using the README commands. Raw data is not duplicated here: the downloader retrieves the existing companion snapshot at a pinned revision and verifies its hash. Scratch runs, environments, caches, and credentials are excluded.
