# Syracuse forecasting maintenance

Repository: `ichen27/Temporal-Crime-Prediction-Classification-Model-w-Random-Forest-`. README.md contains setup/validation; ARTIFACTS.md indexes deliverables.

- `forecasting.py` owns target aggregation, shifted features, chronological screening, monthly refitting and inference. `forecast_report.py` renders findings/figures. The pinned analysis package owns schema/date parsing.
- Choose target/model specifications on validation only. The current count experiment has already evaluated 2023; treat later changes tested there as exploratory or obtain a fresh holdout. Preserve the tests that mutate future data without changing past features, selection or predictions.
- Daily targets cover one day; weekly targets cover complete Monday–Sunday periods. Training labels must end before each forecast origin. Source-record availability and completeness remain unverified.
- Report counts as source records. State whether forecasts roll forward with newly observed outcomes. Preserve negative results and compare to baselines in matching units.
- `smoothing.py` owns the exploratory simple/damped-Holt comparison, JSON state inference, and monthly fit audit. Preserve its predeclared final-week sensitivity and convergence-only solver fallback. `audit_tail.py` fetches dated public aggregates separately; do not refresh source inputs implicitly.
- `evaluation.py` and `cli.py` preserve the earlier category classifier. Keep its evidence separate from count forecasting; accuracy and MAE are not comparable.
- Commit aggregate reports and modest fitted artifacts under `reports/`; use ignored `work/` for downloads and scratch runs. Preserve `legacy/`.
- Use a feature branch. Run lint, tests, full experiments and serialized prediction checks; review the diff and update the PR. Merge only with current-revision CI passing and within authorized scope.
