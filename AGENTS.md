# Syracuse temporal model maintenance

Repository: `ichen27/Temporal-Crime-Prediction-Classification-Model-w-Random-Forest-`. Read README.md for setup and validation commands; ARTIFACTS.md indexes deliverables.

- `evaluation.py` owns the feature contract, chronological partitions, selection, refit, and evaluation. The shared analysis dependency owns schema/date parsing; pin it to an inspected immutable commit.
- Keep test labels out of selection, tuning, and preprocessing fitting. Keep dates together across partitions. Preserve the test-label independence regression test.
- Describe predictions as conditional category scores, with calibration and historical-data limits. Preserve unseen evaluation categories in metrics.
- Save versioned aggregate reports and the modest fitted artifact in `reports/`; use ignored `work/` for input downloads and scratch experiments. Preserve `legacy/` as historical work.
- Work on a feature branch. Run lint, tests, and the full benchmark; review the diff and open a PR. Merge only with current-revision CI passing and within authorized scope.
