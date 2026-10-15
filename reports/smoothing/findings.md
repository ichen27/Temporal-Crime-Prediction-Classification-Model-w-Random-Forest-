# Weekly larceny: exponential smoothing

**Exploratory follow-up:** 2023 was already examined in the earlier experiment. These results are not a new untouched test.

| Evaluation window | Method | MAE (records/week) | RMSE | Weeks |
|---|---|---:|---:|---:|
| 2022 selection | Four-week average | 10.132 | 12.576 | 51 |
| 2022 selection | Simple exponential smoothing | 9.865 | 12.233 | 51 |
| 2022 selection | Damped Holt | 9.646 | 12.034 | 51 |
| 2023 full window | Four-week average | 12.014 | 16.682 | 52 |
| 2023 full window | Damped Holt | 12.398 | 17.260 | 52 |
| 2023 through Dec 24 | Four-week average | 10.863 | 13.623 | 51 |
| 2023 through Dec 24 | Damped Holt | 11.273 | 14.433 | 51 |

Chosen using 2022 MAE: **Damped Holt**.

Full 2023 window: smoothing MAE is **3.2% higher** than the four-week average.

Excluding the previously flagged final week: smoothing MAE is **3.8% higher** than the four-week average.

## Protocol

- Fixed weekly larceny target; no new target search. 208 complete Monday–Sunday weeks in the source window, not 208 independent training cases at every origin.
- Fit all completed history at the first forecast origin each month. Optimize initial states and smoothing parameters by training squared error; choose the method by rolling 2022 MAE. Update state after each newly observed count. L-BFGS-B fits retry SLSQP only on convergence failure; each solver is recorded.
- Predict one week ahead before seeing that week. Negative extrapolations are clipped to zero. No seasonal component or post-2023 tuning.
- Both full-window and final-week-excluded scores were specified before computing this experiment. The sensitivity uses the same predictions for both methods and does not prove earlier coverage is complete.
- The JSON artifact is the initial fit through January 1, 2023. It forecasts January 2–8; optional observations advance states without refitting parameters. The evaluation refits monthly and records every fit separately.

## Coverage and interpretation

The local final week contains four larceny records, ending December 26. An independent read of the city’s currently served 2023 Part 1 data agrees for December; it also stops on December 26. This locates the concern upstream but does not establish whether later dates were missing or truly had no records. See the [source audit](source-audit.json). The database was not replaced.

The city’s [2023 metadata](https://www.arcgis.com/sharing/rest/content/items/94bc33f1a2c646b995d1ab356edf0730/info/metadata/metadata.xml?format=default&output=html) defines DATEEND as the reported date. This corrects the earlier incident-end-date description. It does not provide historical ingestion or revision timestamps. Counts are retrospective source records, not verified unique incidents. Zero-filled gaps remain a coverage assumption.

Small differences here do not establish a reliable forecasting advantage. Resolve source coverage and obtain a genuinely untouched evaluation period before selecting a deployment model. No calibrated intervals are supplied.

## Reproduce

```bash
syracuse-smooth experiment --database work/crime_data.db --output reports/smoothing
syracuse-smooth predict --model reports/smoothing/model.json
```

The source audit is a separately retrieved, dated external snapshot; rerunning the experiment does not silently refresh it. Editable source: `src/syracuse_model/smoothing.py` and `smoothing_report.py`.

Methods: [Forecasting: Principles and Practice](https://otexts.com/fpp3/exponential-smoothing.html); [statsmodels exponential smoothing](https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html).
