# Syracuse count-forecasting experiment

## Input → target

Selected target: **Next-week larceny records**. Inputs: prior completed-period counts, recent averages/variability, and known seasonal calendar features. The target is a source-record count, not a unique-incident count or individual risk.

## Target screening — 2022 only

The target and model were fixed before evaluating 2023. All three targets and two Poisson-loss models were declared before screening. Models refit monthly on expanding history; predictions use only prior completed periods. Partial boundary weeks are excluded.

| Target | Best learned model | Model MAE | Best baseline MAE | MAE reduction |
|---|---|---:|---:|---:|
| Next-day total records | poisson | 5.79 | 5.84 | 0.7% |
| Next-week total records | poisson | 19.96 | 20.16 | 1.0% |
| Next-week larceny records | poisson | 9.74 | 10.13 | 3.9% |

MAE means mean absolute error in records per forecast period. Target selection uses relative reduction against each target’s best baseline, not raw errors across different count scales. The baselines are the previous period, a recent mean (28 days or four weeks), and seasonal naïve (7 days or 52 weeks).

## Fixed-choice evaluation — 2023

**Next-week larceny records**, using **poisson**, across **52** periods.

| Method | MAE (records) | RMSE (records) |
|---|---:|---:|
| last_period | 14.38 | 18.26 |
| recent_mean | 12.01 | 16.68 |
| seasonal_naive | 21.33 | 26.52 |
| poisson | 12.91 | 16.83 |

MAE reduction versus the validation-selected **recent_mean**: **-7.4%**. The validation gain did not generalize: the learned model did not beat the baseline.

This experiment supports count forecasting as a more interpretable task, but does not establish useful learned predictive skill. Do not select another target or tune hyperparameters using these test results and continue calling 2023 untouched.

## Source coverage check

The last observed selected-category date in 2023 is **2023-12-26**. Forecast periods ending after that date: **2023-12-25**. Where flagged, treat the tail as a collection-coverage concern, not evidence of a true crime decline. Scores retain the predefined calendar window; no favorable post-result exclusion was made. This further limits the benchmark’s interpretation.

![Observed and predicted counts](forecast.png)

## Evaluation meaning

- Each prediction is one period ahead. Earlier 2023 outcomes become observed history for later 2023 forecasts and monthly refits. This is not a year-ahead forecast.
- The saved model is the initial fit using only complete periods before 2023. It is not the series of monthly refits used for the rolling evaluation.
- Count-based features are shifted before rolling averages. Future target values cannot change earlier predictions or the validation-selected specification.
- This is a retrospective DATEEND snapshot. Reporting delays and collection gaps cannot be reconstructed; calendar gaps are treated as zero records under an explicit coverage assumption, not proof of zero crime.
- Original category classification remains available as a historical experiment. Its accuracy/F1 metrics are not comparable to count-forecast MAE.
- No calibrated uncertainty intervals, current-data forecast, or deployment are claimed.

Source provenance, immutable input hash, complete validation scores, and per-period predictions are in `metrics.json`. The frozen specification is in `selection.json`.
