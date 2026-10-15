"""Compare aggregate count targets using past-only, rolling forecasts.

The dataset is a retrospective DATEEND snapshot. Missing record-availability
history prevents claiming a deployment-realistic backtest.
"""

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from syracuse_crime import load_records
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class Target:
    name: str
    days: int
    category: str | None = None


TARGETS = (
    Target('daily_total', 1),
    Target('weekly_total', 7),
    Target('weekly_larceny', 7, 'LARCENY'),
)
BASELINES = ('last_period', 'recent_mean', 'seasonal_naive')


def count_series(records, target, start='2020-01-01', end='2023-12-31'):
    """Count source records; fill calendar days under an explicit coverage assumption.

    Weekly periods start Monday. Partial boundary weeks are discarded, including
    an incomplete final week supplied by a custom coverage end.
    """
    if target.days not in (1, 7):
        raise ValueError('Only daily and Monday–Sunday weekly targets are supported')
    dates = pd.date_range(start, end, freq='D')
    selected = records if target.category is None else records.loc[
        records.category.eq(target.category)
    ]
    daily = selected.groupby('date').size().reindex(dates, fill_value=0).astype(float)
    if target.days == 1:
        series = daily
    else:
        grouped = daily.resample('W-MON', closed='left', label='left')
        series = grouped.sum().loc[grouped.count().eq(7)]
    series.index.name = 'period_start'
    return series.rename('count')


def design(series, target):
    """Row t predicts t; every count feature comes from a completed prior period."""
    lags = (1, 2, 7, 14, 28) if target.days == 1 else (1, 2, 4, 8, 13)
    windows = (7, 28) if target.days == 1 else (4, 13)
    x = pd.DataFrame(index=series.index)
    for lag in lags:
        x[f'lag_{lag}'] = series.shift(lag)
    past = series.shift(1)
    for window in windows:
        x[f'mean_{window}'] = past.rolling(window).mean()
        x[f'std_{window}'] = past.rolling(window).std(ddof=0)
    # Known calendar covariates at each forecast origin; no future weather/counts.
    angle = 2 * np.pi * (series.index.dayofyear - 1) / 365.25
    x['annual_sin'], x['annual_cos'] = np.sin(angle), np.cos(angle)
    if target.days == 1:
        angle = 2 * np.pi * series.index.dayofweek / 7
        x['weekday_sin'], x['weekday_cos'] = np.sin(angle), np.cos(angle)
    return x


def model_candidates():
    return {
        'poisson': make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=1000)),
        'boosted_poisson': HistGradientBoostingRegressor(
            loss='poisson', max_iter=100, max_leaf_nodes=7, min_samples_leaf=20,
            l2_regularization=10.0, early_stopping=False, random_state=42,
        ),
    }


def baseline_predictions(series, target):
    return pd.DataFrame({
        'last_period': series.shift(1),
        'recent_mean': series.shift(1).rolling(28 if target.days == 1 else 4).mean(),
        'seasonal_naive': series.shift(7 if target.days == 1 else 52),
    })


def rolling_predictions(series, target, start, end, estimators):
    """One-period-ahead predictions; refit at the first scored origin each month.

    At later origins, completed periods become history. This is sequential
    forecasting, never a whole-year forecast made at the start of January.
    """
    x = design(series, target)
    period_end = series.index + pd.Timedelta(target.days - 1, unit='D')
    mask = (series.index >= pd.Timestamp(start)) & (period_end <= pd.Timestamp(end))
    origins = series.index[mask]
    if len(origins) == 0:
        raise ValueError('No complete evaluation periods')
    baseline = baseline_predictions(series, target).loc[origins]
    if baseline.isna().any().any() or x.loc[origins].isna().any().any():
        raise ValueError('Insufficient prior history for evaluation')
    output = baseline.copy()
    output['actual'] = series.loc[origins]
    fitted, fitted_month = {}, None
    for origin in origins:
        month = origin.to_period('M')
        if month != fitted_month:
            # For aligned daily/weekly periods, every earlier period is complete.
            train_index = x.index[(period_end < origin) & x.notna().all(axis=1)]
            if len(train_index) < 20:
                raise ValueError('Need at least 20 complete training examples')
            fitted = {
                name: clone(estimator).fit(x.loc[train_index], series.loc[train_index])
                for name, estimator in estimators.items()
            }
            fitted_month = month
        for name, model in fitted.items():
            prediction = float(model.predict(x.loc[[origin]])[0])
            if not np.isfinite(prediction):
                raise ValueError(f'{name} produced a nonfinite count')
            output.loc[origin, name] = max(0.0, prediction)
    return output


def scores(predictions):
    y = predictions.actual.to_numpy()
    return {name: {
        'mae': float(mean_absolute_error(y, predictions[name])),
        'rmse': float(np.sqrt(mean_squared_error(y, predictions[name]))),
        'mean_actual': float(np.mean(y)), 'periods': len(y),
    } for name in predictions if name != 'actual'}


def select_target(records, estimators=None):
    """Screen three predeclared targets on 2022 only; freeze before testing 2023."""
    estimators = model_candidates() if estimators is None else estimators
    results, predictions = {}, {}
    for target in TARGETS:
        series = count_series(records, target)
        pred = rolling_predictions(series, target, '2022-01-01', '2022-12-31', estimators)
        metrics = scores(pred)
        best_baseline = min(BASELINES, key=lambda name: metrics[name]['mae'])
        best_model = min(estimators, key=lambda name: metrics[name]['mae'])
        baseline_error = metrics[best_baseline]['mae']
        skill = 1 - metrics[best_model]['mae'] / baseline_error if baseline_error > 0 else 0.0
        results[target.name] = {
            'target': asdict(target), 'scores': metrics,
            'best_baseline': best_baseline, 'best_model': best_model,
            'mae_skill_vs_best_baseline': skill,
        }
        predictions[target.name] = pred
    winner = max(results, key=lambda name: results[name]['mae_skill_vs_best_baseline'])
    return results, winner, predictions


def prediction_rows(frame):
    return [dict(period_start=str(index.date()), **{k: float(v) for k, v in row.items()})
            for index, row in frame.iterrows()]


def run_experiment(database, output):
    records, audit = load_records(database)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        validation, winner, validation_predictions = select_target(records)
        chosen = validation[winner]
        selection = {
            'target': chosen['target'], 'model': chosen['best_model'],
            'baseline': chosen['best_baseline'],
            'criterion': 'Largest 2022 relative MAE reduction vs that target\'s best baseline',
            'screening_window': ['2022-01-01', '2022-12-31'],
            'test_window': ['2023-01-01', '2023-12-31'],
            'validation': validation,
        }
        # Persist the choice before calculating any 2023 forecast error.
        (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
        target = Target(**selection['target'])
        series = count_series(records, target)
        estimator = model_candidates()[selection['model']]
        test_predictions = rolling_predictions(
            series, target, '2023-01-01', '2023-12-31', {selection['model']: estimator}
        )
        test_scores = scores(test_predictions)
        target_records = records if target.category is None else records.loc[
            records.category.eq(target.category)
        ]
        test_dates = target_records.loc[target_records.date.between('2023-01-01', '2023-12-31'), 'date']
        last_observed = test_dates.max()
        tail_periods = test_predictions.index[
            test_predictions.index + pd.Timedelta(target.days - 1, unit='D') > last_observed
        ] if pd.notna(last_observed) else test_predictions.index
        coverage = {
            'last_observed_target_date_in_test': str(last_observed.date()) if pd.notna(last_observed) else None,
            'periods_ending_after_last_observed_record': [str(day.date()) for day in tail_periods],
            'policy': 'Flags are diagnostic only. Test scores retain the predefined calendar window; no post-result filtering.',
            'interpretation': 'Calendar-complete periods do not establish complete source coverage. No records after the final observed date may mean missing collection or zero events.',
        }
        baseline_mae = test_scores[selection['baseline']]['mae']
        test_skill = 1 - test_scores[selection['model']]['mae'] / baseline_mae if baseline_mae else 0.0
        # An inspectable fit at the initial 2023 origin; no held-out targets in artifact.
        x = design(series, target)
        cutoff = pd.Timestamp('2023-01-01')
        train_mask = (x.index + pd.Timedelta(target.days - 1, unit='D') < cutoff) & x.notna().all(axis=1)
        model = clone(estimator).fit(x.loc[train_mask], series.loc[train_mask])
    metadata = {
        'target': asdict(target), 'model': selection['model'],
        'training_through': str((x.index[train_mask][-1] + pd.Timedelta(target.days - 1, unit='D')).date()),
        'feature_columns': list(x.columns),
        'data_sha256': audit['database_sha256'],
        'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
        'interpretation': 'Expected source-record count for the next complete period, not individual risk.',
        'evaluation': 'One-period forecasts with observed history and monthly refitting; not a year-ahead forecast.',
        'test_mae_skill_vs_selected_baseline': test_skill,
        'availability': 'Retrospective DATEEND snapshot; historical report availability is unknown.',
        'test_coverage': coverage,
    }
    report = {
        'selection': selection, 'data_audit': audit, 'metadata': metadata,
        'test': test_scores, 'test_mae_skill_vs_selected_baseline': test_skill,
        'validation_predictions': {name: prediction_rows(pred) for name, pred in validation_predictions.items()},
        'test_predictions': prediction_rows(test_predictions),
    }
    joblib.dump({'model': model, 'metadata': metadata}, output / 'forecast.joblib', compress=3)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'model-card.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return report


def forecast_from_history(bundle, history, period_start):
    """Predict one next period from explicitly supplied, consecutive past counts."""
    from datetime import date

    start = pd.Timestamp(date.fromisoformat(period_start))
    if start.strftime('%Y-%m-%d') != period_start:
        raise ValueError('period_start must use YYYY-MM-DD')
    metadata = bundle['metadata']
    target = Target(**metadata['target'])
    if start <= pd.Timestamp(metadata['training_through']):
        raise ValueError('Forecast must be after the saved model training window')
    if not isinstance(history, list) or not history:
        raise ValueError('history must be a nonempty list of period_start/count objects')
    dates, counts = [], []
    for item in history:
        if not isinstance(item, dict) or set(item) != {'period_start', 'count'}:
            raise ValueError('Each history item needs exactly period_start and count')
        value = item['count']
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('Counts must be nonnegative finite integers')
        if not np.isfinite(value) or value < 0 or int(value) != value:
            raise ValueError('Counts must be nonnegative finite integers')
        day = date.fromisoformat(item['period_start'])
        if day.isoformat() != item['period_start']:
            raise ValueError('History dates must use YYYY-MM-DD')
        dates.append(pd.Timestamp(day))
        counts.append(float(value))
    index = pd.DatetimeIndex(dates)
    expected = pd.date_range(end=start - pd.Timedelta(target.days, unit='D'),
                             periods=len(index), freq=f'{target.days}D')
    if not index.equals(expected):
        raise ValueError('History must be sorted, consecutive, and end immediately before the forecast')
    if target.days == 7 and start.dayofweek != 0:
        raise ValueError('Weekly periods must start Monday')
    required = 28 if target.days == 1 else 13
    if len(history) < required:
        raise ValueError(f'At least {required} completed periods are required')
    series = pd.Series(counts, index=index)
    series.loc[start] = np.nan
    row = design(series, target).loc[[start]]
    if list(row.columns) != metadata['feature_columns']:
        raise ValueError('Saved model feature contract does not match this software')
    with threadpool_limits(limits=1):
        prediction = max(0.0, float(bundle['model'].predict(row)[0]))
    if not np.isfinite(prediction):
        raise ValueError('Model produced a nonfinite count')
    recent_window = 28 if target.days == 1 else 4
    return {
        'target': asdict(target), 'period_start': period_start,
        'period_end': str((start + pd.Timedelta(target.days - 1, unit='D')).date()),
        'model_expected_count': prediction,
        'recent_mean_baseline': float(np.mean(counts[-recent_window:])),
        'model_training_through': metadata['training_through'],
        'interpretation': metadata['interpretation'],
        'test_mae_skill_vs_selected_baseline': metadata.get('test_mae_skill_vs_selected_baseline'),
        'limitation': 'Historical point estimate only; no calibrated uncertainty interval or guaranteed predictive advantage.',
    }
