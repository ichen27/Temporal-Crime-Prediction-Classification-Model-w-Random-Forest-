"""Exploratory weekly exponential smoothing with past-only monthly fits."""

import json
import platform
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
from syracuse_crime import load_records

from .forecasting import Target, count_series, prediction_rows, scores

CANDIDATES = ('simple', 'damped_holt')
TARGET = Target('weekly_larceny', 7, 'LARCENY')


@dataclass
class State:
    level: float
    trend: float
    alpha: float
    beta: float
    phi: float
    optimizer: str = 'manual'

    def forecast(self):
        value = self.level + self.phi * self.trend
        if not np.isfinite(value):
            raise ValueError('Nonfinite forecast')
        return max(0.0, value)

    def update(self, count):
        count = valid_count(count)
        old_level = self.level
        self.level = self.alpha * count + (1 - self.alpha) * (
            old_level + self.phi * self.trend
        )
        self.trend = self.beta * (self.level - old_level) + (
            1 - self.beta
        ) * self.phi * self.trend


def valid_count(value):
    if (isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value) or value < 0 or float(value) != int(value)):
        raise ValueError('Counts must be finite nonnegative integers')
    return float(value)


def fit_state(history, method):
    if method not in CANDIDATES:
        raise ValueError('Unknown smoothing method')
    values = np.asarray([valid_count(v) for v in history], dtype=float)
    if len(values) < 20:
        raise ValueError('Need at least 20 complete weeks')
    # Constant data has an exact solution; avoids undefined likelihood diagnostics.
    if np.ptp(values) == 0:
        return State(float(values[0]), 0.0, 0.2, 0.0, 1.0, 'exact_constant')
    model = (SimpleExpSmoothing(values, initialization_method='estimated')
             if method == 'simple' else
             Holt(values, damped_trend=True, initialization_method='estimated'))
    # Retry only on numerical failure, using the same objective, never test error.
    for optimizer in ('L-BFGS-B', 'SLSQP'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                fit = model.fit(optimized=True, remove_bias=False, method=optimizer)
            if fit.mle_retvals['success']:
                break
        except ConvergenceWarning:
            continue
    else:
        raise ValueError('Smoothing optimization did not converge with either solver')
    return State(float(fit.level[-1]),
                 float(fit.trend[-1]) if method != 'simple' else 0.0,
                 float(fit.params['smoothing_level']),
                 float(fit.params['smoothing_trend']) if method != 'simple' else 0.0,
                 float(fit.params['damping_trend']) if method != 'simple' else 1.0, optimizer)


def rolling_smoothing(series, start, end, methods=CANDIDATES):
    """Predict before observing each target; refit at first origin of each month."""
    if (len(series) == 0 or not isinstance(series.index, pd.DatetimeIndex) or series.index.has_duplicates
            or not series.index.equals(pd.date_range(series.index[0], periods=len(series),
                                                       freq='W-MON'))):
        raise ValueError('History must contain consecutive Monday weeks')
    for value in series:
        valid_count(value)
    origins = series.index[(series.index >= pd.Timestamp(start)) &
                           (series.index + pd.Timedelta(6, unit='D') <= pd.Timestamp(end))]
    if not len(origins) or len(series.loc[series.index < origins[0]]) < 20:
        raise ValueError('Need evaluation weeks and at least 20 prior weeks')
    output = pd.DataFrame({'actual': series.loc[origins],
                           'recent_mean': series.shift(1).rolling(4).mean().loc[origins]})
    states, month, fits = {}, None, []
    for origin in origins:
        if origin.to_period('M') != month:
            history = series.loc[series.index < origin]
            states = {method: fit_state(history, method) for method in methods}
            month = origin.to_period('M')
            fits.append({'origin': str(origin.date()), 'training_weeks': len(history),
                         'training_through': str((history.index[-1] + pd.Timedelta(6, unit='D')).date()),
                         'states': {name: asdict(state) for name, state in states.items()}})
        for name, state in states.items():
            output.loc[origin, name] = state.forecast()
            state.update(series.loc[origin])
    return output, fits


def select_method(series):
    validation, fits = rolling_smoothing(series, '2022-01-01', '2022-12-31')
    metrics = scores(validation)
    winner = min(CANDIDATES, key=lambda name: metrics[name]['mae'])
    return winner, metrics, validation, fits


def forecast_saved(bundle, observations=()):
    """Advance a historical state with consecutive observations; no parameter refit."""
    if bundle['schema_version'] != 1 or bundle['method'] not in CANDIDATES:
        raise ValueError('Unsupported model artifact')
    state = State(**bundle['state'])
    if (not all(np.isfinite(v) for v in (state.level, state.trend, state.alpha, state.beta, state.phi))
            or not all(0 <= v <= 1 for v in (state.alpha, state.beta, state.phi))):
        raise ValueError('Invalid smoothing state')
    origin = pd.Timestamp(bundle['next_period_start'])
    if pd.isna(origin) or origin.dayofweek != 0 or origin != origin.normalize() or origin.tz:
        raise ValueError('Model origin must be a date-only Monday')
    recent = [valid_count(v) for v in bundle['last_four_counts']]
    if len(recent) != 4:
        raise ValueError('Model needs four baseline counts')
    if not isinstance(observations, (list, tuple)):
        raise ValueError('Observations must be a list')
    for row in observations:
        if pd.Timestamp(row['period_start']) != origin:
            raise ValueError('Observations must be consecutive weeks starting at model origin')
        count = valid_count(row['count'])
        state.update(count)
        recent = recent[1:] + [count]
        origin += pd.Timedelta(7, unit='D')
    return {'period_start': str(origin.date()),
            'period_end': str((origin + pd.Timedelta(6, unit='D')).date()),
            'method': bundle['method'], 'model_expected_count': state.forecast(),
            'recent_mean_baseline': float(np.mean(recent)),
            'limitation': 'Historical source-record forecast; fixed parameters between updates. '
                          'No calibrated interval or validated current deployment.'}


def run_smoothing(database, output):
    records, audit = load_records(database)
    series = count_series(records, TARGET)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    winner, validation_scores, validation, validation_fits = select_method(series)
    selection = {'target': asdict(TARGET), 'candidates': list(CANDIDATES), 'model': winner,
                 'criterion': 'Minimum 2022 rolling MAE; first candidate wins ties',
                 'validation': validation_scores, 'seasonal': False,
                 'status': 'Exploratory: 2023 outcomes were already examined in prior experiments',
                 'sensitivity': 'Exclude the previously flagged December 25–31 week for both methods'}
    (output / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    test, test_fits = rolling_smoothing(series, '2023-01-01', '2023-12-31', (winner,))
    first_origin = test.index[0]
    history = series.loc[series.index < first_origin]
    bundle = {'schema_version': 1, 'method': winner,
              'state': test_fits[0]['states'][winner],
              'next_period_start': str(first_origin.date()),
              'training_through': test_fits[0]['training_through'],
              'last_four_counts': history.tail(4).astype(int).tolist(),
              'database_sha256': audit['database_sha256'],
              'python': platform.python_version(), 'statsmodels': statsmodels.__version__}
    report = {'selection': selection, 'model_card': bundle, 'data_audit': audit,
              'weekly_observations': len(series),
              'primary': scores(test), 'sensitivity_through_2023_12_24': scores(test.loc[:'2023-12-18']),
              'validation_predictions': prediction_rows(validation),
              'test_predictions': prediction_rows(test),
              'validation_fits': validation_fits, 'test_fits': test_fits}
    for name, data in [('model.json', bundle), ('metrics.json', report),
                       ('example-forecast.json', forecast_saved(bundle))]:
        (output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    return report
