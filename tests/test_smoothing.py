import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing

from syracuse_model.smoothing import (
    State,
    fit_state,
    forecast_saved,
    rolling_smoothing,
    run_smoothing,
    select_method,
)


@pytest.fixture
def series():
    return pd.Series(np.random.default_rng(73).poisson(50, 208).astype(float),
                     index=pd.date_range('2020-01-06', periods=208, freq='W-MON'))


@pytest.mark.parametrize('method', ['simple', 'damped_holt'])
def test_updates_match_statsmodels_fixed_parameter_recursion(method):
    values = np.array([12, 9, 14, 10, 17, 11, 15, 16, 19, 13, 18, 21], dtype=float)
    if method == 'simple':
        state = State(10, 0, 0.3, 0, 1)
        fit = SimpleExpSmoothing(values, initialization_method='known', initial_level=10).fit(
            smoothing_level=0.3, optimized=False)
    else:
        state = State(10, 1, 0.3, 0.2, 0.9)
        fit = Holt(values, initialization_method='known', initial_level=10,
                   initial_trend=1, damped_trend=True).fit(
                       smoothing_level=0.3, smoothing_trend=0.2,
                       damping_trend=0.9, optimized=False)
    predictions = []
    for value in values:
        predictions.append(state.forecast())
        state.update(value)
    np.testing.assert_allclose(predictions, fit.fittedvalues)
    assert state.forecast() == pytest.approx(float(fit.forecast(1)[0]))


@pytest.mark.parametrize('method', ['simple', 'damped_holt'])
def test_optimized_state_matches_library_forecast(series, method):
    state = fit_state(series, method)
    model = (SimpleExpSmoothing(series.to_numpy(), initialization_method='estimated')
             if method == 'simple' else Holt(series.to_numpy(), damped_trend=True,
                                             initialization_method='estimated'))
    assert state.forecast() == pytest.approx(float(model.fit().forecast(1)[0]))
    assert fit_state([0] * 30, method).forecast() == 0


def test_predictions_do_not_see_current_or_future_observations(series):
    changed = series.copy()
    changed.loc['2023-02-20':] += 100
    before, fits = rolling_smoothing(series, '2023-01-01', '2023-03-31')
    after, _ = rolling_smoothing(changed, '2023-01-01', '2023-03-31')
    columns = ['simple', 'damped_holt', 'recent_mean']
    pd.testing.assert_frame_equal(before.loc[:'2023-02-20', columns],
                                  after.loc[:'2023-02-20', columns])
    assert before.loc['2023-02-27', 'simple'] != after.loc['2023-02-27', 'simple']
    assert all(fit['training_through'] < fit['origin'] for fit in fits)
    assert [fit['origin'] for fit in fits] == ['2023-01-02', '2023-02-06', '2023-03-06']


def test_selection_independent_of_2023(series):
    changed = series.copy()
    changed.loc['2023-01-02':] = 1000
    before = select_method(series)
    after = select_method(changed)
    assert before[:2] == after[:2]
    pd.testing.assert_frame_equal(before[2], after[2])


def test_json_state_roundtrip_and_consecutive_observations():
    bundle = {'schema_version': 1, 'method': 'simple',
              'state': asdict(State(10, 0, 0.3, 0, 1)),
              'next_period_start': '2023-01-02', 'last_four_counts': [8, 9, 10, 13]}
    restored = json.loads(json.dumps(bundle))
    assert forecast_saved(restored)['model_expected_count'] == 10
    result = forecast_saved(restored, [{'period_start': '2023-01-02', 'count': 20}])
    assert result['model_expected_count'] == 13
    assert result['recent_mean_baseline'] == 13
    assert result['period_start'] == '2023-01-09'
    assert result['period_end'] == '2023-01-15'
    assert restored == bundle
    with pytest.raises(ValueError, match='consecutive'):
        forecast_saved(restored, [{'period_start': '2023-01-09', 'count': 20}])
    for count in [True, '10', -1, 1.5, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='Counts'):
            forecast_saved(restored, [{'period_start': '2023-01-02', 'count': count}])


def test_invalid_training_history(series):
    with pytest.raises(ValueError, match='consecutive'):
        rolling_smoothing(series.drop(series.index[10]), '2023-01-01', '2023-12-31')
    with pytest.raises(ValueError, match='20 prior'):
        rolling_smoothing(series, '2020-01-01', '2020-01-31')
    with pytest.raises(ValueError, match='Unknown'):
        fit_state(series, 'unknown')


def test_experiment_sensitivity_and_serialized_forecast(tmp_path, series, monkeypatch):
    import syracuse_model.smoothing as smoothing

    monkeypatch.setattr(smoothing, 'load_records', lambda _: (None, {'database_sha256': 'fixture'}))
    monkeypatch.setattr(smoothing, 'count_series', lambda *_: series)
    report = run_smoothing('unused', tmp_path)
    method = report['selection']['model']
    assert report['primary'][method]['periods'] == 52
    assert report['sensitivity_through_2023_12_24'][method]['periods'] == 51
    rows = report['test_predictions']
    errors = [abs(row['actual'] - row[method]) for row in rows[:-1]]
    assert report['sensitivity_through_2023_12_24'][method]['mae'] == pytest.approx(np.mean(errors))
    bundle = json.loads((tmp_path / 'model.json').read_text())
    assert bundle['training_through'] == '2023-01-01'
    assert forecast_saved(bundle)['model_expected_count'] == rows[0][method]
    assert json.loads((tmp_path / 'selection.json').read_text()) == report['selection']


def test_convergence_retry_is_recorded_and_failure_is_explicit(series, monkeypatch):
    import syracuse_model.smoothing as smoothing

    original = SimpleExpSmoothing.fit
    called = []

    def fail_first(self, **kwargs):
        called.append(kwargs['method'])
        if kwargs['method'] == 'L-BFGS-B':
            raise smoothing.ConvergenceWarning('synthetic failure')
        return original(self, **kwargs)

    monkeypatch.setattr(SimpleExpSmoothing, 'fit', fail_first)
    state = fit_state(series, 'simple')
    assert called == ['L-BFGS-B', 'SLSQP']
    assert state.optimizer == 'SLSQP'

    def fail_both(*args, **kwargs):
        raise smoothing.ConvergenceWarning('synthetic failure')

    monkeypatch.setattr(SimpleExpSmoothing, 'fit', fail_both)
    with pytest.raises(ValueError, match='either solver'):
        fit_state(series, 'simple')
