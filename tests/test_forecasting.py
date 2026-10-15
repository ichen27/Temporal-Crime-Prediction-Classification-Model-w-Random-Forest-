import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from syracuse_model.forecasting import (
    TARGETS,
    baseline_predictions,
    count_series,
    design,
    forecast_from_history,
    rolling_predictions,
    run_experiment,
    select_target,
)


@pytest.fixture
def records():
    dates = pd.date_range('2020-01-01', '2023-12-31')
    rows = []
    for day in dates:
        rows.extend([(day, 'LARCENY')] * (2 + day.dayofweek % 3))
        rows.append((day, 'OTHER'))
    return pd.DataFrame(rows, columns=['date', 'category'])


def test_counts_keep_zero_days_and_exclude_partial_weeks():
    records = pd.DataFrame({'date': pd.to_datetime(['2020-01-06', '2020-01-12',
                                                   '2020-01-13', '2020-01-19']),
                            'category': ['LARCENY', 'OTHER', 'OTHER', 'LARCENY']})
    daily = count_series(records, TARGETS[0], end='2020-01-19')
    assert len(daily) == 19 and daily.sum() == 4
    weekly = count_series(records, TARGETS[1], end='2020-01-19')
    assert list(weekly.index) == list(pd.to_datetime(['2020-01-06', '2020-01-13']))
    assert list(weekly) == [2, 2]  # the final Sunday closes a full week
    category = count_series(records, TARGETS[2], end='2020-01-18')
    assert list(category) == [1]  # only one complete week, one matching category


def test_snapshot_years_have_208_complete_weeks(records):
    assert len(count_series(records, TARGETS[1])) == 208


@pytest.mark.parametrize('target', TARGETS)
def test_features_and_baselines_cannot_see_current_or_future_target(records, target):
    series = count_series(records, target)
    origin = series.index[100]
    changed = series.copy()
    changed.loc[origin:] += 1000
    pd.testing.assert_frame_equal(design(series, target).loc[:origin],
                                  design(changed, target).loc[:origin])
    pd.testing.assert_frame_equal(baseline_predictions(series, target).loc[:origin],
                                  baseline_predictions(changed, target).loc[:origin])


def test_validation_choice_independent_of_test_records(records):
    estimators = {'mean_regressor': DummyRegressor()}
    before, selected, _ = select_target(records, estimators)
    changed = pd.concat([records, records.loc[records.date.ge('2023-01-01')]],
                        ignore_index=True)
    after, selected_after, _ = select_target(changed, estimators)
    assert selected_after == selected
    assert before == after


def test_rolling_forecast_only_adapts_after_observation(records):
    target = TARGETS[1]
    series = count_series(records, target)
    changed = series.copy()
    changed.loc['2023-01-16'] = 10000
    estimators = {'mean_regressor': DummyRegressor()}
    before = rolling_predictions(series, target, '2023-01-01', '2023-02-28', estimators)
    after = rolling_predictions(changed, target, '2023-01-01', '2023-02-28', estimators)
    columns = ['mean_regressor', 'recent_mean', 'last_period', 'seasonal_naive']
    pd.testing.assert_frame_equal(before.loc[:'2023-01-16', columns],
                                  after.loc[:'2023-01-16', columns])
    assert before.loc['2023-01-23', 'recent_mean'] != after.loc['2023-01-23', 'recent_mean']
    assert before.loc['2023-02-06', 'mean_regressor'] != after.loc['2023-02-06', 'mean_regressor']


@pytest.fixture
def bundle_and_history(records):
    target = TARGETS[2]
    series = count_series(records, target)
    x = design(series, target).dropna()
    train_index = x.index[x.index < '2022-12-26']
    model = DummyRegressor().fit(x.loc[train_index], series.loc[train_index])
    bundle = {'model': model, 'metadata': {
        'target': {'name': target.name, 'days': 7, 'category': 'LARCENY'},
        'feature_columns': list(x.columns), 'training_through': '2022-12-25',
        'interpretation': 'Source record count',
    }}
    history = [{'period_start': str(day.date()), 'count': int(value)}
               for day, value in series.loc[:'2022-12-26'].tail(13).items()]
    return bundle, history


def test_prediction_roundtrip_and_history_validation(bundle_and_history, tmp_path):
    bundle, history = bundle_and_history
    path = tmp_path / 'forecast.joblib'
    joblib.dump(bundle, path)
    result = forecast_from_history(joblib.load(path), history, '2023-01-02')
    assert result['period_end'] == '2023-01-08'
    assert result['model_expected_count'] > 0
    assert result['recent_mean_baseline'] == np.mean([row['count'] for row in history[-4:]])
    with pytest.raises(ValueError, match='13 completed'):
        forecast_from_history(bundle, history[-12:], '2023-01-02')
    with pytest.raises(ValueError, match='consecutive'):
        forecast_from_history(bundle, history[1:] + history[:1], '2023-01-02')
    with pytest.raises(ValueError, match='training window'):
        forecast_from_history(bundle, history, '2022-01-03')
    with pytest.raises(ValueError):
        forecast_from_history(bundle, history, '2023-02-30')


@pytest.mark.parametrize('invalid', [-1, 0.5, float('nan'), float('inf'), True, '10'])
def test_history_rejects_invalid_counts(bundle_and_history, invalid):
    bundle, history = bundle_and_history
    history[0]['count'] = invalid
    with pytest.raises(ValueError, match='Counts'):
        forecast_from_history(bundle, history, '2023-01-02')


def test_experiment_writes_selection_artifact_and_complete_test(tmp_path, records, monkeypatch):
    import syracuse_model.forecasting as forecasting

    monkeypatch.setattr(forecasting, 'load_records', lambda _: (records, {'database_sha256': 'fixture'}))
    monkeypatch.setattr(forecasting, 'model_candidates', lambda: {'mean_regressor': DummyRegressor()})
    report = run_experiment('unused', tmp_path)
    selection = json.loads((tmp_path / 'selection.json').read_text())
    assert selection == report['selection']
    assert set(selection['validation']) == {'daily_total', 'weekly_total', 'weekly_larceny'}
    assert len(report['test_predictions']) in (52, 365)
    bundle = joblib.load(tmp_path / 'forecast.joblib')
    assert bundle['metadata']['training_through'] < '2023-01-01'
    assert bundle['metadata']['target'] == selection['target']
    assert (tmp_path / 'model-card.json').exists()
