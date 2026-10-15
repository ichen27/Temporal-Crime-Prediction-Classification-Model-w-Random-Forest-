import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from syracuse_model.cli import predict
from syracuse_model.evaluation import evaluate, features, split_records, train_and_evaluate


def test_date_groups_stay_chronological():
    records = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-01', '2021-12-31', '2021-12-31',
                                '2022-01-01', '2023-01-01', '2024-01-01']),
        'category': ['A', 'B', 'A', 'B', 'A', 'A'],
    })
    train, validation, test = split_records(records)
    assert len(train) == 3
    assert train.date.max() < validation.date.min() < test.date.min()
    assert len(train) + len(validation) + len(test) == 5


def test_empty_partition_rejected():
    records = pd.DataFrame({'date': pd.to_datetime(['2020-01-01']), 'category': ['A']})
    with pytest.raises(ValueError, match='validation partition'):
        split_records(records)


def test_unseen_labels_are_counted_in_metrics():
    train = pd.DataFrame({'date': pd.to_datetime(['2020-01-01', '2020-01-02']),
                          'category': ['A', 'B']})
    model = DummyClassifier(strategy='prior').fit(features(train.date), train.category)
    evaluation = pd.DataFrame({'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
                               'category': ['A', 'NEW']})
    scores = evaluate(model, evaluation)
    assert scores['unseen_category_rows'] == 1
    assert scores['class_report']['NEW']['recall'] == 0
    assert np.isfinite(scores['log_loss'])
    assert sum(map(sum, scores['confusion_matrix'])) == 2


def test_features_have_no_target_or_year_and_reject_invalid_dates():
    x = features(pd.Series(pd.to_datetime(['2020-02-29', '2021-02-28'])))
    assert len(x.columns) == 9
    assert 'category' not in x and 'year' not in x
    assert np.isfinite(x).all().all()
    with pytest.raises(ValueError, match='valid dates'):
        features(pd.Series([pd.NaT]))


def test_predict_validates_calendar_and_preserves_labels():
    x = features(pd.Series(pd.to_datetime(['2020-01-01', '2020-01-02'])))
    model = DummyClassifier(strategy='prior').fit(x, ['B', 'A'])
    bundle = {'model': model, 'metadata': {'training_start': '2020-01-01',
              'training_end': '2022-12-31', 'interpretation': 'conditional category'}}
    result = predict(bundle, '2023-02-28')
    assert result['outside_training_window']
    assert result['category_scores'] == {'A': 0.5, 'B': 0.5}
    with pytest.raises(ValueError):
        predict(bundle, '2023-02-29')


def test_end_to_end_saved_model_and_selection(tmp_path, monkeypatch):
    import sqlite3

    from syracuse_model import evaluation

    # Small genuine database exercises the loader, split, refit, reports and artifact.
    db = tmp_path / 'records.db'
    with sqlite3.connect(db) as conn:
        for year in (2020, 2021, 2022, 2023):
            conn.execute(f'CREATE TABLE crime_{year}_1 (DATEEND TEXT, CODE_DEFINED TEXT)')
            conn.executemany(f'INSERT INTO crime_{year}_1 VALUES (?, ?)',
                             [(f'{year}/01/01', 'A'), (f'{year}/02/01', 'B')])
    monkeypatch.setattr(evaluation, 'candidates', lambda: {
        'prior_baseline': DummyClassifier(strategy='prior'),
        'constant_b': DummyClassifier(strategy='constant', constant='B'),
    })
    report = train_and_evaluate(db, tmp_path / 'reports')
    assert report['metadata']['selected_candidate'] == 'prior_baseline'  # stable tie
    bundle = joblib.load(tmp_path / 'reports/model.joblib')
    assert bundle['metadata']['training_end'] == '2022-02-01'
    assert sum(predict(bundle, '2023-01-01')['category_scores'].values()) == 1
    saved = json.loads((tmp_path / 'reports/metrics.json').read_text())
    assert saved['test']['selected_model']['rows'] == 2
    # Changing only the held-out labels must not affect selection or the saved fit.
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE crime_2023_1 SET CODE_DEFINED = 'UNSEEN'")
    changed = train_and_evaluate(db, tmp_path / 'changed')
    assert changed['metadata']['selected_candidate'] == report['metadata']['selected_candidate']
    changed_bundle = joblib.load(tmp_path / 'changed/model.joblib')
    assert predict(changed_bundle, '2023-01-01') == predict(bundle, '2023-01-01')
    assert changed['test']['selected_model']['unseen_category_rows'] == 2


def test_download_verifies_existing_file_without_overwriting(tmp_path, monkeypatch):
    import hashlib

    from syracuse_model import download

    path = tmp_path / 'data.db'
    path.write_bytes(b'existing input')
    with pytest.raises(ValueError, match='different content'):
        download.download(path)
    assert path.read_bytes() == b'existing input'
    monkeypatch.setattr(download, 'EXPECTED_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    assert download.download(path) == path


def test_download_rejects_corrupt_response(tmp_path, monkeypatch):
    import io

    from syracuse_model import download

    monkeypatch.setattr(download.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'corrupt'))
    path = tmp_path / 'data.db'
    with pytest.raises(ValueError, match='SHA-256'):
        download.download(path)
    assert not path.exists()
