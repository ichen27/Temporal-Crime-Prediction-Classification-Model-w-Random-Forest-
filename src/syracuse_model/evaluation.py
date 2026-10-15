"""Chronological model selection with a held-out calendar year."""

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from syracuse_crime import load_records

SEED = 42


def features(dates):
    dates = pd.to_datetime(dates)
    if dates.isna().any():
        raise ValueError("Features require valid dates")
    dates = pd.Series(dates).reset_index(drop=True)
    out = pd.DataFrame({"month": dates.dt.month, "day_of_week": dates.dt.dayofweek,
                        "day": dates.dt.day})
    for column, period, offset in (("month", 12, 1), ("day_of_week", 7, 0), ("day", 31, 1)):
        angle = 2 * np.pi * (out[column] - offset) / period
        out[column + "_sin"] = np.sin(angle)
        out[column + "_cos"] = np.cos(angle)
    return out


def split_records(records):
    """No date can straddle a partition; partial 2024 remains outside the experiment."""
    train = records.loc[records.date.between('2020-01-01', '2021-12-31')].copy()
    validation = records.loc[records.date.between('2022-01-01', '2022-12-31')].copy()
    test = records.loc[records.date.between('2023-01-01', '2023-12-31')].copy()
    for name, frame in [('train', train), ('validation', validation), ('test', test)]:
        if frame.empty:
            raise ValueError(f"The {name} partition is empty")
    if train.category.nunique() < 2:
        raise ValueError("Training requires at least two categories")
    return train, validation, test


def candidates():
    return {
        'prior_baseline': DummyClassifier(strategy='prior'),
        'logistic': make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, max_iter=2000, random_state=SEED)),
        'forest_regularized': RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            random_state=SEED, n_jobs=2),
        'forest_balanced': RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            class_weight='balanced', random_state=SEED, n_jobs=2),
    }


def evaluate(model, frame):
    x, y = features(frame.date), frame.category
    predicted = model.predict(x)
    # Include unseen evaluation labels rather than dropping these hard cases.
    labels = sorted(set(model.classes_) | set(y))
    probabilities = np.zeros((len(frame), len(labels)))
    raw_probabilities = model.predict_proba(x)
    for i, label in enumerate(model.classes_):
        probabilities[:, labels.index(label)] = raw_probabilities[:, i]
    return {
        'rows': len(frame), 'accuracy': float(accuracy_score(y, predicted)),
        'macro_f1': float(f1_score(y, predicted, labels=labels, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y, predicted, labels=labels, average='weighted', zero_division=0)),
        'log_loss': float(log_loss(y, probabilities, labels=labels)),
        'unseen_category_rows': int((~y.isin(model.classes_)).sum()),
        'class_report': classification_report(y, predicted, labels=labels, output_dict=True,
                                             zero_division=0),
        'confusion_labels': labels,
        'confusion_matrix': confusion_matrix(y, predicted, labels=labels).tolist(),
    }


def train_and_evaluate(database, output):
    records, audit = load_records(database)
    train, validation, test = split_records(records)
    estimators = candidates()
    validation_scores = {}
    for name, estimator in estimators.items():
        estimator.fit(features(train.date), train.category)
        validation_scores[name] = evaluate(estimator, validation)
    # Choice is frozen before accessing test outcomes. Stable insertion order breaks ties.
    winner = max(validation_scores, key=lambda name: validation_scores[name]['macro_f1'])
    fit_data = pd.concat([train, validation], ignore_index=True)
    selected = clone(estimators[winner]).fit(features(fit_data.date), fit_data.category)
    baseline = DummyClassifier(strategy='prior').fit(features(fit_data.date), fit_data.category)
    final_scores = {'selected_model': evaluate(selected, test), 'prior_baseline': evaluate(baseline, test)}
    model_path = Path(output)
    model_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        'selected_candidate': winner, 'selection_metric': '2022 macro F1',
        'seed': SEED, 'feature_columns': list(features(train.date).columns),
        'training_start': str(fit_data.date.min().date()),
        'training_end': str(fit_data.date.max().date()),
        'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
        'pandas': pd.__version__, 'numpy': np.__version__,
        'data_sha256': audit['database_sha256'],
        'interpretation': 'Category distribution conditional on a recorded incident; not occurrence risk.',
        'probabilities': 'Uncalibrated scores; no individual or location prediction.',
    }
    joblib.dump({'model': selected, 'metadata': metadata}, model_path / 'model.joblib', compress=3)
    report = {
        'metadata': metadata, 'data_audit': audit,
        'partitions': {name: {'rows': len(frame), 'start': str(frame.date.min().date()),
                              'end': str(frame.date.max().date())}
                       for name, frame in [('train', train), ('validation', validation), ('test', test)]},
        'excluded_from_experiment': len(records) - len(train) - len(validation) - len(test),
        'validation': validation_scores, 'test': final_scores,
    }
    (model_path / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    (model_path / 'model-card.json').write_text(json.dumps(metadata, indent=2) + '\n')
    write_report(report, model_path)
    return report


def write_report(report, output):
    lines = ['# Temporal classification benchmark', '',
             'Training: 2020–2021. Model selection: 2022 macro F1. Final refit: 2020–2022. '
             'Untouched test: 2023. Partial 2024 is excluded. Records sharing a date stay together.', '',
             '| Candidate | Validation macro F1 | Accuracy | Log loss |',
             '|---|---:|---:|---:|']
    for name, scores in report['validation'].items():
        lines.append(f"| {name} | {scores['macro_f1']:.4f} | {scores['accuracy']:.4f} | {scores['log_loss']:.4f} |")
    selected = report['metadata']['selected_candidate']
    lines += ['', f'Selected **{selected}** using validation results only.', '',
              '| 2023 test | Macro F1 | Accuracy | Log loss |', '|---|---:|---:|---:|']
    for name, scores in report['test'].items():
        lines.append(f"| {name} | {scores['macro_f1']:.4f} | {scores['accuracy']:.4f} | {scores['log_loss']:.4f} |")
    delta = report['test']['selected_model']['macro_f1'] - report['test']['prior_baseline']['macro_f1']
    lines += ['', f'Test macro F1 difference versus the prior baseline: **{delta:+.4f}**.', '',
              '## Interpretation and limits', '',
              '- This predicts a category conditional on a source record, not whether a crime will occur.',
              '- Scores are uncalibrated; balanced class weights can distort category probabilities.',
              '- Date alone contains limited information. Rare-class performance and the confusion matrix '
              'are included in `metrics.json`; accuracy alone is inadequate.',
              '- One validation year and one test year do not establish robustness across future years.',
              '- Source records are not deduplicated unique incidents; reporting and coverage may change.',
              '- Splits use DATEEND, not availability timestamps. This retrospective snapshot cannot establish what was known at each historical training cutoff.',
              '- Dates preserve the source calendar day. Hour, address, person, and location are not used.',
              '- No performance claim is made against the old random-split notebook on different data.',
              '- The saved model retains the evaluated 2020–2022 fit; it is not a current operational model.',
              '', 'See `model-card.json` for versions and the input database SHA-256.']
    (output / 'benchmark.md').write_text('\n'.join(lines) + '\n')
