"""Train a reproducible benchmark or inspect date-conditioned category scores."""

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from .evaluation import features, train_and_evaluate


def predict(bundle, date_string):
    day = date.fromisoformat(date_string)
    if day.isoformat() != date_string:
        raise ValueError('Date must use YYYY-MM-DD')
    x = features(pd.Series([pd.Timestamp(day)]))
    metadata, model = bundle['metadata'], bundle['model']
    probabilities = model.predict_proba(x)[0]
    scores = sorted(zip(model.classes_, probabilities), key=lambda pair: -pair[1])
    return {
        'date': date_string,
        'interpretation': metadata['interpretation'],
        'outside_training_window': not metadata['training_start'] <= date_string <= metadata['training_end'],
        'warning': 'Historical, uncalibrated category scores; not the probability that a crime occurs.',
        'category_scores': {str(label): float(value) for label, value in scores},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--database', type=Path, required=True)
    train.add_argument('--output', type=Path, default=Path('reports'))
    prediction = sub.add_parser('predict', description='Load only a model file you trust.')
    prediction.add_argument('--model', type=Path, default=Path('reports/model.joblib'))
    prediction.add_argument('--date', required=True, help='YYYY-MM-DD')
    args = parser.parse_args()
    try:
        if args.command == 'train':
            report = train_and_evaluate(args.database, args.output)
            print(json.dumps({'selected': report['metadata']['selected_candidate'],
                              'test_macro_f1': report['test']['selected_model']['macro_f1']}, indent=2))
        else:
            print(json.dumps(predict(joblib.load(args.model), args.date), indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
