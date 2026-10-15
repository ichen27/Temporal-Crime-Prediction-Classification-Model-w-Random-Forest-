"""Screen aggregate count targets or predict a next period from explicit history."""

import argparse
import json
from pathlib import Path

import joblib

from .forecast_report import write_forecast_report
from .forecasting import forecast_from_history, run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    experiment = commands.add_parser('experiment')
    experiment.add_argument('--database', type=Path, required=True)
    experiment.add_argument('--output', type=Path, default=Path('reports/forecasting'))
    predict = commands.add_parser('predict', description='Load only trusted joblib files.')
    predict.add_argument('--model', type=Path, default=Path('reports/forecasting/forecast.joblib'))
    predict.add_argument('--history', type=Path, required=True,
                         help='JSON list of consecutive period_start/count objects')
    predict.add_argument('--period-start', required=True, help='YYYY-MM-DD; Monday for weekly models')
    args = parser.parse_args()
    try:
        if args.command == 'experiment':
            report = run_experiment(args.database, args.output)
            write_forecast_report(report, args.output)
            print(json.dumps({'selected_target': report['selection']['target'],
                              'model': report['selection']['model'],
                              'test_mae_skill': report['test_mae_skill_vs_selected_baseline']}, indent=2))
        else:
            history = json.loads(args.history.read_text())
            print(json.dumps(forecast_from_history(joblib.load(args.model), history,
                                                   args.period_start), indent=2))
    except (ValueError, FileNotFoundError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
