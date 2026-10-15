"""Evaluate exponential smoothing or use its inspectable historical JSON state."""

import argparse
import json
from pathlib import Path

from .smoothing import forecast_saved, run_smoothing
from .smoothing_report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    experiment = commands.add_parser('experiment')
    experiment.add_argument('--database', type=Path, required=True)
    experiment.add_argument('--output', type=Path, default=Path('reports/smoothing'))
    predict = commands.add_parser('predict')
    predict.add_argument('--model', type=Path, default=Path('reports/smoothing/model.json'))
    predict.add_argument('--observations', type=Path,
                         help='Optional consecutive weekly observations after the saved state')
    args = parser.parse_args()
    try:
        if args.command == 'experiment':
            result = run_smoothing(args.database, args.output)
            write_report(result, args.output)
            print(json.dumps({'model': result['selection']['model'],
                              'primary': result['primary'],
                              'sensitivity': result['sensitivity_through_2023_12_24']}, indent=2))
        else:
            observations = json.loads(args.observations.read_text()) if args.observations else []
            print(json.dumps(forecast_saved(json.loads(args.model.read_text()), observations), indent=2))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
