"""Readable evidence for the target-screening experiment."""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    'daily_total': 'Next-day total records',
    'weekly_total': 'Next-week total records',
    'weekly_larceny': 'Next-week larceny records',
}


def write_forecast_report(report, output):
    output = Path(output)
    selection = report['selection']
    name, model, baseline = selection['target']['name'], selection['model'], selection['baseline']
    test = report['test']
    skill = report['test_mae_skill_vs_selected_baseline']
    lines = ['# Syracuse count-forecasting experiment', '',
             '## Input → target', '',
             f"Selected target: **{LABELS[name]}**. Inputs: prior completed-period counts, "
             'recent averages/variability, and known seasonal calendar features. '
             'The target is a source-record count, not a unique-incident count or individual risk.', '',
             '## Target screening — 2022 only', '',
             'The target and model were fixed before evaluating 2023. All three targets and two '
             'Poisson-loss models were declared before screening. Models refit monthly on expanding '
             'history; predictions use only prior completed periods. Partial boundary weeks are excluded.', '',
             '| Target | Best learned model | Model MAE | Best baseline MAE | MAE reduction |',
             '|---|---|---:|---:|---:|']
    for target_name, result in selection['validation'].items():
        metrics = result['scores']
        lines.append(f"| {LABELS[target_name]} | {result['best_model']} | "
                     f"{metrics[result['best_model']]['mae']:.2f} | "
                     f"{metrics[result['best_baseline']]['mae']:.2f} | "
                     f"{result['mae_skill_vs_best_baseline']:.1%} |")
    lines += ['', 'MAE means mean absolute error in records per forecast period. '
              'Target selection uses relative reduction against each target’s best baseline, '
              'not raw errors across different count scales. The baselines are the previous period, '
              'a recent mean (28 days or four weeks), and seasonal naïve (7 days or 52 weeks).', '',
              '## Fixed-choice evaluation — 2023', '',
              f"**{LABELS[name]}**, using **{model}**, across **{test[model]['periods']}** periods.", '',
              '| Method | MAE (records) | RMSE (records) |', '|---|---:|---:|']
    for method, metrics in test.items():
        lines.append(f"| {method} | {metrics['mae']:.2f} | {metrics['rmse']:.2f} |")
    coverage = report['metadata']['test_coverage']
    conclusion = ('The learned model reduced error on this test.' if skill > 0 else
                  'The validation gain did not generalize: the learned model did not beat the baseline.')
    lines += ['', f"MAE reduction versus the validation-selected **{baseline}**: **{skill:.1%}**. "
              + conclusion, '',
              'This experiment supports count forecasting as a more interpretable task, but does '
              'not establish useful learned predictive skill. Do not select another target or tune '
              'hyperparameters using these test results and continue calling 2023 untouched.', '',
              '## Source coverage check', '',
              f"The last observed selected-category date in 2023 is **{coverage['last_observed_target_date_in_test']}**. "
              f"Forecast periods ending after that date: **{', '.join(coverage['periods_ending_after_last_observed_record']) or 'none'}**. "
              'Where flagged, treat the tail as a collection-coverage concern, not '
              'evidence of a true crime decline. Scores retain the predefined calendar window; '
              'no favorable post-result exclusion was made. This further limits the benchmark’s interpretation.', '',
              '![Observed and predicted counts](forecast.png)', '',
              '## Evaluation meaning', '',
              '- Each prediction is one period ahead. Earlier 2023 outcomes become observed history '
              'for later 2023 forecasts and monthly refits. This is not a year-ahead forecast.',
              '- The saved model is the initial fit using only complete periods before 2023. '
              'It is not the series of monthly refits used for the rolling evaluation.',
              '- Count-based features are shifted before rolling averages. Future target values '
              'cannot change earlier predictions or the validation-selected specification.',
              '- This is a retrospective DATEEND snapshot. Reporting delays and collection gaps '
              'cannot be reconstructed; calendar gaps are treated as zero records under an explicit '
              'coverage assumption, not proof of zero crime.',
              '- Original category classification remains available as a historical experiment. '
              'Its accuracy/F1 metrics are not comparable to count-forecast MAE.',
              '- No calibrated uncertainty intervals, current-data forecast, or deployment are claimed.', '',
              'Source provenance, immutable input hash, complete validation scores, and per-period '
              'predictions are in `metrics.json`. The frozen specification is in `selection.json`.']
    (output / 'findings.md').write_text('\n'.join(lines) + '\n')
    frame = pd.DataFrame(report['test_predictions'])
    dates = pd.to_datetime(frame.period_start)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    ax.plot(dates, frame.actual, color='#283747', linewidth=2, label='Observed records')
    ax.plot(dates, frame[model], color='#176b91', label=f'Learned model: {model}')
    ax.plot(dates, frame[baseline], color='#b96625', linestyle='--', label='Four-week mean' if selection['target']['days'] == 7 else '28-day mean')
    if coverage['periods_ending_after_last_observed_record']:
        tail_start = pd.Timestamp(coverage['last_observed_target_date_in_test'])
        ax.axvspan(tail_start, pd.Timestamp('2023-12-31'), alpha=0.15, color='#b96625',
                   label='Unverified source tail')
    ax.set(title=f'{LABELS[name]} · rolling 2023 evaluation', xlabel='Forecast period start',
           ylabel='Source records')
    ax.legend(frameon=False)
    fig.savefig(output / 'forecast.png', dpi=170)
    plt.close(fig)
