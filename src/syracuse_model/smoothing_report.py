"""Readable findings and an exportable chart for the smoothing comparison."""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {'simple': 'Simple exponential smoothing', 'damped_holt': 'Damped Holt',
          'recent_mean': 'Four-week average'}


def write_report(report, output):
    output = Path(output)
    model = report['selection']['model']
    rows = ['# Weekly larceny: exponential smoothing', '',
            '**Exploratory follow-up:** 2023 was already examined in the earlier experiment. '
            'These results are not a new untouched test.', '',
            '| Evaluation window | Method | MAE (records/week) | RMSE | Weeks |',
            '|---|---|---:|---:|---:|']
    for label, metrics in [('2022 selection', report['selection']['validation']),
                            ('2023 full window', report['primary']),
                            ('2023 through Dec 24', report['sensitivity_through_2023_12_24'])]:
        for name, score in metrics.items():
            rows.append(f'| {label} | {LABELS[name]} | {score["mae"]:.3f} | '
                        f'{score["rmse"]:.3f} | {score["periods"]} |')
    rows.extend(['', f'Chosen using 2022 MAE: **{LABELS[model]}**.', ''])
    for label, key in [('Full 2023 window', 'primary'),
                        ('Excluding the previously flagged final week', 'sensitivity_through_2023_12_24')]:
        metrics = report[key]
        skill = 1 - metrics[model]['mae'] / metrics['recent_mean']['mae']
        rows.append(f'{label}: smoothing MAE is **{abs(skill):.1%} '
                    f'{"lower" if skill > 0 else "higher"}** than the four-week average.')
        rows.append('')
    rows.extend(['## Protocol', '',
                 '- Fixed weekly larceny target; no new target search. 208 complete Monday–Sunday '
                 'weeks in the source window, not 208 independent training cases at every origin.',
                 '- Fit all completed history at the first forecast origin each month. Optimize '
                 'initial states and smoothing parameters by training squared error; choose the '
                 'method by rolling 2022 MAE. Update state after each newly observed count. L-BFGS-B fits retry SLSQP only on convergence failure; each solver is recorded.',
                 '- Predict one week ahead before seeing that week. Negative extrapolations are '
                 'clipped to zero. No seasonal component or post-2023 tuning.',
                 '- Both full-window and final-week-excluded scores were specified before '
                 'computing this experiment. The sensitivity uses the same predictions for both '
                 'methods and does not prove earlier coverage is complete.',
                 '- The JSON artifact is the initial fit through January 1, 2023. It forecasts '
                 'January 2–8; optional observations advance states without refitting parameters. '
                 'The evaluation refits monthly and records every fit separately.', '',
                 '## Coverage and interpretation', '',
                 'The local final week contains four larceny records, ending December 26. '
                 'An independent read of the city’s currently served 2023 Part 1 data agrees '
                 'for December; it also stops on December 26. This locates the concern upstream '
                 'but does not establish whether later dates were missing or truly had no records. '
                 'See the [source audit](source-audit.json). The database was not replaced.', '',
                 'The city’s [2023 metadata](https://www.arcgis.com/sharing/rest/content/items/'
                 '94bc33f1a2c646b995d1ab356edf0730/info/metadata/metadata.xml?format=default&output=html) '
                 'defines DATEEND as the reported date. This corrects the earlier incident-end-date '
                 'description. It does not provide historical ingestion or revision timestamps. '
                 'Counts are retrospective source records, not verified unique incidents. '
                 'Zero-filled gaps remain a coverage assumption.', '',
                 'Small differences here do not establish a reliable forecasting advantage. '
                 'Resolve source coverage and obtain a genuinely untouched evaluation period '
                 'before selecting a deployment model. No calibrated intervals are supplied.', '',
                 '## Reproduce', '', '```bash',
                 'syracuse-smooth experiment --database work/crime_data.db --output reports/smoothing',
                 'syracuse-smooth predict --model reports/smoothing/model.json', '```', '',
                 'The source audit is a separately retrieved, dated external snapshot; rerunning '
                 'the experiment does not silently refresh it. Editable source: '
                 '`src/syracuse_model/smoothing.py` and `smoothing_report.py`.', '',
                 'Methods: [Forecasting: Principles and Practice]'
                 '(https://otexts.com/fpp3/exponential-smoothing.html); '
                 '[statsmodels exponential smoothing]'
                 '(https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html).', ''])
    (output / 'findings.md').write_text('\n'.join(rows))
    frame = pd.DataFrame(report['test_predictions'])
    dates = pd.to_datetime(frame.period_start)
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    ax.plot(dates, frame.actual, color='#343a40', label='Observed records', linewidth=1.3)
    ax.plot(dates, frame.recent_mean, color='#c77c22', label=LABELS['recent_mean'], linewidth=1.6)
    ax.plot(dates, frame[model], color='#176e9c', label=LABELS[model], linewidth=1.6)
    ax.axvspan(pd.Timestamp('2023-12-25'), pd.Timestamp('2023-12-31'),
               color='#bb3333', alpha=0.15, label='Coverage unresolved')
    ax.set(title='Weekly Syracuse larceny records — exploratory 2023 comparison',
           ylabel='Records per Monday–Sunday week', xlabel='Week starting')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.18)
    fig.savefig(output / 'forecast.png', dpi=160)
    plt.close(fig)
