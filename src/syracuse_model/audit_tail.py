"""Fetch public daily aggregates to audit the sparse December 2023 source tail."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from syracuse_crime import load_records

SERVICE = ('https://services6.arcgis.com/bdPqSfflsdgFRVVM/arcgis/rest/services/'
           'crime_data_1_auto_update/FeatureServer/0/query')
METADATA = ('https://www.arcgis.com/sharing/rest/content/items/'
            '94bc33f1a2c646b995d1ab356edf0730/info/metadata/metadata.xml?format=default&output=html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('reports/smoothing/source-audit.json'))
    args = parser.parse_args()
    params = {'f': 'json', 'where': "DATEEND >= TIMESTAMP '2023-12-01 00:00:00' "
              "AND DATEEND < TIMESTAMP '2024-01-01 00:00:00'",
              'groupByFieldsForStatistics': 'DATEEND,CODE_DEFINED',
              'outStatistics': json.dumps([{'statisticType': 'count',
                  'onStatisticField': 'ObjectId', 'outStatisticFieldName': 'records'}]),
              'orderByFields': 'DATEEND,CODE_DEFINED'}
    query = SERVICE + '?' + urlencode(params)
    with urlopen(query, timeout=60) as response:
        result = json.load(response)
    if 'error' in result or result.get('exceededTransferLimit'):
        raise ValueError('Source query failed or was truncated')
    rows = [{'date': str(pd.Timestamp(row['attributes']['DATEEND'], unit='ms').date()),
             'category': row['attributes']['CODE_DEFINED'],
             'records': row['attributes']['records']} for row in result['features']]
    records, audit = load_records(args.database)
    local = records.loc[records.date.between('2023-12-01', '2023-12-31') &
                        records.category.eq('LARCENY')].groupby('date').size()
    remote = {row['date']: row['records'] for row in rows if row['category'] == 'LARCENY'}
    comparison = [{'date': str(day.date()), 'local_larceny': int(local.get(day, 0)),
                   'remote_larceny': remote.get(str(day.date()), 0)}
                  for day in pd.date_range('2023-12-01', '2023-12-31')]
    artifact = {'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                'database_sha256': audit['database_sha256'], 'query_url': query,
                'metadata_url': METADATA,
                'date_definition': 'City 2023 metadata defines DATEEND as date reported; '
                                   'historical publication/revision timing is unavailable.',
                'all_december_larceny_daily_counts_match': all(
                    row['local_larceny'] == row['remote_larceny'] for row in comparison),
                'last_remote_record_date_in_december': max(row['date'] for row in rows),
                'last_remote_larceny_date_in_december': max(remote),
                'final_week_local_larceny': sum(row['local_larceny'] for row in comparison
                                                if row['date'] >= '2023-12-25'),
                'final_week_remote_larceny': sum(row['remote_larceny'] for row in comparison
                                                 if row['date'] >= '2023-12-25'),
                'conclusion': 'Agreement alone cannot establish coverage; missing collection versus '
                              'zero records requires independent verification.',
                'comparison': comparison, 'remote_daily_aggregates': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + '\n')
    print(json.dumps({k: v for k, v in artifact.items()
                      if k not in ('query_url', 'comparison', 'remote_daily_aggregates')}, indent=2))


if __name__ == '__main__':
    main()
