"""Local, read-only audit of FloodNet CSVs; outputs derived summaries, not raw data.

Candidate episodes are contiguous threshold runs, not verified floods.
Only standard-library Python is required. Run: python audit_floodnet.py
"""
import calendar
import collections
import csv
import datetime as dt
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
THRESHOLDS = (10, 100, 200)  # mm: sensitivity, research reference, higher depth


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()


def write_csv(name, rows):
    if rows:
        with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    files = sorted(ROOT.glob('*.csv'), key=lambda p: (
        int(p.stem.split('-')[-1]),
        list(calendar.month_abbr).index(p.stem.split('-')[-2].title())))
    stations = {}
    month_rows = []
    episodes = []
    file_rows = []

    def close(s, threshold, reason):
        e = s['active'].pop(threshold, None)
        if e:
            e['end_reason'] = reason
            e['duration_min'] = round((e['end'] - e['start']) / 60, 2)
            e['sustained'] = e['samples'] >= 3 and e['end'] - e['start'] >= 120
            episodes.append(e)

    for path in files:
        groups = collections.defaultdict(list)
        metadata = {}
        invalid_time = 0
        input_disorder = 0
        previous = {}
        count = 0
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                count += 1
                key = r['deployment_id']
                try:
                    t = timestamp(r['time'])
                except (ValueError, TypeError):
                    invalid_time += 1
                    continue
                if key in previous and t < previous[key]:
                    input_disorder += 1
                previous[key] = t
                groups[key].append((t, number(r['depth_proc_mm']),
                                    number(r['depth_raw_mm']), number(r['depth_filt_mm'])))
                metadata[key] = (r.get('deploy_type', '').strip().lower(), r.get('sensor_display_name', '').strip())
        file_rows.append({'file': path.name, 'bytes': path.stat().st_size,
                          'rows': count, 'invalid_time': invalid_time,
                          'input_time_reversals': input_disorder, 'deployments': len(groups)})
        for key, records in groups.items():
            records.sort(key=lambda x: x[0])
            intervals = [b[0] - a[0] for a, b in zip(records, records[1:]) if b[0] > a[0]]
            cadence = statistics.median(intervals) if intervals else 60
            gap_limit = max(180, 3 * cadence)
            typ, name = metadata[key]
            s = stations.setdefault(key, {'type': typ, 'name': name, 'counts': collections.Counter(),
                                         'active': {}, 'last': None, 'max_mm': None,
                                         'first': records[0][0], 'last_time': records[-1][0],
                                         'cadences': [], 'types': set()})
            s['types'].add(typ)
            s['cadences'].append(cadence)
            c = collections.Counter()
            for t, depth, raw, filt in records:
                c['rows'] += 1
                prior = s['last']
                if prior and t == prior[0]:
                    c['duplicate_timestamps'] += 1
                    if (depth, raw, filt) != prior[1:]:
                        c['conflicting_duplicate_timestamps'] += 1
                    # Duplicate timestamps are counted but not used as extra observations.
                    continue
                gap = t - prior[0] if prior else None
                if gap is not None:
                    if gap <= 0:
                        c['nonpositive_intervals_after_sort'] += 1
                    if gap > 600:
                        c['gaps_gt_10min'] += 1
                    if gap > gap_limit:
                        c['gaps_gt_3cadence_or_3min'] += 1
                        for h in list(s['active']):
                            close(s, h, 'gap')
                for field, val in [('proc', depth), ('raw', raw), ('filt', filt)]:
                    if val is None:
                        c['missing_' + field] += 1
                    elif val < 0:
                        c['negative_' + field] += 1
                if raw is not None and depth is not None:
                    if abs(raw - depth) >= 100:
                        c['raw_proc_difference_ge_100mm'] += 1
                    if raw >= 100 and depth < 10:
                        c['raw_ge_100mm_proc_lt_10mm'] += 1
                if depth is not None:
                    s['max_mm'] = max(s['max_mm'], depth) if s['max_mm'] is not None else depth
                    if prior and prior[1] is not None and gap and 0 < gap <= gap_limit:
                        if abs(depth - prior[1]) / (gap / 60) > 254:
                            c['proc_abs_gradient_gt_254mm_per_min'] += 1
                for h in THRESHOLDS:
                    if depth is not None and depth >= h:
                        c['samples_ge_' + str(h) + 'mm'] += 1
                        if h not in s['active']:
                            s['active'][h] = {'deployment_id': key, 'type': typ,
                                'threshold_mm': h, 'start': t, 'end': t,
                                'samples': 0, 'peak_mm': depth,
                                'left_censored': prior is None or gap > gap_limit or prior[1] is None}
                        e = s['active'][h]
                        e['end'] = t
                        e['samples'] += 1
                        e['peak_mm'] = max(e['peak_mm'], depth)
                    else:
                        close(s, h, 'missing' if depth is None else 'below_threshold')
                s['last'] = (t, depth, raw, filt)
                s['last_time'] = max(s['last_time'], t)
            s['counts'].update(c)
            month_rows.append({'file': path.name, 'deployment_id': key, 'type': typ,
                               'median_interval_sec': round(cadence, 2),
                               'first_utc': iso(records[0][0]), 'last_utc': iso(records[-1][0]),
                               **dict(c)})
        print(f'Processed {path.name}: {count:,} rows, {len(groups)} deployments', flush=True)
    for s in stations.values():
        for h in list(s['active']):
            close(s, h, 'end_of_dataset')
    # Stable columns, including counters absent in some groups.
    keys = sorted({k for r in month_rows for k in r})
    month_rows = [{k: r.get(k, 0) for k in keys} for r in month_rows]
    station_rows = []
    counter_keys = sorted({k for s in stations.values() for k in s['counts']})
    for key, s in stations.items():
        es = [e for e in episodes if e['deployment_id'] == key]
        station_rows.append({'deployment_id': key, 'type': '|'.join(sorted(s['types'])),
            'name': s['name'], 'first_utc': iso(s['first']), 'last_utc': iso(s['last_time']),
            'median_of_monthly_cadences_sec': round(statistics.median(s['cadences']), 2),
            'max_proc_cm': s['max_mm'] / 10 if s['max_mm'] is not None else None,
            **{k: s['counts'].get(k, 0) for k in counter_keys},
            **{f'sustained_episodes_ge_{h}mm': sum(e['threshold_mm'] == h and e['sustained'] for e in es)
               for h in THRESHOLDS}})
    # Temporal clusters are only a grouping aid, not meteorologically verified storms.
    primary = sorted([e for e in episodes if e['type'] == 'pluvial' and
                      e['threshold_mm'] == 100 and e['sustained']], key=lambda e: e['start'])
    clusters = []
    for e in primary:
        if not clusters or e['start'] > clusters[-1]['end'] + 6 * 3600:
            clusters.append({'start': e['start'], 'end': e['end'], 'episodes': 0, 'stations': set()})
        g = clusters[-1]
        g['end'] = max(g['end'], e['end'])
        g['episodes'] += 1
        g['stations'].add(e['deployment_id'])
    cluster_rows = [{'start_utc': iso(g['start']), 'end_utc': iso(g['end']),
                     'station_episodes': g['episodes'], 'deployments': len(g['stations'])} for g in clusters]
    totals = collections.Counter()
    for s in stations.values():
        totals.update(s['counts'])
    summary = {'files': len(files), 'bytes': sum(r['bytes'] for r in file_rows),
        'input_rows': sum(r['rows'] for r in file_rows), 'invalid_time': sum(r['invalid_time'] for r in file_rows),
        'input_time_reversals': sum(r['input_time_reversals'] for r in file_rows),
        'deployments': len(stations), 'types': dict(collections.Counter(s['type'] for s in stations.values())),
        'totals': dict(totals), 'candidate_episodes': len(episodes),
        'sustained_counts': {typ: {str(h): sum(e['type'] == typ and e['threshold_mm'] == h and e['sustained']
                       for e in episodes) for h in THRESHOLDS} for typ in sorted({s['type'] for s in stations.values()})},
        'pluvial_100mm_temporal_clusters_6h': len(clusters),
        'method': {'thresholds_mm': THRESHOLDS, 'sustained': '>=3 observations and >=2 min first-to-last',
                   'split': 'below threshold, missing proc, or gap > max(180sec, 3*monthly median cadence)',
                   'duplicates': 'same deployment/timestamp excluded from episode sample counts; conflicts counted',
                   'clusters': 'pluvial sustained >=100mm runs grouped with <=6h gap; NOT verified independent storms',
                   'caveats': ['Processed depth only; causal filtering unverified.',
                               'No weather or external flood verification.',
                               'Thresholds are research definitions, not safe travel limits.',
                               'Episode duration excludes unobserved time between samples.',
                               'No raw records copied into outputs.']}}
    write_csv('files.csv', file_rows)
    write_csv('station_month_quality.csv', month_rows)
    write_csv('stations.csv', station_rows)
    write_csv('candidate_episodes.csv', [{**e, 'start': iso(e['start']), 'end': iso(e['end'])} for e in episodes])
    write_csv('temporal_clusters.csv', cluster_rows)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
