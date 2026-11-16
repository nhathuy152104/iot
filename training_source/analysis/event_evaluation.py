"""Observed-grid, slot-matched event metrics. No merging across lows or gaps."""
import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=5)
POLICY = {
    'status': 'EXPLORATORY_RETROSPECTIVE_NOT_DEPLOYMENT_APPROVAL',
    'threshold_cm': 10,
    'event': 'one or more consecutive eligible future-proc labels >=10cm; exact 5min decision grid; no merge across low or missing slots',
    'censoring': 'left/right censored unless immediately adjacent 5min eligible slot is observed low; separate all/onset-observed/fully-observed counts',
    'alarm': 'consecutive predictions >=10cm on exact 5min decision grid; immediate clear on low; missing slot terminates episode, never carry-forward',
    'confirmation': 'primary one high bin; fixed sensitivity two consecutive highs, activates at SECOND high decision, not retroactively',
    'matching': 'alarm active on an eligible decision whose corresponding future-proc label belongs to event; many-to-many slot overlap, not temporal proximity',
    'false_episode': 'no high label in any of its active forecast slots; mixed episodes separately expose false windows; not verified false floods',
    'lead': 'first matched true-positive decision to interval (previous observed-low target source time, first observed-high target source time]; only onset-observed detected events; NOT alarm episode start',
    'lead_limits': 'processed-archive observation clock only; no source-filter/transport availability or ESP32 transfer claims; no fixed ten-minute lead',
    'coverage': 'eligible decision slots * 5min / 1440, pooled across stations; observed station-day equivalents, not calendar station-days',
    'false_episode_rate': 'all observed false episodes / observed station-day equivalents *100; includes fragments censored by gaps; observable-start false counts also reported',
    'short_events': 'retain singleton highs; report >=2-bin event recall as descriptive subset, never relabel singleton highs as false',
    'selection': 'no parameter search, refit, threshold tuning, station removal or model replacement',
}


def runs(active, index):
    """Inclusive positional intervals, resetting on inactive rows or grid gaps."""
    result, start = [], None
    for i, value in enumerate(active):
        if start is not None and (not value or index[i] - index[i - 1] != STEP):
            result.append((start, i - 1))
            start = None
        if value and start is None:
            start = i
    if start is not None:
        result.append((start, len(active) - 1))
    return result


def ratio(a, b):
    return a / b if b else None


def evaluate_station(frame, confirm_bins=1):
    if frame.empty or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('Nonempty, unique, ordered single-station frame required')
    if frame.index.tz is None or confirm_bins not in (1, 2):
        raise ValueError('UTC-aware clock and declared confirmation policy required')
    if not np.isfinite(frame[['target', 'prediction']]).all().all():
        raise ValueError('Finite target/prediction required; no imputation')
    if (frame['target_time'].isna().any() or not frame.target_time.is_monotonic_increasing):
        raise ValueError('Ordered actual target source timestamps required')
    index = frame.index
    wet = frame.target.to_numpy() >= 10
    high = frame.prediction.to_numpy() >= 10
    active = np.zeros(len(frame), dtype=bool)
    for start, end in runs(high, index):
        active[start + confirm_bins - 1:end + 1] = True
    event_id = np.full(len(frame), -1, dtype=int)
    alarm_id = np.full(len(frame), -1, dtype=int)
    event_runs, alarm_runs = runs(wet, index), runs(active, index)
    for number, (start, end) in enumerate(event_runs):
        event_id[start:end + 1] = number
    for number, (start, end) in enumerate(alarm_runs):
        alarm_id[start:end + 1] = number
    events, alarms = [], []
    for number, (start, end) in enumerate(event_runs):
        left = start == 0 or index[start] - index[start - 1] != STEP
        right = end == len(frame) - 1 or index[end + 1] - index[end] != STEP
        hits = np.flatnonzero(active[start:end + 1]) + start
        row = {'event_id': number, 'start_decision_utc': str(index[start]),
            'end_decision_utc': str(index[end]), 'first_high_target_source_utc': str(frame.target_time.iloc[start]),
            'high_windows': end - start + 1, 'left_censored': left, 'right_censored': right,
            'detected': bool(len(hits)), 'matching_alarm_episodes': len(set(alarm_id[hits])),
            'peak_label_cm': float(frame.target.iloc[start:end + 1].max()),
            'first_match_decision_utc': None, 'lead_lower_minutes': None, 'lead_upper_minutes': None}
        if len(hits):
            decision = index[hits[0]]
            row['first_match_decision_utc'] = str(decision)
            if not left:
                row['lead_lower_minutes'] = (frame.target_time.iloc[start - 1] - decision).total_seconds() / 60
                row['lead_upper_minutes'] = (frame.target_time.iloc[start] - decision).total_seconds() / 60
        events.append(row)
    for number, (start, end) in enumerate(alarm_runs):
        matches = set(event_id[start:end + 1]) - {-1}
        false_windows = int((~wet[start:end + 1]).sum())
        alarms.append({'alarm_id': number, 'start_decision_utc': str(index[start]),
            'end_decision_utc': str(index[end]), 'active_windows': end - start + 1,
            'matching_events': len(matches), 'false_episode': not bool(matches),
            'false_windows': false_windows, 'mixed_episode': bool(matches) and false_windows > 0,
            'left_censored': start == 0 or index[start] - index[start - 1] != STEP,
            'right_censored': end == len(frame) - 1 or index[end + 1] - index[end] != STEP})
    summary = summarize(events, alarms, len(frame))
    summary['false_alarm_windows'] = int((active & ~wet).sum())
    summary['missed_wet_windows'] = int((~active & wet).sum())
    summary['wet_windows'] = int(wet.sum())
    assert sum(r['high_windows'] for r in events) == summary['wet_windows']
    assert sum(r['false_windows'] for r in alarms) == summary['false_alarm_windows']
    return events, alarms, summary


def summarize(events, alarms, windows):
    detected = sum(e['detected'] for e in events)
    onset = [e for e in events if not e['left_censored']]
    full = [e for e in onset if not e['right_censored']]
    sustained = [e for e in events if e['high_windows'] >= 2]
    lead = [e for e in onset if e['detected']]
    false = sum(a['false_episode'] for a in alarms)
    days = windows * 5 / 1440
    return {'windows': windows, 'observed_station_day_equivalents': days,
        'events': len(events), 'detected_events': detected, 'missed_events': len(events) - detected,
        'event_recall': ratio(detected, len(events)),
        'onset_observed_events': len(onset), 'onset_observed_detected': sum(e['detected'] for e in onset),
        'fully_observed_events': len(full), 'fully_observed_detected': sum(e['detected'] for e in full),
        'sustained_ge2_events': len(sustained), 'sustained_ge2_detected': sum(e['detected'] for e in sustained),
        'alarm_episodes': len(alarms), 'false_alarm_episodes': false,
        'false_episodes_per_100_observed_station_days': 100 * false / days,
        'false_episode_observable_starts': sum(a['false_episode'] and not a['left_censored'] for a in alarms),
        'mixed_alarm_episodes': sum(a['mixed_episode'] for a in alarms),
        'fragmented_detected_events': sum(e['matching_alarm_episodes'] > 1 for e in events),
        'onset_observed_detected_with_lead': len(lead),
        'certain_early_events': sum(e['lead_lower_minutes'] > 0 for e in lead),
        'possibly_early_events': sum(e['lead_upper_minutes'] > 0 for e in lead),
        'certain_late_events': sum(e['lead_upper_minutes'] <= 0 for e in lead),
        'lead_lower_median_minutes_detected_only': float(np.median([e['lead_lower_minutes'] for e in lead])) if lead else None,
        'lead_upper_median_minutes_detected_only': float(np.median([e['lead_upper_minutes'] for e in lead])) if lead else None}
