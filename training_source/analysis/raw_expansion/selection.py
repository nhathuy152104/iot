"""Fixed coverage rules; flags are diagnostic, not proof of flood truth."""
import numpy as np

from five_minute.pipeline import temporal_groups, split_masks
from raw_branch.pipeline import make_raw_grid, make_raw_features, eligibility_reasons


CANDIDATES = (
    'daily_gentle_beetle', 'evenly_divine_dingo', 'overly_heroic_squid',
    'simply_pet_joey', 'tired_red_monkey', 'weekly_fair_dove', 'weekly_poetic_guinea',
)
RULES = {
    'windows': 1000,
    'decision_days': 30,
    'wet_ge_10cm': 20,
    'nonwet_lt_10cm': 1000,
    'wet_days': 3,
    'wet_temporal_groups_6h': 3,
}


def selection_failures(audit, rules=RULES):
    return [key for key, minimum in rules.items() if audit[key] < minimum]


def build_candidate_train(station, source, features, label):
    grid = make_raw_grid(source)
    frame = make_raw_features(grid, station, label)
    reason = eligibility_reasons(frame, label, features)
    train_mask = split_masks(frame.index)['train']
    windows = frame.loc[train_mask & reason.eq('kept')].copy()
    assert np.isfinite(windows[features]).all().all()
    wet = windows[label].ge(10)
    wet_times = windows.index[wet]
    bins = grid.loc[train_mask]
    proc_now = grid.depth_proc_mm.reindex(windows.index) / 10
    comparable = proc_now.notna() & np.isfinite(proc_now)
    mismatch = (windows.depth_lag_0_cm - proc_now).abs().ge(10)
    audit = {
        'station': station, 'windows': len(windows),
        'decision_days': int(windows.index.normalize().nunique()),
        'wet_ge_10cm': int(wet.sum()), 'nonwet_lt_10cm': int((~wet).sum()),
        'wet_days': int(wet_times.normalize().nunique()),
        'wet_temporal_groups_6h': temporal_groups(wet_times),
        'raw_negative_history_fraction': float(windows.audit_negative_raw_history.mean()) if len(windows) else None,
        'sparse_history_fraction': float(windows.audit_sparse_raw_history.mean()) if len(windows) else None,
        'raw_proc_current_disagreement_ge_10cm_fraction': float(mismatch[comparable].mean()) if comparable.any() else None,
        'raw_proc_current_comparable_windows': int(comparable.sum()),
        'decision_bins_before_eligibility': int(train_mask.sum()),
        'missing_raw_bins': int(bins.raw_median_cm.isna().sum()),
        'negative_raw_rows_floored': int(bins.audit_negative_raw_rows.sum()),
        'raw_ge_100cm_rows_retained': int(bins.audit_high_raw_ge_100cm_rows.sum()),
        'target_max_cm': float(windows[label].max()) if len(windows) else None,
        'raw_current_max_cm': float(windows.depth_lag_0_cm.max()) if len(windows) else None,
        'first_train_decision_utc': str(windows.index.min()) if len(windows) else None,
        'last_train_decision_utc': str(windows.index.max()) if len(windows) else None,
    }
    audit['failed_criteria'] = selection_failures(audit)
    audit['selected_for_exploratory_expansion'] = not audit['failed_criteria']
    audit['quality_status'] = 'UNVERIFIED; all eligible windows retained, including ambiguous high/noisy cases'
    reasons = [{'station': station, 'reason': name, 'windows': int(count)}
               for name, count in reason.loc[train_mask].value_counts().items()]
    return windows, audit, reasons


def select_overall(rows):
    """Exact ties prefer persistence, then trend, then the frozen three-point arm."""
    from delta_experiment import METRIC
    usable = [r for r in rows if r[METRIC] is not None and np.isfinite(r[METRIC])]
    if not usable:
        raise ValueError('Validation must contain both target groups')
    def priority(name):
        if name == 'persistence':
            return 0
        if name == 'linear_trend_5min':
            return 1
        return 2 if name.startswith('three_') else 3
    return min(usable, key=lambda r: (r[METRIC], r['mae_cm'], priority(r['model']), r['model']))['model']
