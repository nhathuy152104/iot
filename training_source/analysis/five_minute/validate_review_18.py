"""Validate every diagnostic mask against the unmodified original train file."""
import pandas as pd

from pipeline import OUT as BASE, save_json, split_masks
from review_18 import OUT, describe_windows, proposed_role


def main():
    report=pd.read_csv(OUT/'station_review.csv')
    original=pd.read_csv(BASE/'station_selection.csv')
    expected=set(original.loc[original.category.eq('review_quality'),'deployment_id'])
    assert set(report.deployment_id)==expected
    assert report.deployment_id.is_unique
    total=0
    low_total=0
    for r in report.to_dict('records'):
        station=r['deployment_id']
        mask=pd.read_parquet(OUT/'train_review_masks'/f'{station}.parquet')
        windows=pd.read_parquet(BASE/'provisional_windows'/f'{station}_train.parquet')
        assert mask.index.equals(windows.index),station
        assert split_masks(mask.index)['train'].all(),station
        assert not mask.isna().any().any(),station
        assert all(str(t)=='bool' for t in mask.dtypes),station
        flag_columns=[c for c in mask if c!='proposed_low_flag_subset']
        assert mask['proposed_low_flag_subset'].equals(~mask[flag_columns].any(axis=1)),station
        low=windows.loc[mask.proposed_low_flag_subset]
        stats=describe_windows(low)
        for k,v in stats.items():
            assert v==r[f'low_flag_{k}'],(station,k)
        assert proposed_role(stats)==r['proposal'],station
        assert len(windows)==r['original_train_windows'],station
        assert len(low)+r['flagged_for_quarantine_windows']==len(windows),station
        total+=len(windows)
        low_total+=len(low)
    result={'status':'PASS','stations_checked':len(report),'train_windows_checked':total,
            'low_flag_windows':low_total,'quarantined_for_review_windows':total-low_total,
            'limits':'Structural validation only. Low-flag does not mean true flood labels; offline flags must not be model inputs.'}
    save_json(OUT/'validation.json',result)
    print(result)


if __name__=='__main__':
    main()
