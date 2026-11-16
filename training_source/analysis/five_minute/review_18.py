"""Train-only diagnostic review. Preserve original selection and observations.

Shape rules flag ambiguity, NOT known sensor errors. Proposed subsets are
retrospective sensitivity cohorts, not validated labels or production filters.
"""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import (OUT as BASE, ROOT, DEPTHS, TRAIN_END, EMBARGO,
                      cache_sources, make_grid, split_masks, temporal_groups, save_json)

OUT = BASE / 'review_18'


def runs(mask):
    """Contiguous True positions on a regular grid; no bridging missing bins."""
    values = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, values, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def episode_review(grid):
    depth = grid['depth_proc_mm']
    wet = depth.ge(100).to_numpy()
    suspect = pd.Series(False, index=grid.index)
    rows = []
    for start, end in runs(wet):
        part = depth.iloc[start:end]
        before = depth.iloc[start - 1] if start else np.nan
        after = depth.iloc[end] if end < len(depth) else np.nan
        bracketed = bool(np.isfinite(before) and np.isfinite(after))
        low_bracket = bracketed and before < 10 and after < 10
        blip = bool(end - start <= 2 and low_bracket)
        nearconstant = float((part.sub(part.median()).abs() <= 10).mean())
        step_edges = bracketed and abs(part.iloc[0] - before) >= 100 and abs(part.iloc[-1] - after) >= 100
        box = bool(end - start >= 3 and nearconstant >= .8 and step_edges)
        # A flag is an ambiguity for review; genuine flash floods can be abrupt too.
        flagged = blip or box
        if flagged:
            suspect.iloc[start:end] = True
        rows.append({'start_utc': str(part.index[0]), 'end_utc': str(part.index[-1]),
                     'bins': len(part), 'duration_first_to_last_min': (len(part)-1)*5,
                     'peak_cm': float(part.max()/10), 'nearconstant_fraction': nearconstant,
                     'bracketed_by_valid_bins': bracketed,
                     'possible_short_pulse': blip, 'possible_box': box,
                     'shape_ambiguous': flagged})
    return rows, suspect


def subset_mask(windows, grid, shape):
    index = windows.index
    target_index = index + pd.Timedelta(minutes=10)
    target = grid.reindex(target_index)
    target.index = index
    target_diff = (target['depth_raw_mm'] - target['depth_proc_mm']).abs().ge(100)
    target_missing_raw = target['depth_raw_mm'].isna()
    history_shape = shape.rolling(3, min_periods=1).max().astype(bool).reindex(index, fill_value=False)
    target_shape = shape.reindex(target_index, fill_value=False)
    target_shape.index = index
    flags = pd.DataFrame(index=index)
    flags['history_raw_proc_disagreement'] = windows['review_raw_proc_diff_ge_10cm_history']
    flags['history_proc_jump'] = windows['review_proc_jump_history']
    flags['history_raw_missing'] = windows['review_raw_missing_history']
    flags['offline_target_raw_proc_disagreement'] = target_diff
    flags['offline_target_raw_missing'] = target_missing_raw
    flags['offline_history_shape_ambiguous'] = history_shape
    flags['offline_target_shape_ambiguous'] = target_shape
    flags['proposed_low_flag_subset'] = ~flags.any(axis=1)
    return flags


def describe_windows(windows):
    wet = windows['target_depth_bin_t_plus_10min_cm'].ge(10)
    times = windows.index[wet]
    return {'windows': len(windows), 'decision_days': int(windows.index.normalize().nunique()),
            'wet_windows_ge_10cm': int(wet.sum()), 'nonwet_windows_lt_10cm': int((~wet).sum()),
            'wet_days': int(times.normalize().nunique()), 'wet_groups_6h': temporal_groups(times)}


def proposed_role(stats):
    if (stats['windows'] >= 1000 and stats['decision_days'] >= 30 and
        stats['wet_windows_ge_10cm'] >= 20 and stats['nonwet_windows_lt_10cm'] >= 1000 and
        stats['wet_days'] >= 3 and stats['wet_groups_6h'] >= 3):
        return 'candidate_event_segments'
    if stats['windows'] >= 1000 and stats['decision_days'] >= 30:
        return 'background_or_limited_wet'
    return 'insufficient_low_flag_coverage'


def plot_cases(station, frame, grid, segments, shape):
    train = grid['depth_proc_mm']
    centers = []
    if train.notna().any():
        centers.append(('processed_peak', train.idxmax()))
    diff = (grid['depth_raw_mm'] - grid['depth_proc_mm']).abs().dropna()
    if len(diff):
        centers.append(('raw_proc_disagreement', diff.idxmax()))
    smooth = []
    for segment in segments:
        if segment['shape_ambiguous'] or segment['bins'] < 6:
            continue
        a, b = pd.Timestamp(segment['start_utc']), pd.Timestamp(segment['end_utc'])
        part = train.loc[a:b]
        # A descriptive shape score for choosing a REVIEW image, not selecting labels.
        changes = part.diff().dropna()
        if changes.gt(0).any() and changes.lt(0).any():
            smooth.append((len(part), part.idxmax()))
    if smooth:
        centers.append(('longer_rise_fall_candidate', max(smooth, key=lambda x:x[0])[1]))
    ambiguous = train.loc[shape]
    if len(ambiguous):
        centers.append(('shape_ambiguity', ambiguous.idxmax()))
    rows = []
    folder = OUT / 'plots'
    folder.mkdir(exist_ok=True)
    for kind, center in centers:
        start = center - pd.Timedelta(hours=3)
        end = min(center + pd.Timedelta(hours=3), TRAIN_END - EMBARGO)
        view = frame.loc[(frame.index >= start) & (frame.index < end)]
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout='constrained')
        for col, color in zip(DEPTHS, ['#777777','#e69f00','#0072b2']):
            axes[0].scatter(view.index, view[col]/10, s=8, label=col, color=color, alpha=.65)
        axes[0].axhline(10, color='red', linestyle='--', linewidth=.7)
        axes[0].set_ylabel('Depth (cm)')
        axes[0].legend(fontsize=8)
        axes[1].scatter(view.index, (view['depth_raw_mm']-view['depth_proc_mm'])/10, s=8, color='#8e44ad')
        axes[1].set_ylabel('raw - proc (cm)')
        axes[1].set_xlabel('Actual UTC source times; no interpolation or gap bridging')
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle(f'{station} | TRAIN | {kind} | {center:%Y-%m-%d %H:%M} UTC\nA review flag is NOT confirmed noise or a verified flood')
        path = folder / f'{station}_{kind}.png'
        fig.savefig(path, dpi=115)
        plt.close(fig)
        rows.append({'deployment_id':station, 'kind':kind, 'center_utc':str(center),
                     'path':str(path.relative_to(OUT)), 'source_points':len(view),
                     'proc_peak_cm':float(view['depth_proc_mm'].max()/10),
                     'max_abs_raw_proc_difference_cm':float((view['depth_raw_mm']-view['depth_proc_mm']).abs().max()/10)})
    # Daily plot separates total absent bins from null readings on rows that exist.
    daily = pd.DataFrame({'valid_proc_fraction':grid['depth_proc_mm'].notna().resample('1D').mean(),
                          'proc_peak_cm':grid['depth_proc_mm'].resample('1D').max()/10})
    fig, ax = plt.subplots(2,1,figsize=(12,5),sharex=True,layout='constrained')
    ax[0].plot(daily.index,daily['valid_proc_fraction']*100,linewidth=.8)
    ax[0].set_ylabel('Valid 5-min bins (%)')
    ax[1].scatter(daily.index,daily['proc_peak_cm'],s=7)
    ax[1].set_ylabel('Daily proc max (cm)')
    ax[1].set_xlabel('UTC; TRAIN only')
    fig.suptitle(f'{station} | TRAIN coverage and processed-depth peaks')
    path=folder/f'{station}_coverage.png'
    fig.savefig(path,dpi=115)
    plt.close(fig)
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    selection = pd.read_csv(BASE/'station_selection.csv')
    review = selection.loc[selection.category.eq('review_quality')]
    all_pluvial=set(selection.deployment_id)
    _, folders=cache_sources(all_pluvial)
    records, episode_rows, continuous_rows, cases = [],[],[],[]
    (OUT/'train_review_masks').mkdir(exist_ok=True)
    for station in review.deployment_id:
        parts=[pd.read_parquet(folder/f'{station}.parquet') for folder in folders if (folder/f'{station}.parquet').exists()]
        frame=pd.concat(parts).sort_index(kind='stable')
        frame=frame.loc[~frame.index.duplicated(keep='first')]
        frame=frame.loc[frame.index < TRAIN_END-EMBARGO]
        grid=make_grid(frame)
        # Include the training labels' bins too, not only decision-eligible bins.
        # Otherwise the final two valid train labels would look artificially missing.
        grid=grid.loc[grid.index < TRAIN_END-EMBARGO]
        source=frame.loc[frame.index<=grid.index.max()]
        episodes, shape=episode_review(grid)
        episode_rows.extend([{'deployment_id':station,**r} for r in episodes])
        valid=grid.depth_proc_mm.notna() & grid.depth_proc_mm.ge(0)
        segments=[]
        for a,b in runs(valid):
            if b-a<3:
                continue
            part=grid.iloc[a:b]
            segment={'deployment_id':station,'start_utc':str(part.index[0]),'end_utc':str(part.index[-1]),
                     'valid_bins':len(part),'duration_first_to_last_hours':(len(part)-1)/12,
                     'wet_bins_ge_10cm':int(part.depth_proc_mm.ge(100).sum())}
            segments.append(segment)
        continuous_rows.extend(segments)
        windows=pd.read_parquet(BASE/'provisional_windows'/f'{station}_train.parquet')
        flags=subset_mask(windows,grid,shape)
        flags.to_parquet(OUT/'train_review_masks'/f'{station}.parquet')
        low=windows.loc[flags.proposed_low_flag_subset]
        stats=describe_windows(low)
        role=proposed_role(stats)
        original=review.loc[review.deployment_id.eq(station)].iloc[0]
        triggers=[]
        if original.train_missing_grid_fraction>.3:triggers.append('missing_grid_gt_30pct')
        if original.train_history_raw_proc_disagreement_fraction>.1:triggers.append('history_disagreement_gt_10pct')
        if original.train_history_proc_jump_fraction>.005:triggers.append('history_jump_gt_0.5pct')
        record={'deployment_id':station,'name':original['name'],'original_triggers':'|'.join(triggers),
                'proposal':role,'train_source_rows':len(source),
                'train_missing_source_proc_fraction':float(source.depth_proc_mm.isna().mean()),
                'train_missing_grid_proc_fraction':float(grid.depth_proc_mm.isna().mean()),
                'original_selection_train_missing_grid_fraction':float(original.train_missing_grid_fraction),
                'continuous_segments_ge_3bins':len(segments),
                'longest_continuous_hours':max([s['duration_first_to_last_hours'] for s in segments],default=0),
                'threshold_runs_ge_10cm':len(episodes),
                'possible_short_pulse_runs':sum(e['possible_short_pulse'] for e in episodes),
                'possible_box_runs':sum(e['possible_box'] for e in episodes),
                'original_train_windows':len(windows),
                'flagged_for_quarantine_windows':int((~flags.proposed_low_flag_subset).sum()),
                **{f'low_flag_{k}':v for k,v in stats.items()},
                **{f'count_{col}':int(flags[col].sum()) for col in flags if col!='proposed_low_flag_subset'}}
        records.append(record)
        cases.extend(plot_cases(station,source,grid,episodes,shape))
        print(f'{station}: {role}; low_flag={len(low):,}; wet={stats["wet_windows_ge_10cm"]:,}; days={stats["wet_days"]}',flush=True)
    pd.DataFrame(records).to_csv(OUT/'station_review.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(episode_rows).to_csv(OUT/'threshold_run_review.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(continuous_rows).to_csv(OUT/'continuous_segments.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(cases).to_csv(OUT/'plot_cases.csv',index=False,encoding='utf-8-sig')
    manifest={'status':'Proposed train-only diagnostic cohorts; not confirmed clean data',
              'stations_reviewed':len(records),'plots':len(cases)+len(records),
              'roles':pd.Series([r['proposal'] for r in records]).value_counts().to_dict(),
              'candidate_event_segments':[r['deployment_id'] for r in records if r['proposal']=='candidate_event_segments'],
              'original_selection_unchanged':True,'model_trained':False,
              'time_scope':'TRAIN support only; before 2023-01-01 minus 24h embargo',
              'shape_rules':{'wet_threshold_mm':100,'short_pulse':'<=2 bins bracketed immediately by valid proc<10mm',
                             'box':'>=3 bins; >=80% within10mm of median; immediate valid entry and exit jumps>=100mm'},
              'low_flag_subset':'No original history raw/proc>=100mm mismatch, jump flag or missing raw; no target raw missing/mismatch; no history/target ambiguous shape flag.',
              'important_limits':['No observations edited or removed; review masks only.',
                 'No weather, field reports, or independent depth labels were checked.',
                 'Shape flags can quarantine REAL flash floods; low-flag subset is only a sensitivity cohort.',
                 'Archive proc causality remains unverified.',
                 'Future-target and full-run morphology used ONLY for retrospective cohort annotation, NEVER features or a live filter.',
                 'Screened/quarantined windows must remain visible; evaluation only on low-flag data is selection biased.',
                 'No change to original station list, policy thresholds, validation or test.']}
    save_json(OUT/'review_manifest.json',manifest)
    write_report(manifest,records,cases)
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


def write_report(manifest,records,cases):
    lines=['# Đối soát 18 trạm cần xem xét — chỉ dùng train','',
           '**Không xác nhận nhãn ngập thật, không train model, không thay đổi sàng lọc gốc.**','',
           '## Phương pháp','',
           'Phân biệt mẫu null trên hàng có mặt với bin 5 phút không quan sát; tìm các đoạn liên tục không fill/nội suy. Kiểm tra các đoạn proc >=10 cm, cờ pulse/box có ngữ cảnh sát trước/sau và raw/proc. Tạo mask đề xuất theo cửa sổ, không sửa hoặc xóa dữ liệu.',
           'Low-flag không đồng nghĩa sạch: shape dùng cả cuối sự kiện/nhãn tương lai, chỉ để đối soát hồi cứu. Có thể gắn cờ nhầm ngập thật; không dùng làm bộ lọc online hay biến đầu vào, không báo cáo hiệu quả model chỉ trên phần dễ.',
           'Không tra cứu thời tiết hay hiện trường; không có cơ sở kết luận nguyên nhân vật cản, mất mạng hoặc mưa cho từng trường hợp. Bộ lọc proc causal chưa được xác minh.','',
           '## Kết quả tự động bổ sung','',
           f'Đối soát {manifest["stations_reviewed"]} trạm, xuất {manifest["plots"]} biểu đồ. Phân nhóm đề xuất: {manifest["roles"]}.',
           'candidate_event_segments: có thể xem xét thêm các đoạn cụ thể, không nhận nguyên trạm. background_or_limited_wet: giữ làm dữ liệu nền/ít ngập. insufficient_low_flag_coverage: chưa đủ theo quy tắc hiện tại. Không tự thay thế nhóm core cũ.','',
           '| Trạm | Lý do cờ cũ | Cửa sổ train gốc | Low-flag | Nhãn >=10 cm còn lại | Ngày nhãn cao | Nhóm 6h | Đề xuất |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    for r in records:
        lines.append(f'| {r["deployment_id"]} | {r["original_triggers"]} | {r["original_train_windows"]:,} | {r["low_flag_windows"]:,} | {r["low_flag_wet_windows_ge_10cm"]:,} | {r["low_flag_wet_days"]} | {r["low_flag_wet_groups_6h"]} | {r["proposal"]} |')
    notes_path=OUT/'visual_notes.json'
    if notes_path.exists():
        notes=json.loads(notes_path.read_text(encoding='utf-8'))
        lines+=['','## Đối soát trực quan','',notes['scope']]
        for note in notes['notes']:
            lines.append(f'- **{note["deployment_id"]}** ({note["status"]}): {note["observation_vi"]}')
        lines+=['','### Tám ca liên tục bổ sung đã xem','']
        for note in notes.get('additional_cases_viewed',[]):
            lines.append(f'- **{note["deployment_id"]}**: {note["observation_vi"]}')
        if notes.get('conclusion_vi'):
            lines+=['',f'**Kết luận trực quan:** {notes["conclusion_vi"]}']
    lines+=['','Ngưỡng đề xuất để xem xét đoạn có sự kiện: >=1.000 cửa sổ, >=30 ngày, >=20 nhãn cao, >=1.000 nhãn thấp, >=3 ngày và >=3 nhóm nhãn cao cách nhau 6 giờ. Các nhóm thời gian không phải trận mưa độc lập đã xác minh.',
            'Ngưỡng thiếu cấp trạm 30% ban đầu không áp lại cho đoạn: một trạm offline dài vẫn có thể có các cửa sổ đầy đủ. Đây là phân tích bổ sung ở cấp đoạn, không âm thầm nới chính sách gốc.','',
            'Tỷ lệ bin thiếu ở lượt đối soát tính trong span từ quan trắc train đầu đến cuối hiện diện, trước embargo; khác mẫu số của sàng lọc gốc khi có khoảng ngừng cuối train nhưng trạm lại xuất hiện ở 2023. Không so hai tỷ lệ để kết luận dữ liệu tự cải thiện. Tỷ lệ null trên hàng có mặt cũng không tính các bản tin hoàn toàn vắng.','',
            'Tên ảnh longer_rise_fall_candidate là heuristic chọn đoạn dài có cả thay đổi tăng và giảm, có thể chỉ là rung nhẹ ở một mức cao. Không dùng tên ảnh làm nhãn một đợt nước dâng/rút đã xác minh.','',
            '## File','',
            '- `station_review.csv`: lý do gắn cờ, độ thiếu nguồn/lưới, đoạn liên tục và số cửa sổ theo đề xuất.',
            '- `continuous_segments.csv`: các đoạn liên tục với ít nhất 3 bin; chưa tự bảo đảm đủ mọi nhãn tương lai.',
            '- `threshold_run_review.csv`: đoạn vượt 10 cm và cờ hình thái; các mức ngưỡng là quy ước nghiên cứu, không phải mức an toàn đi đường.',
            '- `train_review_masks/`: mask theo thời điểm train, gồm cờ OFFLINE; không phải feature/model artifact. Giữ nguyên cửa sổ gốc.',
            '- `index.html`, `plots/`: biểu đồ raw/filt/proc, coverage, peak và trường hợp nghi vấn. Tạo tự động không đồng nghĩa đã xem thủ công hết.',
            '- `review_manifest.json`: định nghĩa và giới hạn.','',
            'Nguồn: FloodNet (New York University and The City University of New York). Chỉ dùng cục bộ, không phân phối nguồn/cache/masks khi chưa có quyền. Phân tích/biểu diễn phái sinh được phép: CC BY-NC-SA 4.0 và điều kiện bổ sung của Data Access License Agreement.','']
    (OUT/'BAO_CAO_DOI_SOAT_18.md').write_text('\n'.join(lines),encoding='utf-8')
    cards=[]
    for r in records:
        imgs=[f'<h3>{html.escape(c["kind"])}</h3><img loading="lazy" src="{html.escape(c["path"])}" alt="TRAIN diagnostic">' for c in cases if c['deployment_id']==r['deployment_id']]
        imgs.insert(0,f'<img loading="lazy" src="plots/{r["deployment_id"]}_coverage.png" alt="TRAIN coverage">')
        cards.append(f'<section id="{r["deployment_id"]}"><h2>{html.escape(r["deployment_id"])} — {html.escape(str(r["name"]))}</h2><p>{html.escape(r["proposal"])}</p>'+''.join(imgs)+'</section>')
    (OUT/'index.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Đối soát 18 trạm</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px;background:#f4f6fa}section{background:white;padding:20px;margin:24px 0}img{width:100%}</style><h1>TRAIN-only review: 18 trạm</h1><p>Cờ nghi vấn không xác nhận nhiễu hoặc ngập. Không train model; không sửa dữ liệu nguồn. Xem báo cáo trước sử dụng.</p><p><a href="BAO_CAO_DOI_SOAT_18.md">Báo cáo</a> · <a href="station_review.csv">Bảng trạm</a></p>'+''.join(cards)+'</html>',encoding='utf-8')


if __name__=='__main__':
    main()
