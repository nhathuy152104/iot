"""Report both arms and all quality warnings without claiming clean labels."""
from delta_experiment import METRIC


def write_report(output, summary):
    lines = ['# Mở rộng điểm train cho nhánh raw', '',
        '**Thử nghiệm hồi cứu, không xác nhận nhãn sạch hoặc phê duyệt cảnh báo/định tuyến.**', '',
        '## Thiết kế', '',
        '- Giữ nguyên nhánh raw ba điểm, preprocessing 5 phút, nhãn proc tương lai, 12 feature, bốn cấu hình XGBoost delta và mốc chia tập/embargo.',
        '- Chọn điểm bổ sung CHỈ theo train: ≥1.000 cửa sổ, ≥30 ngày, ≥20 nhãn cao, ≥1.000 nhãn thấp, ≥3 ngày và ≥3 nhóm nhãn cao cách nhau hơn 6 giờ.',
        '- Nhóm 6 giờ không phải các trận mưa độc lập. Đạt độ phủ không đồng nghĩa ngập thật hoặc nhãn sạch.',
        '- Đọc cache ứng viên trước 01/01/2023, không đọc validation của điểm mới hoặc test. Cảnh báo chất lượng không bị giấu.',
        '- Với điểm đạt độ phủ, giữ TẤT CẢ cửa sổ raw hợp lệ, kể cả ca cao/nhảy mạnh/nghi vật cản; không dùng mask offline hoặc chỉ lấy ca đẹp.',
        '- Đây là thử nghiệm bổ sung toàn bộ cửa sổ đủ điều kiện, KHÔNG nâng bảy điểm thành core đã xác minh sạch.', '',
        f'Số hàng: **{summary["counts"]}**. Bổ sung **{len(summary["selected_additions"])}** điểm, '
        f'tổng **{len(summary["core_stations"]) + len(summary["selected_additions"])}** mã triển khai train (`deployment_id`), không mặc định là từng ấy vị trí độc lập.', '',
        '## Kiểm kê ứng viên — chỉ train', '',
        '| Điểm | Cửa sổ | Nhãn ≥10cm | Ngày nhãn cao | Nhóm 6h | Tỷ lệ lịch sử raw âm | Chọn | Lý do không đạt |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for r in summary['station_audits_train_only']:
        negative = 'N/A' if r['raw_negative_history_fraction'] is None else f'{100*r["raw_negative_history_fraction"]:.1f}%'
        lines.append(f'| {r["station"]} | {r["windows"]} | {r["wet_ge_10cm"]} | {r["wet_days"]} | '
            f'{r["wet_temporal_groups_6h"]} | {negative} | {r["selected_for_exploratory_expansion"]} | {", ".join(r["failed_criteria"])} |')
    lines += ['', 'Raw âm đã quy về 0 theo chính sách nhánh raw cũ, không tự kết luận là lỗi. '
        'Các điểm này từng có dấu hiệu mức cao kéo dài, lệch mốc hoặc dao động mạnh; '
        '`station_audit_train.csv` ghi thêm sparse/mismatch và các cờ, không loại cửa sổ chỉ vì cờ.', '',
        '## Validation cố định — mọi phương pháp trên đúng cùng hàng/nhãn', '',
        '| Model | MAE toàn tập | MAE ≥10cm | MAE cân bằng | FP | FN |', '|---|---:|---:|---:|---:|---:|']
    for r in summary['validation_metrics']:
        lines.append(f'| {r["model"]} | {r["mae_cm"]:.4f} | {r["mae_wet_cm"]:.4f} | {r[METRIC]:.4f} | {r["fp"]} | {r["fn"]} |')
    lines += ['', f'Ba điểm tốt nhất: **{summary["best_three"]}**. Nhóm mở rộng tốt nhất: **{summary["best_expanded"]}**.',
        f'Phương pháp được chọn trong cả baseline/hai nhóm: **{summary["selected_forecaster"]}**.',
        f'Chênh lệch MAE cân bằng expanded − three: **{summary["expanded_vs_three_balanced_mae_change_cm"]:+.6f} cm** (âm tốt hơn).', '',
        'Không ép kết luận thêm dữ liệu là tốt hơn. Không thay tiêu chí sau khi xem kết quả; FP/FN vẫn phải đọc riêng.', '',
        '## So từng cấu hình giống nhau', '',
        '| Cấu hình | MAE cân bằng 3 điểm | Mở rộng | Thay đổi cm | FP 3 / mở rộng | FN 3 / mở rộng |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary['matched_config_comparison']:
        lines.append(f'| {r["config"]} | {r["three_balanced_mae_cm"]:.4f} | {r["expanded_balanced_mae_cm"]:.4f} | '
            f'{r["change_cm"]:+.4f} | {r["three_fp"]} / {r["expanded_fp"]} | {r["three_fn"]} / {r["expanded_fn"]} |')
    lines += ['', '## Giới hạn', '',
        '- Validation vẫn chỉ có hai trạm core cũ, không chứng minh dự báo tốt ở các trạm mới.',
        '- Validation được dùng nhiều lần; không là bộ xác nhận độc lập. Không nạp test để điều chỉnh.',
        '- Lịch sử raw chỉ dùng quá khứ theo timestamp archive, chưa tái dựng độ trễ nhận hoặc hiệu chuẩn nguồn.',
        '- Nhãn proc tương lai là tham chiếu hồi cứu chưa xác nhận hiện trường; không phải dự báo đúng tức thời t+10.',
        '- Các cửa sổ tương quan; FP/FN không phải số đợt cảnh báo độc lập.',
        '- Không sửa cảnh báo theo ngưỡng, backend/frontend hoặc dữ liệu/artifacts cũ.', '',
        '## Artifacts', '',
        '- `config_before_preparation.json`, `station_selection_before_fit.json`, `config_before_fit.json`: chính sách/quyết định trước fit.',
        '- `station_audit_train.csv`, `candidate_exclusion_reasons.csv`, `candidate_train/`: kiểm kê và tất cả cửa sổ ứng viên, không giấu ca khó.',
        '- `frozen_input_provenance.json`, `source_provenance.json`, `candidate_window_provenance.json`: nguồn/checksum.',
        '- `expanded_*.ubj`, `metrics.csv`, `selection_validation_only.json`, `validation_predictions_local.parquet`: bốn model mới và điểm trên validation cố định.',
        '- `matched_config_comparison.csv`, `validation_breakdown.csv`: so từng cấu hình và từng trạm.', '',
        'Nguồn: FloodNet (New York University and The City University of New York). Giữ artifacts cục bộ; '
        'không phân phối dữ liệu/trọng số khi chưa làm rõ quyền. Phân tích/biểu diễn phái sinh được phép theo '
        'CC BY-NC-SA 4.0 cùng Data Access License Agreement.', '']
    (output / 'BAO_CAO_MO_RONG_RAW.md').write_text('\n'.join(lines), encoding='utf-8')
