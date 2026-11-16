"""Human-readable report; explicit denominators and retrospective-label limits."""
from delta_experiment import METRIC


def table(rows):
    lines = ['| Model | Hàng | Nhãn ≥10cm | MAE toàn tập | MAE ≥10cm | MAE cân bằng | FP | FN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        fmt = lambda v: 'N/A' if v is None else f'{v:.4f}'
        lines.append(f'| {r["model"]} | {r["windows"]} | {r["wet_targets_ge_10cm"]} | {fmt(r["mae_cm"])} | '
            f'{fmt(r["mae_wet_cm"])} | {fmt(r[METRIC])} | {r["fp"]} | {r["fn"]} |')
    return lines


def write_report(output, summary):
    common_validation = summary['paired_validation_metrics'][0]['windows'] if summary['paired_validation_metrics'] else 0
    proc_validation = sum(r['proc_windows'] for r in summary['coverage'] if r['split'] == 'validation')
    same_validation = common_validation == summary['counts']['validation'] == proc_validation
    lines = ['# Nhánh raw trên ba điểm: baseline và XGBoost delta', '',
        '**Đầu vào raw được xử lý chỉ theo quá khứ trên đồng hồ archive. Nhãn tương lai vẫn là proc hồi cứu. '
        'Chưa chứng minh hiệu quả vận hành thời gian thực hoặc chuyển sang ESP32.**', '',
        '## 1. Quy tắc chốt trước khi chuẩn bị/train', '',
        '- Ba điểm giữ nguyên: daily_new_falcon, ugliest_cyan_elephant, widely_proud_lizard. Không thêm bảy trạm hoặc xếp hạng lại.',
        '- Bin UTC (t−5,t]: trung vị raw hữu hạn, sau khi quy raw âm về 0. Chỉ hoàn tất bin tại t; không dùng mẫu sau t.',
        '- Bin rỗng hoặc raw ở hàng vật lý cuối là null/không hữu hạn: đánh dấu thiếu, không thay bằng số đo cũ.',
        '- Không nội suy/fill, không cắt trên, không loại nước cao/nhảy mạnh hoặc dùng mask offline. Ghi rõ số raw âm và mẫu cao còn giữ.',
        '- Trung vị giảm một phần xung nhiễu khi bin có nhiều mẫu; bin chỉ một mẫu và vật cản kéo dài vẫn có thể gây sai.',
        '- 12 feature có ý nghĩa như trước nhưng được tính từ raw-derived depth. Proc KHÔNG vào feature hoặc điều kiện lịch sử raw.',
        '- Nhãn vẫn là proc cuối bin (t+5,t+10], chỉ dùng làm giám sát tương lai. Nhãn không phải cực đại hoặc đúng tức thời t+10.',
        '- Giữ mốc 01/01/2023, 01/07/2023, embargo 24h và kiểm tra support đầy đủ 15 phút lịch sử/10 phút tương lai.',
        '- Không đọc test hoặc cache nguồn từ 07/2023 trở đi. Chỉ chọn bằng validation raw-eligible.', '',
        '## 2. Số cửa sổ thay đổi', '',
        '| Tập | Trạm | Proc cũ | Raw mới | Chung | Chỉ raw | Chỉ proc | Nhãn ≥10cm raw / proc |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['coverage']:
        lines.append(f'| {r["split"]} | {r["station"]} | {r["proc_windows"]} | {r["raw_windows"]} | '
            f'{r["common_windows"]} | {r["raw_only_windows"]} | {r["proc_only_windows"]} | '
            f'{r["raw_wet_ge_10cm"]} / {r["proc_wet_ge_10cm"]} |')
    lines += ['', f'Tổng cửa sổ raw: **{summary["counts"]}**.', '',
        'Các cửa sổ cũ bị mất, gồm cả nhãn cao nếu có, được giữ dấu vết trong `window_coverage.csv` và '
        '`window_membership_local.parquet`. Không bỏ vì model dự báo sai.', '',
        '### Lý do loại, phân nhóm không chồng nhau', '',
        '| Tập | Trạm | Lý do | Số bin quyết định |', '|---|---|---|---:|']
    for r in summary['exclusion_reasons']:
        lines.append(f'| {r["split"]} | {r["station"]} | {r["reason"]} | {r["windows"]} |')
    lines += ['', 'Ưu tiên lý do theo thứ tự: raw bin hiện tại thiếu → lag5 thiếu → lag10 thiếu → nhãn tương lai thiếu/âm → '
        'feature không hữu hạn → kept. Tổng các lý do bằng tổng bin trong support từng trạm/tập. '
        'Bin missing bao gồm khoảng không quan sát và null; xem tách riêng ở `raw_bin_audit.csv`.', '',
        '## 3. Validation chính: toàn bộ cửa sổ raw đủ điều kiện', '']
    lines += table(summary['validation_metrics'])
    lines += ['', f'Delta tốt nhất: **{summary["best_delta"]}**. Phương pháp được chọn trong cả baseline/AI: '
        f'**{summary["selected_forecaster"]}**. Không ép AI thắng.', '',
        '## 4. Đối chiếu raw/proc trên đúng cùng cửa sổ validation', '']
    lines += table(summary['paired_validation_metrics'])
    lines += ['', 'Các model ở bảng này được chốt TRƯỚC khi tính bảng đối chiếu. Nhãn tương lai ở các hàng chung khớp tuyệt đối. '
        'Chỉ đối chiếu điểm raw/proc trực tiếp trên giao cửa sổ này; kiểm tra bảng coverage trước khi coi toàn validation hai nhánh là cùng mẫu số. ' +
        (f'Lượt này: {common_validation:,} hàng chung; raw {summary["counts"]["validation"]:,}, proc {proc_validation:,}. '
         + ('Toàn bộ validation trùng nhau.' if same_validation else 'Validation hai nhánh không hoàn toàn trùng nhau.')), '',
        '**Giới hạn so sánh:** lịch sử raw/proc và số hàng train có thể khác. Giao validation loại khác biệt mẫu số đánh giá, '
        'nhưng không biến đây thành thí nghiệm chỉ thay đúng một biến trong toàn bộ quá trình học. '
        '`new_crossing` nếu xem trong CSV là tương đối với mức hiện tại của từng pipeline, không phải cùng nhãn khởi phát độc lập.', '',
        '## 5. Kiểm tra và giới hạn', '',
        '- Cả bốn model lưu/nạp lại khớp mọi dự đoán validation. Kiểm tra độc lập ở `validate_raw.py`.',
        '- Prefix replay trên train kiểm tra feature cùng thời điểm không thay đổi khi có thêm dữ liệu tương lai.',
        '- CSV chỉ có timestamp archive, không có thời điểm nhận raw. Chưa mô phỏng độ trễ mạng hay xác nhận hiệu chuẩn nguồn.',
        '- Nhãn proc có thể qua xử lý dùng mẫu sau. Dùng làm nhãn hồi cứu không đưa nó vào đầu vào, nhưng không xác minh nhãn đúng.',
        '- Việc chọn ba trạm ban đầu dựa trên train proc; không tuyên bố việc chọn trạm hoàn toàn độc lập proc.',
        '- Raw âm quy về 0 chỉ là quy ước đầu vào, không sửa mốc hiệu chuẩn hoặc chứng minh đường khô.',
        '- Các cửa sổ tương quan, validation được xem nhiều lần; cần đánh giá theo sự kiện và kiểm định độc lập.',
        '- Không sửa backend/frontend, không đưa model vào cảnh báo/định tuyến hoặc thay thế luật ngưỡng.', '',
        '## 6. Artifacts', '',
        '- `config_before_preparation.json`, `config_before_fit.json`: chính sách/cấu hình chốt trước thực nghiệm.',
        '- `source_provenance.json`, `proc_window_provenance.json`, `window_provenance.json`: nguồn và checksum.',
        '- `windows/`, `window_membership_local.parquet`: cửa sổ và dấu vết khác biệt với nhánh proc, giữ cục bộ.',
        '- `raw_delta_*.ubj`, `selection_validation_only.json`, `metrics.csv`, `summary.json`: model và lựa chọn.',
        '- `paired_validation_metrics.csv`, `validation_breakdown.csv`: đối chiếu/phân tầng, không lọc ca khó.', '',
        'Nguồn: FloodNet (New York University and The City University of New York). '
        'Giữ mọi dữ liệu/đầu ra/trọng số cục bộ; làm rõ quyền trước phân phối. '
        'Phân tích/biểu diễn phái sinh được phép theo CC BY-NC-SA 4.0 cùng Data Access License Agreement.', '']
    (output / 'BAO_CAO_RAW.md').write_text('\n'.join(lines), encoding='utf-8')
